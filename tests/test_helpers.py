"""Tests for `verilog_ml_benchmark_generator` pyMTL Components."""
import numpy
import pytest
import random
import math
from pymtl3 import *

from click.testing import CliRunner

from verilog_ml_benchmark_generator import utils
from verilog_ml_benchmark_generator import module_classes
from verilog_ml_benchmark_generator import cli

def merge_bus(v,width):
    sum = 0
    for i in range(len(v)):
        sum += v[i] * (2 ** (width * i))
    return sum

def load_buffers(testinst, we_portname, addr_portname, datain_portname, buffer_values, dwidth, outertestinst=None):
    #eg. ibuf = [[[1
    #         for k in range(ivalues_per_buf)]            # values per word
    #         for i in range(ibuf_len)]                   # words per buffer
    #         for j in range (ibuf_count)]                # buffers
    for j in range(len(buffer_values)):  # For each buffer...
        curr_we = we_portname.format(j)
        load_buffer(testinst, curr_we, addr_portname, datain_portname, buffer_values[j], dwidth, outertestinst)

def load_buffer(testinst, we_portname, addr_portname, datain_portname, buffer_values, dwidth, outertestinst=None):
    if not outertestinst:
        outertestinst = testinst
    curr_we = getattr(testinst, we_portname)
    addr_port = getattr(testinst, addr_portname)
    datain_port = getattr(testinst, datain_portname)
    curr_we @= 1
    for i in range(len(buffer_values)):
        addr_port @= i
        datain_port @= merge_bus(buffer_values[i], dwidth)
        outertestinst.sim_tick()
    curr_we @= 0

def load_buffers_sm(testinst, datain_portname, buffer_values, dwidth, outertestinst=None):
    if not outertestinst:
        outertestinst = testinst
    datain_port = getattr(testinst, datain_portname)
    for j in range(len(buffer_values)):  # For each buffer...
        for i in range(len(buffer_values[j])):
            outertestinst.sim_tick()
            datain_port @= merge_bus(buffer_values[j][i], dwidth)
            testinst.sm_start @= 0
            #print("SM: buf[{}] = {} ({})".format(int(j),int(datain_port),buffer_values[j][i]))
            #print("ENs: w:{} i0:{} i1:{} i2:{} i3:{} i4:{}".format(testinst.datapath.weight_modules_portawe_0_top, testinst.datapath.input_act_modules_portawe_0_top, testinst.datapath.input_act_modules_portawe_1_top, testinst.datapath.input_act_modules_portawe_2_top, testinst.datapath.input_act_modules_portawe_3_top, testinst.datapath.input_act_modules_portawe_4_top))
    outertestinst.sim_tick()

def check_buffers(testinst, outer_inst, inner_inst_name, buffer_values, dwidth, outertestinst=None):
    for j in range(len(buffer_values)):
        inner_inst = getattr(outer_inst, inner_inst_name.format(j))
        check_buffer(testinst, inner_inst, buffer_values[j], dwidth, outertestinst)
                
def load_mlb_values(testinst, chainlen, chainstart, buflen, addr_portname, en_portname, outertestinst=None):
    if not outertestinst:
        outertestinst = testinst
    addr_port = getattr(testinst, addr_portname)
    en_port = getattr(testinst, en_portname)
    en_port @= 1
    for i in range(chainlen):
        prev_addr = (chainstart+i) % buflen
        addr_port @= prev_addr
        outertestinst.sim_tick()
    en_port @= 0
        
def check_mlb_chain_values(testinst,
                           mlb_count, mac_count,
                           mlb_start, mac_start,
                           mlb_name, weight_out_name, 
                           buffer_values, dwidth, i=0,
                           bo_chain_len=0, ubo=0,
                           bi_chain_len=0, ubi=0):
    buflen = len(buffer_values)
    all_good = True
    for t in range(mlb_start+mlb_count-1,mlb_start-1,-1):
        
        for r in range(mac_count-1,-1,-1):
            mac_idx = t*mac_count + r
            curr_mlb = getattr(testinst.mlb_modules, mlb_name.format(t))
            curr_out = getattr(curr_mlb.sim_model.mac_modules, weight_out_name.format(r))
            # Outer weight buffer index
            total_mac_idx = mac_count*mlb_count-mac_idx-1
            #print(total_mac_idx)
            buffer_idxo = math.floor(total_mac_idx/(bo_chain_len*ubo*ubi))
            buffer_idxo = buffer_idxo*bo_chain_len
            #print(buffer_idxo)
            # Inner weight buffer index
            buffer_idxi = math.floor((total_mac_idx%(bo_chain_len*ubi))/(bi_chain_len*ubi))
            buffer_idxi = buffer_idxi*bi_chain_len
            #print(buffer_idxi)
            # weight index
            buf_item_idx = (total_mac_idx%bi_chain_len)
            all_good &= (curr_out == buffer_values[(buffer_idxo + buffer_idxi + buf_item_idx)% buflen][i])
    return all_good

def check_mlb_chains_values(testinst,
                            mlb_count, mac_count,
                            outer_dwidth, inner_dwidth,
                            mlb_name, weight_out_name, 
                            buffer_values, dwidth,
                            bo_chain_len=0, ubo=0,
                            bi_chain_len=0, ubi=0):
    part_mlb_count = math.ceil(mlb_count/outer_dwidth)
    part_mac_count = math.ceil(mac_count/inner_dwidth)
    all_good = True
    for t in range(outer_dwidth):
        mlb_start_i = t*math.ceil(mlb_count/len(buffer_values))
        for r in range(inner_dwidth):
            stream_idx = t*inner_dwidth+r
            buf_idx = math.floor(stream_idx/len(buffer_values))
            part_idx = stream_idx % len(buffer_values)
            mac_start_i = r*part_mac_count
            all_good &= check_mlb_chain_values(testinst,
                                   part_mlb_count, part_mac_count,
                                   mlb_start_i, mac_start_i,
                                   mlb_name,
                                   weight_out_name, 
                                   buffer_values[buf_idx], dwidth, part_idx, bo_chain_len, ubo, bi_chain_len, ubi)
    return all_good

def stream_mlb_values(testinst, time, addr_portnames, os, buf_lens, en_portnames, starti = 0, outertestinst=None):
    if not outertestinst:
        outertestinst = testinst
    for en_portname in en_portnames:
        en_port = getattr(testinst, en_portname)
        en_port @= 1
    for i in range(starti, time+starti):
        for r in range(len(addr_portnames)):
            addr_port = getattr(testinst, addr_portnames[r])
            addr = i % buf_lens[r]
            addr_port @= addr + os[r]
        outertestinst.sim_tick()
    for en_portname in en_portnames:
        en_port = getattr(testinst, en_portname)
        en_port @= 0
    return i

def get_expected_outputs(obuf, ostreams_per_buf, wbuf, ibuf, ivalues_per_buf, projection):
    inner_uw = projection["inner_projection"]["URW"]["value"]
    inner_un = projection["inner_projection"]["URN"]["value"]
    inner_ue = projection["inner_projection"]["UE"]["value"]
    inner_ub = projection["inner_projection"]["UB"]["value"]
    inner_ug = projection["inner_projection"]["UG"]["value"]
    outer_uw = projection["outer_projection"]["URW"]["value"]
    outer_un = projection["outer_projection"]["URN"]["value"]
    outer_ue = projection["outer_projection"]["UE"]["value"]
    outer_ub = projection["outer_projection"]["UB"]["value"]
    outer_ug = projection["outer_projection"]["UG"]["value"]
    mlb_count = utils.get_mlb_count(projection["outer_projection"])
    mac_count = utils.get_mlb_count(projection["inner_projection"])
    obuf_len = len(obuf[0])
    wbuf_len = len(wbuf[0])
    ibuf_len = len(ibuf[0])
    for i in range(obuf_len):
        for ugo in range(outer_ug): 
            for ugi in range(inner_ug):
                for ubo in range(outer_ub): 
                    for ubi in range(inner_ub):
                        for ueo in range(outer_ue):
                            for uei in range(inner_ue):
                                correct_sum = 0
                                for urno in range(outer_un):
                                    for urni in range(inner_un):
                                        for urwo in range(outer_uw):
                                            for urwi in range(inner_uw):
                                                urw = urwo*inner_uw + urwi
                                                mlb_inst = ugo*outer_ub*outer_ue*outer_uw*outer_un + \
                                                           ubo*outer_ue*outer_uw*outer_un + \
                                                           ueo*outer_uw*outer_un + \
                                                           urno*outer_uw + \
                                                           urwo
                                                mac_idx = mlb_inst*mac_count + \
                                                          ugi*inner_ub*inner_ue*inner_uw*inner_un + \
                                                          ubi*inner_ue*inner_uw*inner_un + \
                                                          uei*inner_uw*inner_un + \
                                                          urni*inner_uw + \
                                                          urwi
                                                w_buf_inst_idx = \
                                                    ugo*outer_ue*outer_uw*outer_un*inner_ug*inner_ue*inner_un*inner_uw + \
                                                    ueo*outer_uw*outer_un*inner_ug*inner_ue*inner_un*inner_uw + \
                                                    urwo*outer_un*inner_ug*inner_ue*inner_un*inner_uw + \
                                                    urno*inner_ug*inner_ue*inner_un*inner_uw + \
                                                    ugi*inner_ue*inner_un*inner_uw + \
                                                    uei*inner_un*inner_uw + \
                                                    urni*inner_uw + \
                                                    urwi
                                                buffer_idx = (outer_ug*outer_ue*outer_uw*outer_un*inner_ug*\
                                                              inner_ue*inner_un*inner_uw - w_buf_inst_idx - 1)\
                                                              % wbuf_len
                                                w = merge_bus(wbuf[0][buffer_idx % wbuf_len],
                                                            projection["stream_info"]["W"])
                                                if ((i - urw) >= 0) and \
                                                   ((i - urw) < wbuf_len):
                                                    i_stream_idx = (outer_ub*outer_un*ugo + \
                                                                    ubo*outer_un + \
                                                                    urno)
                                                    i_value_idx = i_stream_idx*utils.get_proj_stream_count(projection["inner_projection"], 'I') + \
                                                                  (inner_ub*inner_un*ugi + \
                                                                   ubi*inner_un + \
                                                                   urni)
                                                    ibuf_idx = math.floor(i_value_idx / ivalues_per_buf)
                                                    iv_idx = i_value_idx % ivalues_per_buf
                                                    correct_sum += (ibuf[ibuf_idx][(i - urw)%ibuf_len][iv_idx] * w)
                                out_act_idx = ugo*outer_ub*outer_ue*inner_ug*inner_ub*inner_ue + \
                                              ubo*outer_ue*inner_ug*inner_ub*inner_ue + \
                                              ueo*inner_ug*inner_ub*inner_ue + \
                                              ugi*inner_ub*inner_ue + \
                                              ubi*inner_ue + \
                                              uei
                                obuf_idx = math.floor(out_act_idx/ostreams_per_buf)
                                os_idx = out_act_idx % ostreams_per_buf
                                obuf[obuf_idx][i][os_idx] = correct_sum%(2**projection["stream_info"]["I"])
    return obuf


def read_out_stored_buffer_values(testinst, inner_inst, addr_portname, dataout_portname,
                                buffer_values, dwidth, outertestinst=None):
    if not outertestinst:
        outertestinst = testinst
    addr_port = getattr(outertestinst, addr_portname)
    #dataout_port = getattr(inner_inst, dataout_portname)
    try:
        dataout_port = getattr(outertestinst, dataout_portname)
    except:
        dataout_port = getattr(inner_inst, dataout_portname)
        
    for i in range(len(buffer_values)):
        dataout_val = getattr(inner_inst.sim_model_inst0,"V"+str(i))
        addr_port @= i
        outertestinst.sim_tick()
        curr_obuf_out = int(dataout_port)
        assert(curr_obuf_out == dataout_val.dataout)
        for section in range(len(buffer_values[i])):
            buffer_values[i][section] = int(curr_obuf_out%(2**dwidth))
            curr_obuf_out = math.floor(curr_obuf_out / (2**dwidth))
    return buffer_values

def read_out_stored_buffer_values_from_sm(dataout_portname,
                                buffer_values, dwidth, outertestinst):
    dataout_port = getattr(outertestinst, dataout_portname)
    for i in range(len(buffer_values)):
        outertestinst.sim_tick()
        curr_obuf_out = int(dataout_port)
        for section in range(len(buffer_values[i])):
            buffer_values[i][section] = int(curr_obuf_out%(2**dwidth))
            curr_obuf_out = math.floor(curr_obuf_out / (2**dwidth))
    return buffer_values

               
def read_out_stored_values(testinst, addr_portname, dataout_portname,
                         buffer_values, dwidth, outertestinst=None):
    if outertestinst:
        dataout = dataout_portname
    else:
        dataout = "portadataout"
    for obufi in range(len(buffer_values)):
        print("OBUF = " + str(int(obufi)))
        curr_obuf = getattr(testinst.output_act_modules, "mlb_outs_inst_" + str(obufi))
        read_out_stored_buffer_values(testinst, curr_obuf, addr_portname, dataout,
                                    buffer_values[obufi], dwidth, outertestinst)
    return buffer_values

def read_out_stored_values_from_sm(dataout_portname,
                         buffer_values, dwidth, outertestinst=None):
    for obufi in range(len(buffer_values)):
        read_out_stored_buffer_values_from_sm(dataout_portname,
                                    buffer_values[obufi], dwidth, outertestinst)
    return buffer_values

def gather_stored_buffer_values(testinst, inner_inst,
                                buffer_values, dwidth, outertestinst=None):
    if not outertestinst:
        outertestinst = testinst
    for i in range(len(buffer_values)):
        curr_obuf= getattr(inner_inst.sim_model_inst0,"V"+str(i))
        curr_obuf_out = int(curr_obuf.dataout)
        for section in range(len(buffer_values[i])):
            buffer_values[i][section] = int(curr_obuf_out%(2**dwidth))
            curr_obuf_out = math.floor(curr_obuf_out / (2**dwidth))
    return buffer_values

def check_buffer(testinst, inner_inst, buffer_values, dwidth, outertestinst=None):
    new_buf = [[0 for i in range(len(buffer_values[0]))] for j in range(len(buffer_values))]
    new_buf = gather_stored_buffer_values(testinst, inner_inst,
                                          new_buf, dwidth, outertestinst)
    print("STORED_VALUES")
    print("SM: ={}".format(inner_inst.sim_model_inst0.data))
    print(new_buf)
    print("EXPECTED_VALUES")
    print(buffer_values)
    assert(new_buf == buffer_values)
