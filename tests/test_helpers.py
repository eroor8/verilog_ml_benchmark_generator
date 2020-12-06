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

def check_mac_weight_values(proj_yaml, curr_mlb,
                           weight_out_name, 
                           buffer_values,
                           mlb_start_addr, i=0):
    print(proj_yaml["inner_projection"])
    mac_count = utils.get_mlb_count(proj_yaml["inner_projection"])
    if ("PRELOAD" in proj_yaml["inner_projection"]):
        bi_chain_len = proj_yaml["inner_projection"]["UE"]["value"] * \
                            proj_yaml["inner_projection"]["URN"]["value"] * \
                            proj_yaml["inner_projection"]["URW"]["value"]
        ubi = proj_yaml["inner_projection"]["UB"]["value"]
        buflen = len(buffer_values)
        for r in range(mac_count-1,-1,-1):
            curr_out = getattr(curr_mlb.sim_model.mac_modules, weight_out_name.format(r))
            mac_idx = mac_count-r-1
            
            # Calculate expected buffer value
            buffer_idxi = math.floor(mac_idx/(bi_chain_len*ubi))*bi_chain_len
            buf_item_idx = (mac_idx%bi_chain_len)
            assert (curr_out == buffer_values[(mlb_start_addr + buffer_idxi + buf_item_idx)% buflen][i])
    else:
        print(buffer_values)
        for ugi in range(proj_yaml["inner_projection"]["UG"]["value"]):
            for ubi in range(proj_yaml["inner_projection"]["UB"]["value"]):
                for uei in range(proj_yaml["inner_projection"]["UE"]["value"]):
                    for uni in range(proj_yaml["inner_projection"]["URN"]["value"]):
                        for uwi in range(proj_yaml["inner_projection"]["URW"]["value"]):
                            mac_idx = utils.get_overall_idx(proj_yaml["inner_projection"],
                                {'URN': uni, 'UB': ubi, 'UG': ugi, 'UE': uei, 'URW':uwi})
                            curr_out = getattr(curr_mlb.sim_model.mac_modules,
                                               weight_out_name.format(mac_idx))
                            stream_idx = utils.get_overall_idx(proj_yaml["inner_projection"],
                                {'URN': uni, 'UG': ugi, 'UE': uei, 'URW':uwi})
                            assert (curr_out ==
                                    buffer_values[mlb_start_addr % len(buffer_values)][i+stream_idx])
    return True

def check_mlb_chain_values(testinst,
                           mlb_count, mac_count,
                           mlb_start, mac_start,
                           mlb_name, weight_out_name, 
                           buffer_values, dwidth, i=0,
                           bo_chain_len=0, ubo=0,
                           bi_chain_len=0, ubi=0, proj_yaml={}):
    buflen = len(buffer_values)
    all_good = True
    for t in range(mlb_start+mlb_count-1,mlb_start-1,-1):
        curr_mlb = getattr(testinst.mlb_modules, mlb_name.format(t))
        if (len(proj_yaml) > 0):
            total_mac_idx = mac_count*mlb_count-t*mac_count-1
            buffer_idxo = math.floor(total_mac_idx/(bo_chain_len*ubo*ubi))
            buffer_idxo = buffer_idxo*bo_chain_len
            
            ugi = proj_yaml["inner_projection"]["UG"]["value"]
            buffer_idxi = math.floor((total_mac_idx%(bo_chain_len*ubi))/(bi_chain_len*ubi*ugi))
            buffer_idxi = buffer_idxi*bi_chain_len*ugi
            values_per_stream = utils.get_proj_stream_count(
                                proj_yaml["inner_projection"], 'W')
            all_good &= check_mac_weight_values(proj_yaml, curr_mlb,
                           weight_out_name, 
                           buffer_values,
                           int((buffer_idxo + buffer_idxi)//values_per_stream), i=0)
        else:
            total_mac_idx = mac_count*mlb_count-t*mac_count-1
            buffer_idxo = math.floor(total_mac_idx/(bo_chain_len*ubo*ubi))
            buffer_idxo = buffer_idxo*bo_chain_len

            for r in range(mac_count-1,-1,-1):
                mac_idx = t*mac_count + r
                curr_out = getattr(curr_mlb.sim_model.mac_modules, weight_out_name.format(r))
                
                # Inner weight buffer index
                total_mac_idx = mac_count*mlb_count-mac_idx-1
                buffer_idxi = math.floor((total_mac_idx%(bo_chain_len*ubi))/(bi_chain_len*ubi))
                buffer_idxi = buffer_idxi*bi_chain_len
                buf_item_idx = (total_mac_idx%bi_chain_len)
                # weight index
                all_good &= (curr_out ==
                             buffer_values[(buffer_idxo + buffer_idxi + buf_item_idx)% buflen][i])

    return all_good

def check_mlb_chains_values(testinst,
                            mlb_count, mac_count,
                            outer_dwidth, inner_dwidth,
                            mlb_name, weight_out_name, 
                            buffer_values, dwidth,
                            bo_chain_len=0, ubo=0,
                            bi_chain_len=0, ubi=0, proj_yaml={}):
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
                                   buffer_values[buf_idx], dwidth,
                                   part_idx, bo_chain_len, ubo, bi_chain_len, ubi, proj_yaml)
    return all_good


def check_weight_contents(testinst, proj_yaml, mlb_name, weight_out_name, 
                            buffer_values):
    mlb_count = utils.get_mlb_count(proj_yaml["outer_projection"])
    mac_count = utils.get_mlb_count(proj_yaml["inner_projection"])
    bi_chain_len = proj_yaml["inner_projection"]["UE"]["value"] * \
                            proj_yaml["inner_projection"]["URN"]["value"] * \
                            proj_yaml["inner_projection"]["URW"]["value"]
    inner_ub = proj_yaml["inner_projection"]["UB"]["value"]
    print(buffer_values)
    if ("PRELOAD" in proj_yaml["outer_projection"]):
        outer_ub = proj_yaml["outer_projection"]["UB"]["value"]
        bo_chain_len = proj_yaml["outer_projection"]["UE"]["value"] * \
                            proj_yaml["outer_projection"]["URN"]["value"] * \
                            proj_yaml["outer_projection"]["URW"]["value"] *\
                            proj_yaml["inner_projection"]["UG"]["value"] *  bi_chain_len
        
        # Calculate required buffers etc.
        return check_mlb_chains_values(testinst,
                                mlb_count, mac_count,
                                1,1,
                                mlb_name, weight_out_name, 
                                buffer_values, proj_yaml["stream_info"]["W"],
                                bo_chain_len, outer_ub,
                                bi_chain_len, inner_ub, proj_yaml)
    else:
        buflen = len(buffer_values)
        print(proj_yaml["outer_projection"])
        for ugo in range(proj_yaml["outer_projection"]["UG"]["value"]):
            for ubo in range(proj_yaml["outer_projection"]["UB"]["value"]):
                for ueo in range(proj_yaml["outer_projection"]["UE"]["value"]):
                    for uno in range(proj_yaml["outer_projection"]["URN"]["value"]):
                        for uwo in range(proj_yaml["outer_projection"]["URW"]["value"]):
                            mlb_idx = utils.get_overall_idx(proj_yaml["outer_projection"],
                                {'URN': uno, 'UB': ubo, 'UG': ugo, 'UE': ueo, 'URW':uwo})
                            curr_mlb = getattr(testinst.mlb_modules, mlb_name.format(mlb_idx))
                            stream_idx = utils.get_overall_idx(proj_yaml["outer_projection"],
                                {'URN': uno, 'UG': ugo, 'UE': ueo, 'URW':uwo})
                            print("MLB: " + str(mlb_idx))
                            values_per_stream = utils.get_proj_stream_count(
                                proj_yaml["inner_projection"], 'W')
                            streams_per_buf = len(buffer_values[0][0]) / values_per_stream
                            buffer_idx = int(stream_idx // streams_per_buf)
                            buffer_stream_idx = int(stream_idx % streams_per_buf)
                            print("Stream: " + str(stream_idx) + " -- B" + str(buffer_idx) + " values " + 
                                  str(buffer_stream_idx*values_per_stream) + " to " +
                                  str((buffer_stream_idx+1)*values_per_stream-1))
                            print("Check MLB " + str(curr_mlb))
                            check_mac_weight_values(proj_yaml, curr_mlb, 
                                weight_out_name, buffer_values[buffer_idx],
                                mlb_start_addr=0,
                                i=buffer_stream_idx)
                            return True
        

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
        assert (curr_obuf_out == dataout_val.dataout)
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
            curr_obuf_out = curr_obuf_out // (2**dwidth)
    return buffer_values

def check_buffer(testinst, inner_inst, buffer_values, dwidth, outertestinst=None):
    new_buf = [[0 for i in range(len(buffer_values[0]))] for j in range(len(buffer_values))]
    new_buf = gather_stored_buffer_values(testinst, inner_inst,
                                          new_buf, dwidth, outertestinst)
    print("STORED_VALUES")
    print(new_buf)
    print("EXPECTED_VALUES")
    print(buffer_values)
    assert (new_buf == buffer_values), "Invalid contents for buffer " + str(inner_inst)
