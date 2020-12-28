"""Tests for `verilog_ml_benchmark_generator` pyMTL Components."""
import numpy
import pytest
import random
import math
import os
import sys
from pymtl3 import *

from click.testing import CliRunner

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
import module_classes
import cli

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
   
def reorder_input_array(inputs, proj_yaml, ab_yaml, obuf_len):
    ivalues_per_buf, ibuf_len, ibuf_count = utils.get_iw_buffer_dimensions(
        ab_yaml, proj_yaml, 'I')
    #assert(ibuf_count == 2)
    inner_ug = proj_yaml["inner_projection"]["UG"]["value"]
    outer_ug = proj_yaml["outer_projection"]["UG"]["value"]
    temp_ug = proj_yaml.get("temporal_projection",{}).get("UG",{}).get("value", 1)
    
    inner_ub = proj_yaml["inner_projection"]["UB"]["value"]
    inner_ubb = proj_yaml["inner_projection"]["UB"]["batches"]
    inner_ubx = proj_yaml["inner_projection"]["UB"]["x"]
    inner_uby = proj_yaml["inner_projection"]["UB"]["y"]
    outer_ub = proj_yaml["outer_projection"]["UB"]["value"]
    outer_ubb = proj_yaml["outer_projection"]["UB"]["batches"]
    outer_ubx = proj_yaml["outer_projection"]["UB"]["x"]
    outer_uby = proj_yaml["outer_projection"]["UB"]["y"]
    temp_ub = proj_yaml.get("temporal_projection",{}).get("UB",{}).get("value", obuf_len)
    temp_ubb = proj_yaml.get("temporal_projection",{}).get("UB",{}).get("batches", 1)
    temp_ubx = proj_yaml.get("temporal_projection",{}).get("UB",{}).get("x", obuf_len)
    temp_uby = proj_yaml.get("temporal_projection",{}).get("UB",{}).get("y", 1)

    inner_un = proj_yaml["inner_projection"]["URN"]["value"]
    inner_unc = proj_yaml["inner_projection"]["URN"]["chans"]
    inner_unx = proj_yaml["inner_projection"]["URN"]["x"]
    inner_uny = proj_yaml["inner_projection"]["URN"]["y"]
    outer_un = proj_yaml["outer_projection"]["URN"]["value"]
    outer_unc = proj_yaml["outer_projection"]["URN"]["chans"]
    outer_unx = proj_yaml["outer_projection"]["URN"]["x"]
    outer_uny = proj_yaml["outer_projection"]["URN"]["y"]
    temp_un = proj_yaml.get("temporal_projection",{}).get("URN",{}).get("value", 1)
    temp_unc = proj_yaml.get("temporal_projection",{}).get("URN",{}).get("chans", 1)
    temp_unx = proj_yaml.get("temporal_projection",{}).get("URN",{}).get("x", 1)
    temp_uny = proj_yaml.get("temporal_projection",{}).get("URN",{}).get("y", 1)
    
    ibuf = [[[0 for k in range(ivalues_per_buf)]
             for i in range(ibuf_len)]           
             for j in range (ibuf_count)]
    print(inputs)
    div_factor = (outer_uny*inner_uny)
    for ugt in range(temp_ug):
        for ugo in range(outer_ug): 
            for ugi in range(inner_ug):
                for ubox in range(outer_ubx):
                    for ubix in range(inner_ubx):
                        for ubtx in range(temp_ubx):
                            for uboy in range(outer_uby):
                                for ubiy in range(inner_uby):
                                    for ubty in range(0,temp_uby,div_factor):
                                        for ubob in range(outer_ubb):
                                            for ubib in range(inner_ubb):
                                                for ubtb in range(temp_ubb):
                                                    for urnoc in range(outer_unc):
                                                        for urnic in range(inner_unc):
                                                            for urntc in range(temp_unc):
                                                                for urnoy in range(outer_uny):
                                                                    for urniy in range(inner_uny):
                                                                        for urnty in range(temp_uny):
                                                                            groups = ugt*outer_ug*inner_ug+ugo*inner_ug+ugi
                                                                            batches = ubtb*outer_ubb*inner_ubb+ubob*inner_ubb+ubib
                                                                            channels = urntc*outer_unc*inner_unc+urnoc*inner_unc+urnic
                                                                            uby = ubty*outer_uby*inner_uby+uboy*inner_uby+ubiy
                                                                            uny = urnty*outer_uny*inner_uny+urnoy*inner_uny+urniy
                                                                            uy = inner_uny*outer_uny*temp_uny
                                                                            ubx = ubtx*outer_ubx*inner_ubx+ubox*inner_ubx+ubix
                                                                            
                                                                            print(inputs[groups][batches][channels]\
                                                                                      [(uby+uny)%len(inputs[groups]\
                                                                                      [batches][channels])][ubx]
                                                                            )
                                                                            i = inputs[groups][batches][channels]\
                                                                                      [(uby+uny)%len(inputs[groups]\
                                                                                      [batches][channels])][ubx]
                                                                            print("Ok try to place value " + str(i))
                                                                            print("group: " + str(groups) + " batches:" + str(batches) + " channels:" + str(channels) + " uby:" + str(uby) + " uny:"+str(uny) + " width:" + str(len(inputs[groups]\
                                                                                      [batches][channels])))
                                                                            
                                                                            ubo = ubob*outer_ubx*outer_uby + ubox*outer_uby + uboy
                                                                            ubi = ubib*inner_ubx*inner_uby + ubix*inner_uby + ubiy
                                                                            ubt = ubtb*temp_ubx*temp_uby + int(ubty/div_factor)*temp_ubx + ubtx
                                                                            urno = urnoc*outer_unx*outer_uny + urnoy*outer_unx 
                                                                            urni = urnic*inner_unx*inner_uny + urniy*inner_unx 
                                                                            urnt = urntc*temp_unx*temp_uny + urnty*temp_unx
                                                                        
                                                                            i_stream_idx = utils.get_overall_idx_new(proj_yaml["outer_projection"],
                                                                                                               {'URN': {'y':urnoy, 'chans':urnoc},
                                                                                                                'UB': {'y':uboy, 'batches':ubob},
                                                                                                                'UG': {'value': ugo}},
                                                                                 order=utils.input_order, default=['batches','chans'])
                                                                            i_value_idx = i_stream_idx*utils.get_proj_stream_count(proj_yaml["inner_projection"], 'I') + \
                                                                                          utils.get_overall_idx_new(proj_yaml["inner_projection"],
                                                                                                               {'URN': {'y':urniy, 'chans':urnic},
                                                                                                                'UB': {'y':ubiy, 'batches':ubib},
                                                                                                                'UG': {'value': ugi}},
                                                                                 order=utils.input_order, default=['batches','chans'])
                                                                            ibuf_idx = math.floor(i_value_idx / ivalues_per_buf)
                                                                            iv_idx = i_value_idx % ivalues_per_buf
                                                                            print("idx: " + str(i_value_idx))
                                                                            ibuf[ibuf_idx][(ugt*temp_ub*temp_un+ubt*temp_un + urnt)%ibuf_len][iv_idx] = i
    print(ibuf)
    return ibuf


def reorder_weight_array(weights, proj_yaml, wb_yaml):
    
    wvalues_per_buf, wbuf_len, wbuf_count = utils.get_iw_buffer_dimensions(
        wb_yaml, proj_yaml, 'W')
    inner_ug = proj_yaml["inner_projection"]["UG"]["value"]
    outer_ug = proj_yaml["outer_projection"]["UG"]["value"]
    temp_ug = proj_yaml.get("temporal_projection",{}).get("UG",{}).get("value", 1)
    inner_ue = proj_yaml["inner_projection"]["UE"]["value"]
    outer_ue = proj_yaml["outer_projection"]["UE"]["value"]
    temp_ue = proj_yaml.get("temporal_projection",{}).get("UE",{}).get("value", 1)
    inner_un = proj_yaml["inner_projection"]["URN"]["value"]
    inner_unc = proj_yaml["inner_projection"]["URN"]["chans"]
    inner_unx = proj_yaml["inner_projection"]["URN"]["x"]
    inner_uny = proj_yaml["inner_projection"]["URN"]["y"]
    outer_un = proj_yaml["outer_projection"]["URN"]["value"]
    outer_unc = proj_yaml["outer_projection"]["URN"]["chans"]
    outer_unx = proj_yaml["outer_projection"]["URN"]["x"]
    outer_uny = proj_yaml["outer_projection"]["URN"]["y"]
    temp_un = proj_yaml.get("temporal_projection",{}).get("URN",{}).get("value", 1)
    temp_unc = proj_yaml.get("temporal_projection",{}).get("URN",{}).get("chans", 1)
    temp_unx = proj_yaml.get("temporal_projection",{}).get("URN",{}).get("x", 1)
    temp_uny = proj_yaml.get("temporal_projection",{}).get("URN",{}).get("y", 1)
    inner_uw = proj_yaml["inner_projection"]["URW"]["value"]
    inner_uwx = proj_yaml["inner_projection"]["URW"]["x"]
    inner_uwy = proj_yaml["inner_projection"]["URW"]["y"]
    outer_uw = proj_yaml["outer_projection"]["URW"]["value"]
    outer_uwx = proj_yaml["outer_projection"]["URW"]["x"]
    outer_uwy = proj_yaml["outer_projection"]["URW"]["y"]  

    # Move the weights and inputs into the EMIF in the expected order
    wbuf = [[[0 for k in range(wvalues_per_buf)] 
            for i in range(wbuf_len)]              
            for j in range(wbuf_count)]
    print(weights)
    for ugt in range(temp_ug):
        for ugo in range(outer_ug): 
            for ugi in range(inner_ug):
                for ueo in range(outer_ue):
                    for uei in range(inner_ue):
                        for uet in range(temp_ue):
                            for urnoc in range(outer_unc):
                                for urnic in range(inner_unc):
                                    for urntc in range(temp_unc):
                                        for urnox in range(outer_unx):
                                            for urnix in range(inner_unx):
                                                for urntx in range(temp_unx):
                                                    for urnoy in range(outer_uny):
                                                        for urniy in range(inner_uny):
                                                            for urnty in range(temp_uny):
                                                                for urwox in range(outer_uwx):
                                                                    for urwix in range(inner_uwx):
                                                                        for urwoy in range(outer_uwy):
                                                                            for urwiy in range(inner_uwy):
                                                                                urno = urnox*outer_uny + urnoy + urnoc*outer_unx*outer_uny
                                                                                urni = urnix*inner_uny + urniy + urnic*inner_unx*inner_uny 
                                                                                urnt = urntx*temp_uny  + urnty + urntc*temp_unx*temp_uny
                                                                                ury = urwoy*inner_uwy+urwiy + urnoy*inner_uny+urniy
                                                                                w = weights[ugt*outer_ug*inner_ug+ugo*inner_ug+ugi]\
                                                                                           [uet*outer_ue*inner_ue+ueo*inner_ue+uei]\
                                                                                           [urntc*outer_unc*inner_unc+urnoc*inner_unc+urnic]\
                                                                                           [ury]\
                                                                                           [urwox*inner_uwx+urwix]
                                                                                print("Value: " + str(w))
                                                                                print("ug: " + str(ugt*outer_ug*inner_ug+ugo*inner_ug+ugi))
                                                                                print("ue: " + str(uet*outer_ue*inner_ue+ueo*inner_ue+uei))
                                                                                print("uc: " + str(urntc*outer_unc*inner_unc+urnoc*inner_unc+urnic))
                                                                                print("ury: " + str(urwoy*inner_uwy+urwiy))
                                                                                print("urx: " + str(urwox*inner_uwx+urwix))
                                                                                urwo = max(urwox,urwoy)
                                                                                urwi = max(urwix,urwiy)
                                                                                w_buf_inst_idx = 0
                                                                                buffer_cnt = 0
                                                                                stream_width = inner_ug*inner_ue*inner_un*inner_uw
                                                                                bus_idx=0
                                                                                mlb_chain_len=1
                                                                                outer_chain_len=1
                                                                                if ("PRELOAD" in proj_yaml["inner_projection"]):
                                                                                    mlb_chain_len=inner_ug*inner_ue*inner_un*inner_uw
                                                                                    w_buf_inst_idx = \
                                                                                        ugi*inner_ue*inner_un*inner_uw + \
                                                                                        uei*inner_un*inner_uw + \
                                                                                        urni*inner_uw + \
                                                                                        urwi
                                                                                    stream_width = 1
                                                                                else:
                                                                                    bus_idx = ugi*inner_ue*inner_un*inner_uw + \
                                                                                              uei*inner_un*inner_uw + \
                                                                                              urni*inner_uw + \
                                                                                              urwi
                                                                                    stream_width=inner_ug*inner_ue*inner_un*inner_uw
                                                                                if ("PRELOAD" in proj_yaml["outer_projection"]):
                                                                                    w_buf_inst_idx = \
                                                                                        (ugo*outer_ue*outer_un*outer_uw + \
                                                                                        ueo*outer_un*outer_uw + \
                                                                                        urno*outer_uw + \
                                                                                        urwo)*mlb_chain_len + \
                                                                                        w_buf_inst_idx
                                                                                    outer_chain_len = outer_ug*outer_ue*outer_uw*outer_un
                                                                                else:
                                                                                    stream_idx = ugo*outer_ue*outer_un*outer_uw + \
                                                                                        ueo*outer_un*outer_uw + \
                                                                                        urno*outer_uw + \
                                                                                        urwo
                                                                                    streams_per_buffer = math.floor(len(wbuf[0][0]) / stream_width)
                                                                                    buffer_cnt = math.floor(stream_idx / streams_per_buffer)
                                                                                    bus_idx = (stream_idx % streams_per_buffer)*stream_width + bus_idx
                                                                                buffer_idx = (outer_chain_len*mlb_chain_len - w_buf_inst_idx - 1)
                                                                                buffer_idx += ugt*temp_ue*temp_un + uet*temp_un
                                                                                wbuf[buffer_cnt][(buffer_idx + urnt) % wbuf_len][bus_idx] = w
    return wbuf



def reorder_output_array(outvals_yaml, proj_yaml, ab_yaml, outarray, ibuf_len):    
    ovalues_per_buf, obuf_len, obuf_count = utils.get_obuffer_dimensions(
        ab_yaml, proj_yaml)
    
    inner_uw = proj_yaml["inner_projection"]["URW"]["value"]
    inner_uwx = proj_yaml["inner_projection"]["URW"]["x"]
    inner_uwy = proj_yaml["inner_projection"]["URW"]["y"]
    outer_uw = proj_yaml["outer_projection"]["URW"]["value"]
    outer_uwx = proj_yaml["outer_projection"]["URW"]["x"]
    outer_uwy = proj_yaml["outer_projection"]["URW"]["y"]
    
    inner_un = proj_yaml["inner_projection"]["URN"]["value"]
    inner_unc = proj_yaml["inner_projection"]["URN"]["chans"]
    inner_unx = proj_yaml["inner_projection"]["URN"]["x"]
    inner_uny = proj_yaml["inner_projection"]["URN"]["y"]
    outer_un = proj_yaml["outer_projection"]["URN"]["value"]
    outer_unc = proj_yaml["outer_projection"]["URN"]["chans"]
    outer_unx = proj_yaml["outer_projection"]["URN"]["x"]
    outer_uny = proj_yaml["outer_projection"]["URN"]["y"]
    temp_un = proj_yaml.get("temporal_projection",{}).get("URN",{}).get("value", 1)
    temp_unc = proj_yaml.get("temporal_projection",{}).get("URN",{}).get("chans", 1)
    temp_unx = proj_yaml.get("temporal_projection",{}).get("URN",{}).get("x", 1)
    temp_uny = proj_yaml.get("temporal_projection",{}).get("URN",{}).get("y", 1)
    
    inner_ug = proj_yaml["inner_projection"]["UG"]["value"]
    outer_ug = proj_yaml["outer_projection"]["UG"]["value"]
    temp_ug = proj_yaml.get("temporal_projection",{}).get("UG",{}).get("value", 1)
    stridex = proj_yaml.get("stride",{}).get("x",1)
    stridey = proj_yaml.get("stride",{}).get("y",1)
    
    inner_ub = proj_yaml["inner_projection"]["UB"]["value"]
    inner_ubb = proj_yaml["inner_projection"]["UB"]["batches"]
    inner_ubx = proj_yaml["inner_projection"]["UB"]["x"]
    inner_uby = proj_yaml["inner_projection"]["UB"]["y"]
    outer_ub = proj_yaml["outer_projection"]["UB"]["value"]
    outer_ubb = proj_yaml["outer_projection"]["UB"]["batches"]
    outer_ubx = proj_yaml["outer_projection"]["UB"]["x"]
    outer_uby = proj_yaml["outer_projection"]["UB"]["y"]
    temp_ub = proj_yaml.get("temporal_projection",{}).get("UB",{}).get("value", obuf_len)
    temp_ubb = proj_yaml.get("temporal_projection",{}).get("UB",{}).get("batches", 1)
    temp_ubx = proj_yaml.get("temporal_projection",{}).get("UB",{}).get("x", obuf_len)
    temp_uby = proj_yaml.get("temporal_projection",{}).get("UB",{}).get("y", 1)
    
    inner_ue = proj_yaml["inner_projection"]["UE"]["value"]
    outer_ue = proj_yaml["outer_projection"]["UE"]["value"]
    temp_ue = proj_yaml.get("temporal_projection",{}).get("UE",{}).get("value", 1)
    
    for ugt in range(temp_ug):
        for ugo in range(outer_ug): 
            for ugi in range(inner_ug):
                for ubox in range(outer_ubx):
                    for ubix in range(inner_ubx):
                        for ubtx in range(int(temp_ubx/stridex)):
                            for uboy in range(outer_uby):
                                for ubiy in range(inner_uby):
                                    for ubty in range(int(temp_uby/stridey)):
                                        for ubob in range(outer_ubb):
                                            for ubib in range(inner_ubb):
                                                for ubtb in range(temp_ubb):
                                                    for ueo in range(outer_ue):
                                                        for uei in range(inner_ue):
                                                            for uet in range(temp_ue):
                                                                ubo = ubox*outer_uby + uboy + ubob*outer_ubx*outer_uby
                                                                ubi = ubix*inner_uby + ubiy + ubib*inner_ubx*inner_uby 
                                                                ubt = ubty*int(temp_ubx/stridex) + ubtx + ubtb*int(temp_ubx/stridex)*int(temp_uby/stridey)
                                                                
                                                                out_act_idx = ugo*outer_ub*outer_ue*inner_ug*inner_ub*inner_ue + \
                                                                              ubo*outer_ue*inner_ug*inner_ub*inner_ue + \
                                                                              ueo*inner_ug*inner_ub*inner_ue + \
                                                                              ugi*inner_ub*inner_ue + \
                                                                              ubi*inner_ue + uei
                                                                obuf_idx = math.floor(out_act_idx/ovalues_per_buf)
                                                                os_idx = out_act_idx % ovalues_per_buf
                                                                ubx = ubtx*outer_ubx*inner_ubx+ubox*inner_ubx+ubix
                                                                uby = ubty*outer_uby*inner_uby+uboy*inner_uby+ubiy
                                                                ubb = ubtb*outer_ubb*inner_ubb+ubob*inner_ubb+ubib
                                                                max_ubx = outer_ubx*inner_ubx*temp_ubx
                                                                max_uby = outer_uby*inner_uby*temp_uby
                                                                max_unx = inner_uwx*outer_uwx*outer_unx*temp_unx*inner_unx*outer_unx
                                                                max_uny = inner_uwy*outer_uwy*inner_uny*outer_uny*temp_uny
                                                                #print("Add?")
                                                                #print(uby)
                                                                #print(max_uby)
                                                                #print(max_uny)
                                                                #print((max_uby - max_uny))
                                                                if ((ubx <= (max_ubx - max_unx)/stridex) and ((uby <= (max_uby - max_uny)/stridey))):
                                                                    outarray[ugt*outer_ug*inner_ug+ugo*inner_ug+ugi]\
                                                                        [ubb]\
                                                                        [uet*outer_ue*inner_ue+ueo*inner_ue+uei]\
                                                                        [uby][ubx] = \
                                                                        outvals_yaml[obuf_idx*min(obuf_len,ibuf_len) + ugt*temp_ub*temp_ue+uet*temp_ub+ubt][os_idx]
    return outarray
