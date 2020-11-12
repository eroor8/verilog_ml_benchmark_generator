#!/usr/bin/env python

"""Tests for `verilog_ml_benchmark_generator` pyMTL Components."""
import numpy
import pytest
import random
import math
import os
import sys
from pymtl3 import *

from click.testing import CliRunner

from verilog_ml_benchmark_generator import utils
from verilog_ml_benchmark_generator import module_classes
from verilog_ml_benchmark_generator import state_machine_classes
from verilog_ml_benchmark_generator import cli

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_helpers import *

def test_StateMachineSim():
    """Test Component class Statemachine"""
    projection = {"name": "test",
                  "activation_function": "RELU",
                  "stream_info": {"W": 4,
                                  "I": 4,
                                  "O": 16},
                  "inner_projection": {'URN':{'value':2},'URW':{'value':3},
                                       'UB':{'value':2},'UE':{'value':1},
                                       'UG':{'value':2},
                                       'PRELOAD':[{'dtype':'W','bus_count':1}]},
                  "outer_projection": {'URN':{'value':2},'URW':{'value':1},
                                       'UB':{'value':2},'UE':{'value':1},
                                       'UG':{'value':2},
                                       'PRELOAD':[{'dtype':'W','bus_count':1}]}
                  }
    wb_spec = {
        "block_name": "ml_block_weights",
        "simulation_model": "Buffer",
        "ports": [
            {"name":"portaaddr", "width":4, "direction": "in", "type":"ADDRESS"},
            {"name":"portadatain", "width":16, "direction": "in", "type":"DATA"},
            {"name":"portadataout", "width":16, "direction": "out", "type":"DATA"},
            {"name":"portawe", "width":1, "direction": "in", "type":"WEN"},
         ]
    }
    ib_spec = {
        "block_name": "ml_block_inputs",
        "simulation_model": "Buffer",
        "ports": [
            {"name":"portaaddr", "width":3, "direction": "in", "type":"ADDRESS"},
            {"name":"portadatain", "width":32, "direction": "in", "type":"DATA"},
            {"name":"portadataout", "width":32, "direction": "out", "type":"DATA"},
            {"name":"portawe", "width":1, "direction": "in", "type":"WEN"},
        ]
    }
    ob_spec = {
        "block_name": "mlb_outs",
        "simulation_model": "Buffer",
        "ports": [
            {"name":"portaaddr", "width":3, "direction": "in", "type":"ADDRESS"},
            {"name":"portadatain", "width":16, "direction": "in", "type":"DATA"},
            {"name":"portadataout", "width":16, "direction": "out", "type":"DATA"},
            {"name":"portawe", "width":1, "direction": "in", "type":"WEN"},
        ]
    }
    mlb_spec = {
        "block_name": "ml_block",
        "simulation_model": "MLB",
        "MAC_info": { "num_units": 128, "data_widths": {"W":8, "I":8, "O": 32} },
        "ports": [
            {"name":"a_in", "width":32, "direction": "in", "type":"W"},
            {"name":"a_out", "width":32, "direction": "out", "type":"W"},
            {"name":"b_in", "width":64, "direction": "in", "type":"I"},
            {"name":"b_out", "width":64, "direction": "out", "type":"I"},
            {"name":"res_in", "width":128, "direction": "in", "type":"O"},
            {"name":"res_out", "width":128, "direction": "out", "type":"O"},
            {"name":"a_en", "width":1, "direction": "in", "type":"W_EN"},
            {"name":"b_en", "width":1, "direction": "in", "type":"I_EN"},
            {"name":"acc_en", "width":1, "direction": "in", "type":"ACC_EN"},
        ]
    }
    
    sm_testinst = state_machine_classes.StateMachine(
        mlb_spec=mlb_spec,
        wb_spec=wb_spec,
        ib_spec=ib_spec,
        ob_spec=ob_spec,
        proj_spec=projection)

    sm_testinst.elaborate()
    sm_testinst.apply(DefaultPassGroup())
    sm_testinst.sim_reset()
    testinst = sm_testinst.datapath
    
    # Calculate required buffers etc.
    mlb_count = utils.get_mlb_count(projection["outer_projection"])
    mac_count = utils.get_mlb_count(projection["inner_projection"])
    ibuf_len = 2**ib_spec["ports"][0]["width"]
    obuf_len = 2**ob_spec["ports"][0]["width"]
    wbuf_len = 2**wb_spec["ports"][0]["width"]
    
    # Load the weight buffer
    wbuf_count = 1
    weight_stream_count = utils.get_proj_stream_count(projection["outer_projection"], 'W')
    wvalues_per_stream = utils.get_proj_stream_count(projection["inner_projection"], 'W')
    wstream_bitwidth = wvalues_per_stream*projection["stream_info"]["W"]
    wstreams_per_buf = math.floor(wb_spec["ports"][1]["width"]/wstream_bitwidth)
    wbuf_count = math.ceil(weight_stream_count / wstreams_per_buf)
    wvalues_per_buf = min(wstreams_per_buf*wvalues_per_stream, wvalues_per_stream*weight_stream_count)
    
    wbuf = [[[random.randint(0,(2**projection["stream_info"]["W"])-1)
            for k in range(wvalues_per_buf)]    # values per word
            for i in range(wbuf_len)]           # words per buffer
            for j in range(wbuf_count)]         # buffer count
    sm_testinst.sm_start @= 1
    load_buffers_sm(sm_testinst, "datapath_weight_datain_sm",
                 wbuf, projection["stream_info"]["W"])
    sm_testinst.sm_start @= 0
    
    check_buffers(testinst, testinst.weight_modules,
                  "ml_block_weights_inst_{}",
                  wbuf, projection["stream_info"]["W"], sm_testinst)
  
    # Calculate required buffers etc.
    iouter_stream_count = utils.get_proj_stream_count(projection["outer_projection"], 'I')
    iouter_stream_width = utils.get_proj_stream_count(projection["inner_projection"], 'I') * \
                         projection["stream_info"]["I"]
    ototal_stream_count = utils.get_proj_stream_count(projection["outer_projection"], 'O') * \
                          utils.get_proj_stream_count(projection["inner_projection"], 'O') 
    activation_width = projection["stream_info"]["I"]
    istreams_per_buf = math.floor(ib_spec["ports"][1]["width"]/iouter_stream_width)
    ivalues_per_buf = istreams_per_buf*utils.get_proj_stream_count(projection["inner_projection"], 'I')
    ostreams_per_buf = math.floor(ob_spec["ports"][1]["width"]/activation_width)
    ibuf_count = math.ceil(iouter_stream_count/istreams_per_buf)
    obuf_count = math.ceil(ototal_stream_count/ostreams_per_buf)
    
    # Load the input buffer
    # Several values per word, words per buffer, buffers...
    ibuf = [[[random.randint(0,(2**projection["stream_info"]["I"])-1)
             for k in range(ivalues_per_buf)]            # values per word
             for i in range(ibuf_len)]                   # words per buffer
             for j in range (ibuf_count)]                # buffers
    print(ibuf_len)
    load_buffers_sm(sm_testinst, "datapath_input_datain_sm",
                 ibuf, projection["stream_info"]["I"])
    check_buffers(testinst, testinst.input_act_modules,
                  "ml_block_inputs_inst_{}",
                  ibuf, projection["stream_info"]["I"], sm_testinst)
    # Now load the weights into the MLBs
    inner_ub = projection["inner_projection"]["UB"]["value"]
    outer_ub = projection["outer_projection"]["UB"]["value"]
    wbi_section_length = projection["inner_projection"]["UE"]["value"] * \
                        projection["inner_projection"]["URN"]["value"] * \
                        projection["inner_projection"]["URW"]["value"]
    wbo_section_length = projection["outer_projection"]["UE"]["value"] * \
                        projection["outer_projection"]["URN"]["value"] * \
                        projection["outer_projection"]["URW"]["value"] *\
                        projection["inner_projection"]["UG"]["value"] *  wbi_section_length
    sm_testinst.sim_tick()
    starti = 0

    for jj in range(mlb_count*mac_count):
        sm_testinst.sim_tick()
                 
    # Check they are right
    assert(check_mlb_chains_values(testinst, mlb_count, mac_count, 1, 1,
                            "ml_block_inst_{}", "weight_out_{}", wbuf,
                            projection["stream_info"]["W"],
                            wbo_section_length, outer_ub,
                            wbi_section_length, inner_ub))

    # Now stream the inputs, and check the outputs!
    sm_testinst.sim_tick()
    for j in range(obuf_len):
        sm_testinst.sim_tick()
    
    obuf = [[[0
             for i in range(ostreams_per_buf)]
             for i in range(obuf_len)]
             for j in range (obuf_count)]
    obuf = get_expected_outputs(obuf, ostreams_per_buf,
                                wbuf,
                                ibuf, ivalues_per_buf,
                                projection)
    
    obuf_results = [[[[0]
             for i in range(ostreams_per_buf)]
             for i in range(obuf_len)]
             for j in range (obuf_count)]
    
    print("Checking...: " )
    obuf_results = read_out_stored_values_from_sm("external_out",
                                        obuf_results, projection["stream_info"]["I"],
                                        sm_testinst)
    print("DONE!")
    sm_testinst.sim_tick()
    sm_testinst.sim_tick()
    assert(sm_testinst.done)
    
    print("EXPECTED: " + str(obuf))
    print("ACTUAL: " + str(obuf_results))
    print("W: " + str(wbuf))
    print("I: " + str(ibuf))
    for bufi in range(obuf_count):
        for olen in range(min(obuf_len,ibuf_len)-1): 
            assert obuf[bufi][olen] == obuf_results[bufi][olen]
