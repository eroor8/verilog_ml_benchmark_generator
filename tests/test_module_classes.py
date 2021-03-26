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
from verilog_ml_benchmark_generator import cli

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_helpers import *
    
def test_ActivationWrapper():
    """Test Component class RELU"""
    test_vecs = [
        {"ins":[4,3, False], "outs":[[7,7],[8,0]]},
        {"ins":[4,2, False], "outs":[[3,3],[5,0],[9,0]]},
        {"ins":[3,5, False], "outs":[[3,3],[4,0]]}, # 100 -> 0
        {"ins":[4,4, True], "outs":[[7,7],[8,0]]},
        {"ins":[4,2, True], "outs":[[3,3],[5,0],[9,0]]}, # 0101 
        {"ins":[3,5, True], "outs":[[3,Bits5(3)],[4,Bits5(0)]]},
    ]
    i = 0
    for testvec in test_vecs:
        print("VEC" + str(testvec))
        i = i + 1
        testinst = module_classes.ActivationWrapper(len(testvec["outs"]),
                                                    {'type': "RELU"},
                                       testvec["ins"][0],
                                       testvec["ins"][1],
                                       testvec["ins"][2])
        testinst.elaborate()
        testinst.apply(DefaultPassGroup())
        for pairidx in range(len(testvec["outs"])):
            input_bus = getattr(testinst, "activation_function_in_"+ \
                                str(pairidx))
            input_bus @= testvec["outs"][pairidx][0]
        testinst.sim_tick()
        for pairidx in range(len(testvec["outs"])):
            output_bus = getattr(testinst, "activation_function_out_"+ \
                                 str(pairidx))
            assert output_bus == testvec["outs"][pairidx][1]

    testinst = module_classes.ActivationWrapper(2, {'type': "notRELU"},
                                                test_vecs[0]["ins"][0],
                                                test_vecs[0]["ins"][1],
                                                test_vecs[0]["ins"][2])
                                                
    with pytest.raises(AssertionError):
        testinst.elaborate()
    

def test_HWB():
    """Test Component class HWB"""
    spec = {
        "block_name": "emif_block",
        "simulation_model": "EMIF",
        "ports": [
            {"name":"address", "width": 12, "direction": "in", "type": "AVALON_ADDRESS"},
            {"name":"read", "width": 1,"direction": "in","type": "AVALON_READ"},
            {"name":"readdata","width": 32,"direction": "out","type": "AVALON_READDATA"},
            {"name":"writedata","width": 32,"direction": "in","type": "AVALON_WRITEDATA"},
            {"name":"write","width": 1,"direction": "in","type": "AVALON_WRITE"},
            {"name":"waitrequest", "width": 1, "direction": "out", "type": "AVALON_WAITREQUEST"},
            {"name":"readdatavalid", "width": 1, "direction": "out", "type": "AVALON_READDATAVALID"}
        ],
        "pipelined":"True",
        "fill": []
    }
    testinst = module_classes.HWB_Sim(spec)
    testinst.elaborate()
    
    spec = {
        "block_name": "test_block",
        "ports": [{"name":"A", "width":5, "direction": "in", "type":"C"},
                  {"name":"B", "width":5, "direction": "out", "type":"C"},
                  {"name":"C", "width":5, "direction": "in", "type":"W"},
                  {"name":"D", "width":5, "direction": "out", "type":"W"},
                  {"name":"E", "width":5, "direction": "in", "type":"I"},
                  {"name":"F", "width":5, "direction": "out", "type":"I"},
                  {"name":"G", "width":5, "direction": "out", "type":"I"},
                  ],
        "MAC_info": {"num_units":4}
    }
    # Basic case
    proj_spec = {'C':1,'RX':1,
                  'B':1,'PX':1,'PY':1,'E':1,
                  'G':1}
    testinst = module_classes.HWB_Sim(spec, proj_spec)
    testinst.elaborate()
    
def test_MLB_Wrapper():
    """Test Component class HWB Wrapper"""
    spec = {
        "block_name": "test_block",
        "ports": [{"name":"acc_en", "width":1, "direction": "in", "type":"ACC_EN"},
                  {"name":"a", "width":64, "direction": "in", "type":"W"},
                  {"name":"test", "width":1, "direction": "in", "type":"C"},
                  {"name":"testo", "width":1, "direction": "out", "type":"C"},
                  {"name":"a_out", "width":64, "direction": "out", "type":"W"},
                  {"name":"a_en", "width":1, "direction": "in", "type":"W_EN"},
                  {"name":"b", "width":64, "direction": "in", "type":"I"},
                  {"name":"b_out", "width":64, "direction": "out", "type":"I"},
                  {"name":"b_en", "width":1, "direction": "in", "type":"I_EN"},
                  {"name":"res", "width":128, "direction": "in", "type":"O"},
                  {"name":"res_out", "width":128, "direction": "out", "type":"O"}
                  ],
        "simulation_model": "MLB",
        "access_patterns": {"AP1":1, "AP2":8, "AP3":1, "AP4":1, "AP5":1}
    }
    projection = {"name": "test",
                  "activation_function": "RELU",
                  "data_widths": {"W": 4,
                                  "I": 4,
                                  "O": 16},
                  "inner_projection": {'RY':2, 'C':1,'RX':2,
                                       'B':1,'E':1,
                                       'G':1, 'PX':1, 'PY':1,
                                       'PRELOAD':[{'dtype':'W','bus_count':1}]}
                  }
    testinst = module_classes.MLB_Wrapper(spec,[projection])
    testinst.elaborate()
    testinst.apply(DefaultPassGroup()) 
    testinst.a @= 0
    testinst.a_en @= 0
    testinst.b @= 0
    testinst.b_en @= 0
    testinst.res @= 0
    testinst.acc_en @= 0
    testinst.sim_tick()
    
    # Load weights
    testinst.a_en @= 1
    testinst.a @= 1
    for cycle in range(4):
        assert testinst.a_out == 0
        testinst.sim_tick()
    assert testinst.a_out == 1
    assert testinst.sim_model.mac_modules.MAC_inst_0.weight_out == 1
    assert testinst.sim_model.mac_modules.MAC_inst_1.weight_out == 1
    assert testinst.sim_model.mac_modules.MAC_inst_2.weight_out == 1
    assert testinst.sim_model.mac_modules.MAC_inst_3.weight_out == 1
    
    testinst.a_en @= 0
    testinst.b_en @= 1
    testinst.b @= 35 #23
    testinst.sim_tick()

    assert(testinst.b_out[0:4] == 0)
    assert(testinst.b_out[4:8] == 0)
    assert(testinst.res_out == 5)
    testinst.b @= 20 #14
    testinst.sim_tick()
    assert(testinst.b_out[0:4] == 3)
    assert(testinst.b_out[4:8] == 2)
    testinst.sim_tick()
    
def test_HWB_Wrapper():
    """Test Component class HWB Wrapper"""
    spec = {
        "block_name": "test_block",
        "ports": [{"name":"A", "width":5, "direction": "in", "type":"C"},
                  {"name":"B", "width":5, "direction": "out", "type":"C"},
                  {"name":"C", "width":5, "direction": "in", "type":"ADDRESS"},
                  {"name":"D", "width":5, "direction": "out", "type":"ADDRESS"},
                  {"name":"E", "width":5, "direction": "in", "type":"OTHER"},
                  {"name":"F", "width":5, "direction": "out", "type":"OTHER"},
                  ]
    }
    testinst = module_classes.HWB_Wrapper(spec,2)
    testinst.elaborate()
    testinst.apply(DefaultPassGroup()) 
    testinst.A @= 8
    testinst.C @= 9
    testinst.E_0 @= 10
    testinst.E_1 @= 11
    testinst.sim_tick()
    assert testinst.test_block_inst_0.A == 8
    assert testinst.test_block_inst_1.A == 8
    assert testinst.test_block_inst_0.C == 9
    assert testinst.test_block_inst_1.C == 9
    assert testinst.test_block_inst_0.E == 10
    assert testinst.test_block_inst_1.E == 11
    assert testinst.B_0 == 0
    assert testinst.B_1 == 0
    assert testinst.D_0 == 0
    assert testinst.D_1 == 0
    assert testinst.F_0 == 0
    assert testinst.F_1 == 0
    
def test_MergeBusses():
    """Test Component class MergeBusses"""
    test_vecs = [
        {"ins":[2,4,4,4,1], "outs":[[0,1,2,3],[0,1,2,3]]},
        {"ins":[2,4,5,2,2], "outs":[[0,1,2,3],[4,14]]},
        {"ins":[3,8,23,4,6], "outs":[[0,1,2,3,4,5,6,7],[181896,62,0,0]]},
        {"ins":[2,4,4,4], "outs":[[0,1,2,3],[4,14,0,0]]},
        {"ins":[2,4,5,2], "outs":[[0,1,2,3],[4,14]]},
        {"ins":[3,8,23,4], "outs":[[0,1,2,3,4,5,6,7],[1754760,7,0,0]]},
    ]
    for testvec in test_vecs:
        if len(testvec["ins"]) == 5:
            testinst = module_classes.MergeBusses(testvec["ins"][0],
                        testvec["ins"][1], testvec["ins"][2], testvec["ins"][3],
                        ins_per_out=testvec["ins"][4])
        else:
            testinst = module_classes.MergeBusses(testvec["ins"][0],
                        testvec["ins"][1], testvec["ins"][2], testvec["ins"][3])
        testinst.elaborate()
        testinst.apply(DefaultPassGroup())
        for i in range(len(testvec["outs"][0])):
            in_bus = getattr(testinst, "input_"+str(i))
            in_bus @= testvec["outs"][0][i]
        testinst.sim_tick()
        for i in range(len(testvec["outs"][1])):
            out_bus = getattr(testinst, "output_"+str(i))
            assert out_bus == testvec["outs"][1][i]
      
    illegal_test_vecs = [
        #{"ins":[2,4,5,8,3]}, # ins per out not possible 
        #{"ins":[2,8,16,3,2]}, # not enough outputs 
        #{"ins":[2,8,6,2]} # no possible solution 
    ]  
    for testvec in illegal_test_vecs:
        if len(testvec["ins"]) == 5:
            testinst = module_classes.MergeBusses(testvec["ins"][0],
                        testvec["ins"][1], testvec["ins"][2], testvec["ins"][3],
                        ins_per_out=testvec["ins"][4])
        else:
            testinst = module_classes.MergeBusses(testvec["ins"][0],
                        testvec["ins"][1], testvec["ins"][2], testvec["ins"][3])
        with pytest.raises(AssertionError):
            testinst.elaborate()
    
def test_WeightInterconnect():
    """Test Component class WeightInterconnect"""
    test_vecs = [ # bufferwidth, mlbwidth, mlbwidthused, num_buffers, num_mlbs, proj.
        {"ins":[8,8,3,4,10,{'C':1,'RX':1, 'RY':2,
                          'B':1,'PX':2,'PY':1,'E':2,
                          'G':1}],  
         "outs":[[8,26,44,62],[0,1,2,3,0,1,2,3,0,0]]},  #0000 1000 / 0001 1010 / 0010 1100 
        {"ins":[8,8,3,4,15,{'C':1,'RX':2, 'RY':1,
                          'B':3,'PX':1,'PY':1,'E':1,
                          'G':2}],  
         "outs":[[8,26,44,62],[0,1,0,1,0,1,2,3,2,3,2,3,0,0,0]]},
    ]
    for testvec in test_vecs:
        testinst = module_classes.WeightInterconnect(testvec["ins"][0],
                    testvec["ins"][1], testvec["ins"][2], testvec["ins"][3],
                    testvec["ins"][4], testvec["ins"][5])
        testinst.elaborate()
        testinst.apply(DefaultPassGroup())
        for i in range(len(testvec["outs"][0])):
            in_bus = getattr(testinst, "inputs_from_buffer_"+str(i))
            in_bus @= testvec["outs"][0][i]
        testinst.sim_tick()
        for i in range(len(testvec["outs"][1])):
            out_bus = getattr(testinst, "outputs_to_mlb_"+str(i))
            assert out_bus == testvec["outs"][1][i]
    illegal_test_vecs = [
        {"ins":[8,2,3,4,10,{'C':2,'RX':1,
                          'B':2,'PX':1,'PY':1,'E':2,
                          'G':1}]}, # mlb bitwidths dont make sense
        {"ins":[8,8,3,400,15,{'C':1,'RX':2,
                          'B':3,'PX':1,'PY':1,'E':10,
                          'G':2}]}, # not enough mlbs
        {"ins":[8,8,3,4,1000,{'C':1,'RX':2,
                          'B':3,'PX':1,'PY':1,'E':10,
                          'G':2}]} # not enough buffers
    ] 
    for testvec in illegal_test_vecs:
        with pytest.raises(AssertionError):
            testinst = module_classes.WeightInterconnect(testvec["ins"][0],
                        testvec["ins"][1], testvec["ins"][2], testvec["ins"][3],
                        testvec["ins"][4], testvec["ins"][5])
            testinst.elaborate()
    
def test_InputInterconnect():
    """Test Component class InputInterconnect"""
    test_vecs = [ # bufferwidth, mlbwidth, mlbwidthused, num_buffers, num_mlbs, proj.
        {"ins":[8,8,3,2,10,{'C':2, 'RX':2, 'RY':1,
                          'B':1,'PX':2,'PY':1,'E':1,
                          'G':1}],  
         "outs":[[62,26],[1,2,3,4,5,6,7,8,9,10],[6,1,7,3,2,5,3,7,0,0]]},
        {"ins":[8,8,3,2,10,{'C':1, 'RX':2, 'RY':2,
                          'B':2,'PX':1,'PY':1,'E':1,
                          'G':1}],  
         "outs":[[62,26],[1,2,3,4,5,6,7,8,9,10],[6,1,6,3,2,5,2,7,0,0]]},
         # 111 110    /   011 010
        {"ins":[8,8,3,2,10,{'C':1,'RX':2, 'RY':1,
                          'B':2,'PX':1,'PY':1,'E':2,
                          'G':1}],  
         "outs":[[62,26],[1,2,3,4,5,6,7,8,9,10],[6,1,6,3,7,5,7,7,0,0]]}
    ]
    for testvec in test_vecs:
        testinst = module_classes.InputInterconnect(testvec["ins"][0],
                    testvec["ins"][1], testvec["ins"][2], testvec["ins"][3],
                    testvec["ins"][4], testvec["ins"][5], inner_width=testvec["ins"][2])
        testinst.elaborate()
        testinst.apply(DefaultPassGroup())
        for i in range(len(testvec["outs"][0])):
            in_bus = getattr(testinst, "inputs_from_buffer_"+str(i))
            in_bus @= testvec["outs"][0][i]
        for i in range(len(testvec["outs"][1])):
            in_bus = getattr(testinst, "inputs_from_mlb_"+str(i))
            in_bus @= testvec["outs"][1][i]
        testinst.sim_tick()
        for i in range(len(testvec["outs"][2])):
            out_bus = getattr(testinst, "outputs_to_mlb_"+str(i))
            print("--", testvec, " - ", i)
            print(out_bus)
            print(testvec["outs"][2][i])
            assert out_bus == testvec["outs"][2][i]
        #assert(0) 0 111 1 110     and 11 010  
        
    
def test_OutputPSInterconnect():
    """Test Component class InputInterconnect"""
    test_vecs = [ # afwidth, mlbwidth, mlbwidthused, num_afs, num_mlbs, proj.
        {"ins":[3,10,6,6,10,{'C':2,'RX':2, 'RY':1,
                          'B':2,'PX':1,'PY':1,'E':1,
                          'G':1}],  
         "outs":[[62,0,0,62,26,0,0,26,0,0],[6,7,2,3,0,0],[0,62,0,0,0,26,0,0,0,0]]}
    ]
    for testvec in test_vecs:
        testinst = module_classes.OutputPSInterconnect(testvec["ins"][0],
                    testvec["ins"][1], testvec["ins"][2], testvec["ins"][3],
                    testvec["ins"][4], testvec["ins"][5])
        testinst.elaborate()
        testinst.apply(DefaultPassGroup())
        for i in range(len(testvec["outs"][0])):
            in_bus = getattr(testinst, "inputs_from_mlb_"+str(i))
            in_bus @= testvec["outs"][0][i]
        testinst.sim_tick()
        for i in range(len(testvec["outs"][1])):
            out_bus = getattr(testinst, "outputs_to_afs_"+str(i))
            assert out_bus == Bits3(testvec["outs"][1][i])
        for i in range(len(testvec["outs"][2])):
            out_bus = getattr(testinst, "outputs_to_mlb_"+str(i))
            assert out_bus == testvec["outs"][2][i]

def test_Datapath():
    """Test Component class Datapath"""
    projection = {"name": "test",
                  "activation_function": "NONE",
                  "data_widths": {"W": 4,
                                  "I": 4,
                                  "O": 16},
                  "inner_projection": {'C':2,'RX':3, 'RY':1,
                                       'B':1,'PX':2,'PY':1,'E':1,
                                       'G':2,
                                       'PRELOAD':[{'dtype':'W','bus_count':1}]},
                 # "inner_projection": {'C':2,'RX':3, 'RY':1,
                  "outer_projection": {'C':2,'RX':1, 'RY':1,
                                       'B':2,'PX':1,'PY':1,'E':1,
                                       'G':2,
                                       'PRELOAD':[{'dtype':'W','bus_count':1}]}
                  #"outer_projection": {'C':2,'RX':1, 'RY':1,
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
    
    testinst = module_classes.Datapath(
        mlb_spec=mlb_spec,
        wb_spec=wb_spec,
        ib_spec=ib_spec,
        ob_spec=ib_spec,
        proj_specs=[projection])

    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    testinst.sim_reset()
    
    # Calculate required buffers etc.
    mlb_count = utils.get_mlb_count(projection["outer_projection"])
    mac_count = utils.get_mlb_count(projection["inner_projection"])
    ibuf_len = 2**ib_spec["ports"][0]["width"]
    obuf_len = 2**ib_spec["ports"][0]["width"]
    wbuf_len = 2**wb_spec["ports"][0]["width"]
    print(obuf_len)
    # Load the weight buffer
    wbuf_count = 1
    weight_stream_count = utils.get_proj_stream_count(projection["outer_projection"], 'W')
    wvalues_per_stream = utils.get_proj_stream_count(projection["inner_projection"], 'W')
    wstream_bitwidth = wvalues_per_stream*projection["data_widths"]["W"]
    wstreams_per_buf = math.floor(wb_spec["ports"][1]["width"]/wstream_bitwidth)
    wbuf_count = math.ceil(weight_stream_count / wstreams_per_buf)
    wvalues_per_buf = min(wstreams_per_buf*wvalues_per_stream, wvalues_per_stream*weight_stream_count)
    
    wbuf = [[[random.randint(0,3)
            for k in range(wvalues_per_buf)]    # values per word
            for i in range(wbuf_len)]           # words per buffer
            for j in range(wbuf_count)]         # buffer count
    load_buffers(testinst, "weight_modules_portawe_{}_top",
                "weight_modules_portaaddr_top", "weight_datain",
                wbuf, projection["data_widths"]["W"])
    check_buffers(testinst, testinst.weight_modules,
                  "ml_block_weights_inst_{}",
                wbuf, projection["data_widths"]["W"])
    
    # Calculate required buffers etc.
    iouter_stream_count = utils.get_proj_stream_count(projection["outer_projection"], 'I')
    iouter_stream_width = utils.get_proj_stream_count(projection["inner_projection"], 'I') * \
                         projection["data_widths"]["I"]
    ototal_stream_count = utils.get_proj_stream_count(projection["outer_projection"], 'O') * \
                          utils.get_proj_stream_count(projection["inner_projection"], 'O') 
    activation_width = projection["data_widths"]["I"]
    istreams_per_buf = math.floor(ib_spec["ports"][1]["width"]/iouter_stream_width)
    ivalues_per_buf = istreams_per_buf*utils.get_proj_stream_count(projection["inner_projection"], 'I')
    ostreams_per_buf = math.floor(ib_spec["ports"][1]["width"]/activation_width)
    ibuf_count = math.ceil(iouter_stream_count/istreams_per_buf)
    obuf_count = math.ceil(ototal_stream_count/ostreams_per_buf)
    
    # Load the input buffer
    # Several values per word, words per buffer, buffers...
    ibuf = [[[random.randint(0,4)
             for k in range(ivalues_per_buf)]            # values per word
             for i in range(ibuf_len)]                   # words per buffer
             for j in range (ibuf_count)]                # buffers
    load_buffers(testinst, "input_act_modules_portawe_{}_top",
                "input_act_modules_portaaddr_top", "input_datain",
                ibuf, projection["data_widths"]["I"])
    check_buffers(testinst, testinst.input_act_modules,
                  "ml_block_inputs_inst_{}",
                ibuf, projection["data_widths"]["I"])
    
    # Now load the weights into the MLBs
    inner_ub = projection["inner_projection"]["B"] * projection["inner_projection"]["PX"] * projection["inner_projection"]["PY"]
    outer_ub = projection["outer_projection"]["B"] * projection["outer_projection"]["PY"] * projection["outer_projection"]["PX"]
    wbi_section_length = projection["inner_projection"]["E"] * \
                        projection["inner_projection"]["C"] * \
                        projection["inner_projection"]["RY"] * \
                        projection["inner_projection"]["RX"]
    wbo_section_length = projection["outer_projection"]["E"] * \
                        projection["outer_projection"]["C"] * \
                        projection["outer_projection"]["RY"] * \
                        projection["outer_projection"]["RX"] *\
                        projection["inner_projection"]["G"] *  wbi_section_length
    starti = 0
    for ugo in range(projection["outer_projection"]["G"]):
        for ubo in range(outer_ub):
            for ugi in range(projection["inner_projection"]["G"] *
                             projection["outer_projection"]["RX"] *
                             projection["outer_projection"]["RY"] *
                             projection["outer_projection"]["C"] *
                             projection["outer_projection"]["E"]):
                for ubi in range(inner_ub):
                    wbi_section_start = wbo_section_length*ugo + wbi_section_length*ugi
                    print(wbi_section_start)
                    starti = stream_mlb_values(testinst, wbi_section_length,
                              ["weight_modules_portaaddr_top"],
                              [0],
                              [wbuf_len],
                              ["mlb_modules_a_en_top"], starti=wbi_section_start)
      
    # Check they are right
    assert(check_mlb_chains_values(testinst, mlb_count, mac_count, 1, 1,
                            "ml_block_inst_{}", "weight_out_{}", wbuf,
                            projection["data_widths"]["W"],
                            wbo_section_length, outer_ub,
                                   wbi_section_length, inner_ub))
 
    
    # Now stream the inputs, and check the outputs!
    stream_mlb_values(testinst, obuf_len,
                      ["input_act_modules_portaaddr_top", "input_act_modules_portaaddr_o_top"],
                      [0, -1],
                      [ibuf_len, obuf_len],
                      ["mlb_modules_b_en_top"] +
                      ["input_act_modules_portawe_{}_out_top".format(obi+len(ibuf))
                       for obi in range(obuf_count)])
      

    #print(testinst.output_act_modules.mlb_outs_inst_0.sim_model_inst0.data)
    
    obuf = [[[0
             for i in range(ostreams_per_buf)]
             for i in range(obuf_len)]
             for j in range (obuf_count)]
    obuf = get_expected_outputs_old(obuf, ostreams_per_buf,
                                wbuf,
                                ibuf, ivalues_per_buf,
                                projection)
    
    obuf_results = [[[[0]
             for i in range(ostreams_per_buf)]
             for i in range(obuf_len)]
             for j in range (obuf_count)]

    obuf_results = read_out_stored_values(testinst, "input_act_modules_portaaddr_o_top", "dataout",
                                          obuf_results, projection["data_widths"]["I"], start_buffer=len(ibuf))
    
    
    print("EXPECTED: " + str(obuf))
    print("ACTUAL: " + str(obuf_results))

    print("W: " + str(wbuf))
    print("I: " + str(ibuf))
    for bufi in range(obuf_count):
        for olen in range(min(obuf_len,ibuf_len)-1): #(obuf_len-1): 
            assert obuf[bufi][olen] == obuf_results[bufi][olen]
            
def test_multiple_Datapaths():
    """Test Component class Datapath with > 1 projections"""
    projections = [{"name": "",
                  "activation_function": "NONE",
                  "data_widths": {"W": 4,
                                  "I": 4,
                                  "O": 16},
                  "inner_projection": {'C':2,'RY':1,'RX':1,
                                       'B':1,'PX':1,'PY':1,'E':2,
                                       'G':2,
                                       'PRELOAD':[{'dtype':'W','bus_count':1}]},
                  "outer_projection": {'C':1,'RY':1,'RX':1,
                                       'B':1,'PX':1,'PY':1,'E':1,
                                       'G':1,
                                       'PRELOAD':[{'dtype':'W','bus_count':1}]
                  }
                  },
                  {"name": "test2",
                  "activation_function": "NONE",
                  "data_widths": {"W": 4,
                                  "I": 4,
                                  "O": 16},
                  "inner_projection": {'C':2,'RY':1,'RX':1,
                                       'B':1,'PX':1,'PY':1,'E':1,
                                       'G':1,
                                       'PRELOAD':[{'dtype':'W','bus_count':1}]},
                  "outer_projection": {'C':1,'RY':1,'RX':1,
                                       'B':1,'PX':1,'PY':1,'E':1,
                                       'G':1,
                                       'PRELOAD':[{'dtype':'W','bus_count':1}]}
                  }]
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
    mlb_spec = {
        "block_name": "ml_block",
        "simulation_model": "MLB",
        "MAC_info": { "num_units": 128, "data_widths": {"W":8, "I":8, "O": 32} },
        "ports": [
            {"name":"config_in", "width":1, "direction": "in", "type":"MODE"},
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
    
    testinst = module_classes.Datapath(
        mlb_spec=mlb_spec,
        wb_spec=wb_spec,
        ib_spec=ib_spec,
        ob_spec=ib_spec,
        proj_specs=projections)
    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    prev_ibuf_len=0
    testinst.sim_reset()
    
    projection = projections[0]
    testinst.sel @= 0
    iouter_stream_count = utils.get_proj_stream_count(projection["outer_projection"], 'I')
    iouter_stream_width = utils.get_proj_stream_count(projection["inner_projection"], 'I') * \
                             projection["data_widths"]["I"]
    ototal_stream_count = utils.get_proj_stream_count(projection["outer_projection"], 'O') * \
                              utils.get_proj_stream_count(projection["inner_projection"], 'O') 
    activation_width = projection["data_widths"]["I"]
    istreams_per_buf = math.floor(ib_spec["ports"][1]["width"]/iouter_stream_width)
    ivalues_per_buf = istreams_per_buf*utils.get_proj_stream_count(projection["inner_projection"], 'I')
    ostreams_per_buf = math.floor(ib_spec["ports"][1]["width"]/activation_width)
    ibuf_count = math.ceil(iouter_stream_count/istreams_per_buf)
    obuf_count = math.ceil(ototal_stream_count/ostreams_per_buf)
    ibuf_len = 2**ib_spec["ports"][0]["width"]
    obuf_len = 2**ib_spec["ports"][0]["width"]
        
    # Load the input buffer
    # Several values per word, words per buffer, buffers...
    ibuf = [[[random.randint(0,(2**projection["data_widths"]["I"])-1)
             for k in range(ivalues_per_buf)]    
             for i in range(ibuf_len)]           
             for j in range (ibuf_count)]         
    load_buffers(testinst, "input_act_modules_portawe_{}_top",
            "input_act_modules_portaaddr_top", "input_datain",
            ibuf, projection["data_widths"]["I"])
    
    for n in [0,1]:
        projection = projections[n]
        testinst.sel @= n
        
        # Calculate required buffers etc.
        mlb_count = utils.get_mlb_count(projection["outer_projection"])
        mac_count = utils.get_mlb_count(projection["inner_projection"])
        ibuf_len = 2**ib_spec["ports"][0]["width"]
        obuf_len = 2**ib_spec["ports"][0]["width"]
        wbuf_len = 2**wb_spec["ports"][0]["width"]

        # Load the weight buffer
        wbuf_count = 1
        weight_stream_count = utils.get_proj_stream_count(projection["outer_projection"], 'W')
        wvalues_per_stream = utils.get_proj_stream_count(projection["inner_projection"], 'W')
        wstream_bitwidth = wvalues_per_stream*projection["data_widths"]["W"]
        wstreams_per_buf = math.floor(wb_spec["ports"][1]["width"]/wstream_bitwidth)
        wbuf_count = math.ceil(weight_stream_count / wstreams_per_buf)
        wvalues_per_buf = min(wstreams_per_buf*wvalues_per_stream, wvalues_per_stream*weight_stream_count)
        
        wbuf = [[[random.randint(0,(2**projection["data_widths"]["W"])-1)
                for k in range(wvalues_per_buf)]    # values per word
                for i in range(wbuf_len)]           # words per buffer
                for j in range(wbuf_count)]         # buffer count
        load_buffers(testinst, "weight_modules_portawe_{}_top",
                    "weight_modules_portaaddr_top", "weight_datain",
                    wbuf, projection["data_widths"]["W"])
        print("\nCheck that the on-chip weights are correct")
        check_buffers(testinst, testinst.weight_modules,
                      "ml_block_weights_inst_{}",
                    wbuf, projection["data_widths"]["W"])
        
        # Load the input buffer
        print("\nCheck that the on-chip inputs are correct")
        check_buffers(testinst, testinst.input_act_modules,
                      "ml_block_inputs_inst_{}",
                      ibuf, projection["data_widths"]["I"],
                      buf_start=prev_ibuf_len)
        
        # Now load the weights into the MLBs
        inner_ub = projection["inner_projection"]["B"] * projection["inner_projection"]["PX"] * projection["inner_projection"]["PY"]
        outer_ub = projection["outer_projection"]["B"] * projection["outer_projection"]["PY"] * projection["outer_projection"]["PX"]
        wbi_section_length = projection["inner_projection"]["E"] * \
                            projection["inner_projection"]["C"] * \
                            projection["inner_projection"]["RY"] * \
                            projection["inner_projection"]["RX"]
        wbo_section_length = projection["outer_projection"]["E"] * \
                            projection["outer_projection"]["C"] * \
                            projection["outer_projection"]["RY"] * \
                            projection["outer_projection"]["RX"] *\
                            projection["inner_projection"]["G"] *  wbi_section_length
        starti = 0
        for ugo in range(projection["outer_projection"]["G"]):
            for ubo in range(outer_ub):
                for ugi in range(projection["inner_projection"]["G"] *
                                 projection["outer_projection"]["RX"] *
                                 projection["outer_projection"]["C"] *
                                 projection["outer_projection"]["RY"] *
                                 projection["outer_projection"]["E"]):
                    for ubi in range(inner_ub):
                        wbi_section_start = wbo_section_length*ugo + wbi_section_length*ugi
                        print(wbi_section_start)
                        starti = stream_mlb_values(testinst, wbi_section_length,
                                  ["weight_modules_portaaddr_top"],
                                  [0],
                                  [wbuf_len],
                                  ["mlb_modules_a_en_top"], starti=wbi_section_start)
          
        # Check they are right
        print("\nCheck that the MLB values are right!")
        assert(check_mlb_chains_values(testinst, mlb_count, mac_count, 1, 1,
                                "ml_block_inst_{}", "weight_out_{}", wbuf,
                                projection["data_widths"]["W"],
                                wbo_section_length, outer_ub,
                                       wbi_section_length, inner_ub))
        
        
        # Now stream the inputs, and check the outputs!
        print("Stream!")
        stream_mlb_values(testinst, obuf_len,
                          ["input_act_modules_portaaddr_top", "input_act_modules_portaaddr_o_top"],
                          [0, -1],
                          [ibuf_len, obuf_len],
                          ["mlb_modules_b_en_top"] +
                          ["input_act_modules_portawe_{}_out_top".format((obi+len(ibuf)+prev_ibuf_len) % 2) for obi in range(obuf_count)])
                      
        obuf = [[[0
                 for i in range(ostreams_per_buf)]
                 for i in range(obuf_len)]
                 for j in range (obuf_count)]
        obuf = get_expected_outputs_old(obuf, ostreams_per_buf,
                                    wbuf,
                                    ibuf, ivalues_per_buf,
                                    projection)
        
        obuf_results = [[[[0]
                 for i in range(ostreams_per_buf)]
                 for i in range(obuf_len)]
                 for j in range (obuf_count)]
        
        print("\nCheck that the outputs are correct")
        print("EXPECTED: " + str(obuf))
        
        obuf_results = read_out_stored_values(testinst, "input_act_modules_portaaddr_o_top", "dataout",
                                              obuf_results, projection["data_widths"]["I"], start_buffer=(len(ibuf)+prev_ibuf_len)%2)
        
        print("ACTUAL: " + str(obuf_results))
        print("W: " + str(wbuf))
        print("I: " + str(ibuf))
        ibuf = []
        for bufi in range(obuf_count):
            ibuf = ibuf + [[]]
            for olen in range(min(obuf_len,ibuf_len)-1): #(obuf_len-1):
                assert obuf[bufi][olen] == obuf_results[bufi][olen]
                
                # Now the inputs are the outputs of the previous layer
                ibuf[bufi] = ibuf[bufi] + [obuf_results[bufi][olen]]
                
        prev_ibuf_len = len(ibuf)
#    assert(1==0)

def test_Datapath_pingpong():
    """Test Component class Datapath"""
    projection = {"name": "test",
                  "activation_function": "NONE",
                  "data_widths": {"W": 4,
                                  "I": 4,
                                  "O": 16},
                  "inner_projection": {'C':2,'RX':1, 'RY':1,
                                       'B':1,'PX':2,'PY':1,'E':2,
                                       'G':2,
                                       'PRELOAD':[{'dtype':'W','bus_count':1}]},
                 # "inner_projection": {'C':2,'RX':3, 'RY':1,
                  "outer_projection": {'C':2,'RX':1, 'RY':1,
                                       'B':2,'PX':1,'PY':1,'E':1,
                                       'G':2,
                                       'PRELOAD':[{'dtype':'W','bus_count':1}]}
                  #"outer_projection": {'C':2,'RX':1, 'RY':1,
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
    
    testinst = module_classes.Datapath(
        mlb_spec=mlb_spec,
        wb_spec=wb_spec,
        ib_spec=ib_spec,
        ob_spec=ib_spec,
        proj_specs=[projection],
        pingpong_w=True
    )

    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    testinst.sim_reset()
    testinst.bank_sel @= 0
    
    # Calculate required buffers etc.
    mlb_count = utils.get_mlb_count(projection["outer_projection"])
    mac_count = utils.get_mlb_count(projection["inner_projection"])
    ibuf_len = 2**ib_spec["ports"][0]["width"]
    obuf_len = 2**ib_spec["ports"][0]["width"]
    wbuf_len = 2**wb_spec["ports"][0]["width"]

    # Load the weight buffer
    wbuf_count = 1
    weight_stream_count = utils.get_proj_stream_count(projection["outer_projection"], 'W')
    wvalues_per_stream = utils.get_proj_stream_count(projection["inner_projection"], 'W')
    wstream_bitwidth = wvalues_per_stream*projection["data_widths"]["W"]
    wstreams_per_buf = math.floor(wb_spec["ports"][1]["width"]/wstream_bitwidth)
    wbuf_count = math.ceil(weight_stream_count / wstreams_per_buf)
    wvalues_per_buf = min(wstreams_per_buf*wvalues_per_stream, wvalues_per_stream*weight_stream_count)
    
    wbuf1 = [[[random.randint(0,(2**projection["data_widths"]["W"])-1)
            for k in range(wvalues_per_buf)]    # values per word
            for i in range(wbuf_len)]           # words per buffer
            for j in range(wbuf_count)]         # buffer count
    load_buffers(testinst, "weight_modules_portawe_{}_top",
                "weight_modules_portaaddr_top", "weight_datain",
                wbuf1, projection["data_widths"]["W"])
    check_buffers(testinst, testinst.weight_modules,
                  "ml_block_weights_inst_{}",
                wbuf1, projection["data_widths"]["W"])
    
    wbuf2 = [[[random.randint(0,(2**projection["data_widths"]["W"])-1)
            for k in range(wvalues_per_buf)]    # values per word
            for i in range(wbuf_len)]           # words per buffer
            for j in range(wbuf_count)]         # buffer count
    load_buffers(testinst, "weight_modules_portawe_{}_top",
                "weight_modules_portaaddr_top", "weight_datain",
                 wbuf2, projection["data_widths"]["W"],
                 start_buffer=len(wbuf1))
    check_buffers(testinst, testinst.weight_modules,
                  "ml_block_weights_inst_{}",
                  wbuf2, projection["data_widths"]["W"],
                  buf_start=len(wbuf1))
    
    # Calculate required buffers etc.
    iouter_stream_count = utils.get_proj_stream_count(projection["outer_projection"], 'I')
    iouter_stream_width = utils.get_proj_stream_count(projection["inner_projection"], 'I') * \
                         projection["data_widths"]["I"]
    ototal_stream_count = utils.get_proj_stream_count(projection["outer_projection"], 'O') * \
                          utils.get_proj_stream_count(projection["inner_projection"], 'O') 
    activation_width = projection["data_widths"]["I"]
    istreams_per_buf = math.floor(ib_spec["ports"][1]["width"]/iouter_stream_width)
    ivalues_per_buf = istreams_per_buf*utils.get_proj_stream_count(projection["inner_projection"], 'I')
    ostreams_per_buf = math.floor(ib_spec["ports"][1]["width"]/activation_width)
    ibuf_count = math.ceil(iouter_stream_count/istreams_per_buf)
    obuf_count = math.ceil(ototal_stream_count/ostreams_per_buf)
    
    # Load the input buffer
    # Several values per word, words per buffer, buffers...
    ibuf = [[[random.randint(0,(2**projection["data_widths"]["I"])-1)
             for k in range(ivalues_per_buf)]            # values per word
             for i in range(ibuf_len)]                   # words per buffer
             for j in range (ibuf_count)]                # buffers
    load_buffers(testinst, "input_act_modules_portawe_{}_top",
                "input_act_modules_portaaddr_top", "input_datain",
                ibuf, projection["data_widths"]["I"])
    check_buffers(testinst, testinst.input_act_modules,
                  "ml_block_inputs_inst_{}",
                ibuf, projection["data_widths"]["I"])

    # Now load the weights into the MLBs
    inner_ub = projection["inner_projection"]["B"] * projection["inner_projection"]["PX"] * projection["inner_projection"]["PY"]
    outer_ub = projection["outer_projection"]["B"] * projection["outer_projection"]["PY"] * projection["outer_projection"]["PX"]
    wbi_section_length = projection["inner_projection"]["E"] * \
                        projection["inner_projection"]["C"] * \
                        projection["inner_projection"]["RY"] * \
                        projection["inner_projection"]["RX"]
    wbo_section_length = projection["outer_projection"]["E"] * \
                        projection["outer_projection"]["C"] * \
                        projection["outer_projection"]["RY"] * \
                        projection["outer_projection"]["RX"] *\
                        projection["inner_projection"]["G"] *  wbi_section_length

    testinst.bank_sel @= 0
    for wbuf in [wbuf1,wbuf2]:
        starti = 0
        for ugo in range(projection["outer_projection"]["G"]):
            for ubo in range(outer_ub):
                for ugi in range(projection["inner_projection"]["G"] *
                                 projection["outer_projection"]["RX"] *
                                 projection["outer_projection"]["RY"] *
                                 projection["outer_projection"]["C"] *
                                 projection["outer_projection"]["E"]):
                    for ubi in range(inner_ub):
                        wbi_section_start = wbo_section_length*ugo + wbi_section_length*ugi
                        print(wbi_section_start)
                        starti = stream_mlb_values(testinst, wbi_section_length,
                                  ["weight_modules_portaaddr_top"],
                                  [0],
                                  [wbuf_len],
                                  ["mlb_modules_a_en_top"], starti=wbi_section_start)
          
        # Check they are right
        assert(check_mlb_chains_values(testinst, mlb_count, mac_count, 1, 1,
                                "ml_block_inst_{}", "weight_out_{}", wbuf,
                                projection["data_widths"]["W"],
                                wbo_section_length, outer_ub,
                                       wbi_section_length, inner_ub))
        
        
        # Now stream the inputs, and check the outputs!
        stream_mlb_values(testinst, obuf_len,
                          ["input_act_modules_portaaddr_top", "input_act_modules_portaaddr_o_top"],
                          [0, -1],
                          [ibuf_len, obuf_len],
                          ["mlb_modules_b_en_top"] +
                          ["input_act_modules_portawe_{}_out_top".format(obi+len(ibuf))
                           for obi in range(obuf_count)])
        
        obuf = [[[0
                 for i in range(ostreams_per_buf)]
                 for i in range(obuf_len)]
                 for j in range (obuf_count)]
        obuf = get_expected_outputs_old(obuf, ostreams_per_buf,
                                    wbuf,
                                    ibuf, ivalues_per_buf,
                                    projection)
        
        obuf_results = [[[[0]
                 for i in range(ostreams_per_buf)]
                 for i in range(obuf_len)]
                 for j in range (obuf_count)]
        
        obuf_results = read_out_stored_values(testinst, "input_act_modules_portaaddr_o_top", "dataout",
                                              obuf_results, projection["data_widths"]["I"], start_buffer=len(ibuf))
        
        
        print("EXPECTED: " + str(obuf))
        print("ACTUAL: " + str(obuf_results))
        
        print("W: " + str(wbuf))
        print("I: " + str(ibuf))
        print("Bank")
        print(testinst.bank_sel)
        for bufi in range(obuf_count):
            for olen in range(min(obuf_len,ibuf_len)-1): 
                assert obuf[bufi][olen] == obuf_results[bufi][olen]
        testinst.bank_sel @= 1        
        testinst.sim_tick()
        testinst.sim_tick()
