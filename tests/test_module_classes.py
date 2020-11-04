#!/usr/bin/env python

"""Tests for `verilog_ml_benchmark_generator` pyMTL Components."""

import pytest
from pymtl3 import *

from click.testing import CliRunner

from verilog_ml_benchmark_generator import utils
from verilog_ml_benchmark_generator import module_classes
from verilog_ml_benchmark_generator import cli


def test_RELU():
    """Test Component class RELU"""
    test_vecs = [
        {"ins":[4,4, False], "outs":[[7,7],[8,0]]},
        {"ins":[4,2, False], "outs":[[3,3],[5,1],[9,0]]},
        {"ins":[3,5, False], "outs":[[3,Bits5(3)],[4,Bits5(0)]]},
        {"ins":[4,4, True], "outs":[[7,7],[8,0]]},
        {"ins":[4,2, True], "outs":[[3,3],[5,1],[9,0]]},
        {"ins":[3,5, True], "outs":[[3,Bits5(3)],[4,Bits5(0)]]},
    ]
    for testvec in test_vecs:
        testinst = module_classes.RELU(testvec["ins"][0],
                                       testvec["ins"][1],
                                       testvec["ins"][2])
        testinst.elaborate()
        testinst.apply(DefaultPassGroup()) 
        for pair in testvec["outs"]:
            testinst.sim_reset()
            testinst.activation_function_in @= pair[0]
            testinst.sim_tick()
            assert testinst.activation_function_out == pair[1]
    
def test_ActivationWrapper():
    """Test Component class RELU"""
    test_vecs = [
        {"ins":[4,4, False], "outs":[[7,7],[8,0]]},
        {"ins":[4,2, False], "outs":[[3,3],[5,1],[9,0]]},
        {"ins":[3,5, False], "outs":[[3,Bits5(3)],[4,Bits5(0)]]},
        {"ins":[4,4, True], "outs":[[7,7],[8,0]]},
        {"ins":[4,2, True], "outs":[[3,3],[5,1],[9,0]]},
        {"ins":[3,5, True], "outs":[[3,Bits5(3)],[4,Bits5(0)]]},
    ]
    for testvec in test_vecs:
        testinst = module_classes.ActivationWrapper(len(testvec["outs"]), "RELU",
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

    testinst = module_classes.ActivationWrapper(2, "notrelu",
                                                test_vecs[0]["ins"][0],
                                                test_vecs[0]["ins"][1],
                                                test_vecs[0]["ins"][2])
                                                
    with pytest.raises(AssertionError):
        testinst.elaborate()
    

def test_HWB():
    """Test Component class HWB"""
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
    proj_spec = {'URN':{'value':1},'URW':{'value':1},
                  'UB':{'value':1},'UE':{'value':1},
                  'UG':{'value':1}}
    testinst = module_classes.HWB_Sim(spec, proj_spec)
    testinst.elaborate()
        
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
                        testvec["ins"][4])
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
        {"ins":[2,4,5,8,3]}, # ins per out not possible 
        {"ins":[2,8,16,3,2]}, # not enough outputs 
        {"ins":[2,8,6,2]} # no possible solution 
    ]  
    for testvec in illegal_test_vecs:
        if len(testvec["ins"]) == 5:
            testinst = module_classes.MergeBusses(testvec["ins"][0],
                        testvec["ins"][1], testvec["ins"][2], testvec["ins"][3],
                        testvec["ins"][4])
        else:
            testinst = module_classes.MergeBusses(testvec["ins"][0],
                        testvec["ins"][1], testvec["ins"][2], testvec["ins"][3])
        with pytest.raises(AssertionError):
            testinst.elaborate()
        
def test_WeightInterconnect():
    """Test Component class WeightInterconnect"""
    test_vecs = [ # bufferwidth, mlbwidth, mlbwidthused, num_buffers, num_mlbs, proj.
        {"ins":[8,8,3,4,10,{'URN':{'value':2},'URW':{'value':1},
                          'UB':{'value':2},'UE':{'value':2},
                          'UG':{'value':1}}],  
         "outs":[[8,26,44,62],[0,1,2,3,0,1,2,3,0,0]]},
        {"ins":[8,8,3,4,15,{'URN':{'value':1},'URW':{'value':2},
                          'UB':{'value':3},'UE':{'value':1},
                          'UG':{'value':2}}],  
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
        {"ins":[8,2,3,4,10,{'URN':{'value':2},'URW':{'value':1},
                          'UB':{'value':2},'UE':{'value':2},
                          'UG':{'value':1}}]}, # mlb bitwidths dont make sense
        {"ins":[8,8,3,400,15,{'URN':{'value':1},'URW':{'value':2},
                          'UB':{'value':3},'UE':{'value':10},
                          'UG':{'value':2}}]}, # not enough mlbs
        {"ins":[8,8,3,4,1000,{'URN':{'value':1},'URW':{'value':2},
                          'UB':{'value':3},'UE':{'value':10},
                          'UG':{'value':2}}]}, # not enough buffers
        {"ins":[5,8,3,3,15,{'URN':{'value':1},'URW':{'value':2},
                          'UB':{'value':3},'UE':{'value':1},
                          'UG':{'value':2}}]}, # buffer not wide enough
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
        {"ins":[8,8,3,2,10,{'URN':{'value':1},'URW':{'value':2},
                          'UB':{'value':4},'UE':{'value':1},
                          'UG':{'value':1}}],  
         "outs":[[62,26],[1,2,3,4,5,6,7,8,9,10],[6,1,7,3,2,5,3,7,0,0]]},
        {"ins":[8,8,3,2,10,{'URN':{'value':1},'URW':{'value':2},
                          'UB':{'value':2},'UE':{'value':2},
                          'UG':{'value':1}}],  
         "outs":[[62,26],[1,2,3,4,5,6,7,8,9,10],[6,1,6,3,7,5,7,7,0,0]]}
    ]
    for testvec in test_vecs:
        testinst = module_classes.InputInterconnect(testvec["ins"][0],
                    testvec["ins"][1], testvec["ins"][2], testvec["ins"][3],
                    testvec["ins"][4], testvec["ins"][5])
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
            assert out_bus == testvec["outs"][2][i]
        
def test_OutputPSInterconnect():
    """Test Component class InputInterconnect"""
    test_vecs = [ # afwidth, mlbwidth, mlbwidthused, num_afs, num_mlbs, proj.
        {"ins":[3,10,6,6,10,{'URN':{'value':2},'URW':{'value':2},
                          'UB':{'value':2},'UE':{'value':1},
                          'UG':{'value':1}}],  
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
                  "activation_function": "RELU",
                  "stream_info": {"W": 8,
                                  "I": 8,
                                  "O": 32},
                  "outer_projection": {'URN':{'value':2},'URW':{'value':1},
                                       'UB':{'value':2},'UE':{'value':4},
                                       'UG':{'value':1}},
                  "inner_projection": {'URN':{'value':2},'URW':{'value':2},
                                       'UB':{'value':2},'UE':{'value':1},
                                       'UG':{'value':1}}
                  }
    wb_spec = {
        "block_name": "ml_block_weights",
        "ports": [
            {"name":"portaaddr", "width":13, "direction": "in", "type":"ADDRESS"},
            {"name":"portadatain", "width":128, "direction": "in", "type":"DATAIN"},
            {"name":"portadataout", "width":128, "direction": "out", "type":"DATAOUT"},
            {"name":"portawe", "width":1, "direction": "out", "type":"C"},
         ]
    }
    ib_spec = {
        "block_name": "ml_block_inputs",
        "ports": [
            {"name":"portaaddr", "width":8, "direction": "in", "type":"ADDRESS"},
            {"name":"portadatain", "width":32, "direction": "in", "type":"DATAIN"},
            {"name":"portadataout", "width":32, "direction": "out", "type":"DATAOUT"},
            {"name":"portawe", "width":1, "direction": "in", "type":"C"},
        ]
    }
    ob_spec = {
        "block_name": "ml_block_outputs",
        "ports": [
            {"name":"portaaddr", "width":8, "direction": "in", "type":"ADDRESS"},
            {"name":"portadatain", "width":16, "direction": "in", "type":"DATAIN"},
            {"name":"portadataout", "width":16, "direction": "out", "type":"DATAOUT"},
            {"name":"portawe", "width":1, "direction": "out", "type":"C"},
        ]
    }
    mlb_spec = {
        "block_name": "ml_block",
        "simulation_model": "MLB",
        "MAC_info": { "num_units": 12, "data_widths": {"W":8, "I":8, "O": 32} },
        "ports": [
            {"name":"a_in", "width":32, "direction": "in", "type":"W"},
            {"name":"a_out", "width":32, "direction": "out", "type":"W"},
            {"name":"b_in", "width":32, "direction": "in", "type":"I"},
            {"name":"b_out", "width":32, "direction": "out", "type":"I"},
            {"name":"res_in", "width":128, "direction": "in", "type":"O"},
            {"name":"res_out", "width":128, "direction": "out", "type":"O"},
        ]
    }
    
    testinst = module_classes.Datapath(
        mlb_spec=mlb_spec,
        wb_spec=wb_spec,
        ib_spec=ib_spec,
        ob_spec=ob_spec,
        proj_spec=projection)
    
    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    
    testinst.sim_reset()
    #for i in range(len(testvec["outs"][0])):
    #    in_bus = getattr(testinst, "inputs_from_mlb_"+str(i))
    #    in_bus @= testvec["outs"][0][i]
    #testinst.sim_tick()
    #for i in range(len(testvec["outs"][1])):
    #    out_bus = getattr(testinst, "outputs_to_afs_"+str(i))
    #    assert out_bus == Bits3(testvec["outs"][1][i])
    #for i in range(len(testvec["outs"][2])):
    #    out_bus = getattr(testinst, "outputs_to_mlb_"+str(i))
    #    assert out_bus == testvec["outs"][2][i]
