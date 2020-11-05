#!/usr/bin/env python

"""Tests for `verilog_ml_benchmark_generator` package utilities."""

import pytest
from pymtl3 import *

from click.testing import CliRunner

from verilog_ml_benchmark_generator import utils
from verilog_ml_benchmark_generator import cli

def test_get_var_product():
    """Test util function get_var_product"""
    projection_example = {'A':{'value':5},'B':{'value':4},
                          'C':{'value':-1},'D':{'value':2},
                          'E':{'value':0}}
    assert utils.get_var_product(projection_example,"A") == 5
    assert utils.get_var_product(projection_example,"AB") == 20
    assert utils.get_var_product(projection_example,["B","C"]) == -4
    assert utils.get_var_product(projection_example,"AE") == 0
    assert utils.get_var_product(projection_example,[]) == 1

    with pytest.raises(AssertionError):
        utils.get_var_product(projection_example,["T"])

def test_get_mlb_count():
    """Test util function get_mlb_count"""
    projection_example = {'URN':{'value':1},'URW':{'value':3},
                          'UB':{'value':4},'UE':{'value':2},
                          'UG':{'value':1}}
    assert utils.get_mlb_count(projection_example) == 24
    projection_example = {'URN':{'value':1},'URW':{'value':1},
                          'UB':{'value':1},'UE':{'value':1},
                          'UG':{'value':1}}
    assert utils.get_mlb_count(projection_example) == 1
    
    with pytest.raises(AssertionError):
        projection_example = {'URN':{'value':1},'URW':{'value':1},
                              'UB':{'value':1},'UE':{'value':1}}
        utils.get_mlb_count(projection_example)

def test_get_proj_chain_length():
    """Test util function get_proj_stream_count"""
    projection_example = {'URN':{'value':11},'URW':{'value':2},
                          'UB':{'value':3},'UE':{'value':5},
                          'UG':{'value':7}}
    assert utils.get_proj_chain_length(projection_example,"W") == 2
    assert utils.get_proj_chain_length(projection_example,"I") == 2
    assert utils.get_proj_chain_length(projection_example,"O") == 11*2
    assert utils.get_proj_chain_length(projection_example,"") == 0
    with pytest.raises(AssertionError):
        utils.get_proj_chain_length(projection_example,"M")
    

def test_get_proj_stream_count():
    """Test util function get_proj_stream_count"""
    projection_example = {'URN':{'value':11},'URW':{'value':2},
                          'UB':{'value':3},'UE':{'value':5},
                          'UG':{'value':7}}
    assert utils.get_proj_stream_count(projection_example,"W") == 7*5*2*11
    assert utils.get_proj_stream_count(projection_example,"I") == 7*3*11
    assert utils.get_proj_stream_count(projection_example,"O") == 7*5*3
    assert utils.get_proj_stream_count(projection_example,"WIO") == \
        7*5*3+7*3*11+7*5*2*11
    
    projection_example = {'URN':{'value':11},'URW':{'value':2},
                          'UB':{'value':3},'UE':{'value':5},
                          'UG':{'value':7}, "PRELOAD":[{"dtype":'W', "bus_count":4}]}
    assert utils.get_proj_stream_count(projection_example,"W") == 4
    assert utils.get_proj_stream_count(projection_example,"I") == 7*3*11
    assert utils.get_proj_stream_count(projection_example,"WIO") == 4+7*3*11+7*5*3
    
    projection_example = {'URN':{'value':11},'URW':{'value':2},
                          'UB':{'value':3},'UE':{'value':5},
                          'UG':{'value':7}, "PRELOAD":[{"dtype":'I', "bus_count":3}]}
    assert utils.get_proj_stream_count(projection_example,"W") == 7*5*2*11
    assert utils.get_proj_stream_count(projection_example,"I") == 3
    assert utils.get_proj_stream_count(projection_example,"WIO") == 7*5*2*11+3+7*5*3
    
    projection_example = {'URN':{'value':11},'URW':{'value':2},
                          'UB':{'value':3},'UE':{'value':5},
                          'UG':{'value':7}, "PRELOAD":[{"dtype":'O', "bus_count":2}]}
    assert utils.get_proj_stream_count(projection_example,"W") == 7*5*2*11
    assert utils.get_proj_stream_count(projection_example,"O") == 2
    assert utils.get_proj_stream_count(projection_example,"WIO") == 7*5*2*11+7*3*11+2
    
    with pytest.raises(AssertionError):
        projection_example = {'URN':{'value':1},'URW':{'value':1},
                              'UB':{'value':1},'UE':{'value':1}}
        utils.get_proj_stream_count(projection_example,"O")

def test_get_activation_function_name():
    """Test util function get_activation_function_name"""
    assert utils.get_activation_function_name({"somekey":"D",
             "activation_function":"RELU"}) == "RELU"
    assert utils.get_activation_function_name({"somekey":"D",
             "activation_function":""}) == ""
    
    with pytest.raises(AssertionError):
        utils.get_activation_function_name({"somekey":"D"})

def test_get_ports_of_type():
    """Test util function get_ports_of_type"""
    pA = {"name": "testA", "width":3, "direction":"in", "type":"A"}
    pB = {"name": "testB", "width":4, "direction":"in", "type":"A"}
    pC = {"name": "testC", "width":5, "direction":"out", "type":"A"}
    pJ = {"name": "testJ", "width":5, "direction":"out", "type":"A"}
    pD = {"name": "testD", "width":6, "direction":"in", "type":"B"}
    pE = {"name": "testE", "width":7, "direction":"out", "type":"B"}
    pF = {"name": "testF", "width":8, "direction":"out", "type":"C"}
    pG = {"name": "testG", "width":9, "direction":"out", "type":"C"}
    pH = {"name": "testH", "width":8, "direction":"in", "type":"D"}
    pI = {"name": "testI", "width":9, "direction":"in", "type":"D"}
    example_hw = {
        "ports": [pA, pB, pC, pD, pE, pF, pG, pH, pI, pJ]
        }
    assert list(utils.get_ports_of_type(example_hw, 'A',["in"])) == [pA, pB]
    assert list(utils.get_ports_of_type(example_hw, 'B',["in"])) == [pD]
    assert list(utils.get_ports_of_type(example_hw, 'C',["in"])) == []
    assert list(utils.get_ports_of_type(example_hw, 'D',["in"])) == [pH,pI]
    assert list(utils.get_ports_of_type(example_hw, 'A',["out"])) == [pC, pJ]
    assert list(utils.get_ports_of_type(example_hw, 'B',["out"])) == [pE]
    assert list(utils.get_ports_of_type(example_hw, 'C',["out"])) == [pF, pG]
    assert list(utils.get_ports_of_type(example_hw, 'D',["out"])) == []
    assert list(utils.get_ports_of_type(example_hw, 'A')) == [pA, pB, pC, pJ]
    assert list(utils.get_ports_of_type(example_hw, 'B')) == [pD, pE]
    assert list(utils.get_ports_of_type(example_hw, 'C')) == [pF,pG]
    assert list(utils.get_ports_of_type(example_hw, 'D')) == [pH, pI]
    assert list(utils.get_ports_of_type(example_hw, 'A',[])) == []
    assert list(utils.get_ports_of_type(example_hw, '')) == []

def test_get_sum_datatype_width():
    """Test util function get_sum_datatype_width"""
    pA = {"name": "testA", "width":3, "direction":"in", "type":"A"}
    pB = {"name": "testB", "width":4, "direction":"in", "type":"A"}
    pC = {"name": "testC", "width":5, "direction":"out", "type":"A"}
    pJ = {"name": "testJ", "width":5, "direction":"out", "type":"A"}
    pD = {"name": "testD", "width":6, "direction":"in", "type":"B"}
    pE = {"name": "testE", "width":7, "direction":"out", "type":"B"}
    pF = {"name": "testF", "width":8, "direction":"out", "type":"C"}
    pG = {"name": "testG", "width":9, "direction":"out", "type":"C"}
    pH = {"name": "testH", "width":8, "direction":"in", "type":"D"}
    pI = {"name": "testI", "width":9, "direction":"in", "type":"D"}
    example_hw = {
        "ports": [pA, pB, pC, pD, pE, pF, pG, pH, pI, pJ]
        }
    assert utils.get_sum_datatype_width(example_hw, 'A',["in"])  == 7
    assert utils.get_sum_datatype_width(example_hw, 'B',["in"])  == 6
    assert utils.get_sum_datatype_width(example_hw, 'C',["in"])  == 0
    assert utils.get_sum_datatype_width(example_hw, 'D',["in"])  == 17
    assert utils.get_sum_datatype_width(example_hw, 'A',["out"]) == 10
    assert utils.get_sum_datatype_width(example_hw, 'B',["out"]) == 7
    assert utils.get_sum_datatype_width(example_hw, 'C',["out"]) == 17
    assert utils.get_sum_datatype_width(example_hw, 'D',["out"]) == 0
    assert utils.get_sum_datatype_width(example_hw, 'A')         == 7+10
    assert utils.get_sum_datatype_width(example_hw, 'B')         == 6+7
    assert utils.get_sum_datatype_width(example_hw, 'C')         == 17
    assert utils.get_sum_datatype_width(example_hw, 'D')         == 17
    assert utils.get_sum_datatype_width(example_hw, 'A',[])      == 0
    assert utils.get_sum_datatype_width(example_hw, '')          == 0

def test_get_num_buffers_reqd():
    """Test util function get_num_buffers_reqd"""  
    pA = {"name": "testA", "width":3, "direction":"in", "type":"A"}
    pB = {"name": "testD", "width":6, "direction":"in", "type":"B"}
    pC = {"name": "testF", "width":8, "direction":"out", "type":"C"}
    pD = {"name": "dout0", "width":16, "direction":"in", "type":"DATA"}
    pE = {"name": "dout1", "width":8, "direction":"in", "type":"DATA"}
    example_hw = {
        "ports": [pA, pB, pC, pD, pE]
    }
    assert utils.get_num_buffers_reqd(example_hw, 1, 4) == 1 
    assert utils.get_num_buffers_reqd(example_hw, 20, 5) == 5
    assert utils.get_num_buffers_reqd(example_hw, 3, 5) == 1
    assert utils.get_num_buffers_reqd(example_hw, 60, 4) == 10
    assert utils.get_num_buffers_reqd(example_hw, 60, 5) == 15
    
    with pytest.raises(AssertionError):
        utils.get_num_buffers_reqd(example_hw, 1, 40)
    
    with pytest.raises(AssertionError):
        example_hw = {
            "ports": [pA, pB, pC]
        }
        utils.get_num_buffers_reqd(example_hw, 1, 1)

def test_get_overall_idx():
    """Test util function get_overall_idx"""
    projection_example = {'URN':{'value':5},'URW':{'value':6},
                          'UB':{'value':7},'UE':{'value':8},
                          'UG':{'value':9}}
    assert utils.get_overall_idx(projection_example,
                                 {"URN":3,"URW":2,"UG":1}) == 1*6*5+2+3*6
    assert utils.get_overall_idx(projection_example,
                                 {"UE":3,"URN":2,"UG":2}) == 2*5*8+3*5+2
    assert utils.get_overall_idx(projection_example, {}) == 0


    with pytest.raises(AssertionError):
        projection_example = {'URN':{'value':1},'URW':{'value':1},
                              'UB':{'value':1},'UE':{'value':1},
                              'UG':{'value':1}, "A":{"value":0}}
        utils.get_overall_idx(projection_example, {"A":1})
        
    with pytest.raises(AssertionError):
        projection_example = {'URN':{'value':1},'URW':{'value':1},
                          'UB':{'value':1},'UE':{'value':1}}
        utils.get_overall_idx(projection_example, {"URN":3,"URW":6})

    with pytest.raises(AssertionError):
        projection_example = {'URN':{'value':1},'URW':{'value':1},
                          'UB':{'value':1},'UE':{'value':1},
                          'UG':{'value':3}}
        utils.get_overall_idx(projection_example, {"URN":0,"UG":3})

    with pytest.raises(AssertionError):
        projection_example = {'URN':{'value':1},'URW':{'value':1},
                          'UB':{'value':1},'UE':{'value':1},
                          'UG':{'value':3}}
        utils.get_overall_idx(projection_example, {"URN":0,"UG":-1})
        

def test_connect_in_out_to_top():
    """Test util function connect_in_to_top and connect_out_to_top"""

    class test_function(Component):
        def construct(s, width_i=1, width_o=1):  
            # Shorten the module name to the provided name.
            s.function_in = InPort(width_i)
            s.function_out = OutPort(width_o)
            connected_width = min(width_i, width_o)
            s.function_out[0:connected_width] //= s.function_in[0:connected_width]
        
    class TestWrapper(Component):
        def construct(s, wi, wo):
            s.f = test_function(wi,wo)
            utils.connect_in_to_top(s,s.f.function_in, "function_in_outer")
            utils.connect_in_to_top(s,s.f.clk, "outer_clk")
            utils.connect_out_to_top(s,s.f.function_out, "function_out_outer")
            
    testinst = TestWrapper(4,4)
    testinst.elaborate()
    assert(len(list(testinst.get_input_value_ports())) == 3)
    assert(len(list(testinst.get_output_value_ports())) == 1)
    testinst.apply(DefaultPassGroup())
    testinst.function_in_outer @= 8
    testinst.sim_tick()
    assert testinst.function_out_outer == Bits4(8)
    
    testinst = TestWrapper(3,4)
    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    testinst.function_in_outer @= 7
    testinst.sim_tick()
    assert testinst.function_out_outer == Bits4(7)

    testinst = TestWrapper(4,3)
    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    testinst.function_in_outer @= 8
    testinst.sim_tick()
    assert testinst.function_out_outer == Bits3(0)
    testinst.function_in_outer @= 7
    testinst.sim_tick()
    assert testinst.function_out_outer == Bits3(7)
    

def test_add_n_inputs_outputs():
    """Test util functions add_n_inputs and add_n_outputs"""
    class test_function(Component):
        def construct(s, n=1):
            # Shorten the module name to the provided name.
            s.IN0 = InPort(3)
            s.OUT5 = OutPort(3)
            utils.add_n_inputs(s, n, 3, "IN", 0)
            utils.add_n_outputs(s, n, 3, "OUT", 5)
            for i in range(n):
                ip = getattr(s, "IN"+str(i))
                op = getattr(s, "OUT"+str(i+5))
                op //= ip
    N = 3
    testinst = test_function(N)
    testinst.elaborate()
    assert(len(list(testinst.get_input_value_ports())) == 2+N)
    assert(len(list(testinst.get_output_value_ports())) == N)
    testinst.apply(DefaultPassGroup())
    testinst.IN0 @= 1 
    testinst.IN1 @= 2 
    testinst.IN2 @= 3 
    assert testinst.OUT5 == Bits3(1)
    assert testinst.OUT6 == Bits3(2)
    assert testinst.OUT7 == Bits3(3)

def test_tie_off_clk_reset():
    """Test util function tie_off_clk_reset"""
    class test_function(Component):
        def construct(s, n=1):
            # Shorten the module name to the provided name.
            utils.tie_off_clk_reset(s)
    testinst = test_function()
    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    testinst.reset @= 1
    testinst.clk @= 0
    assert testinst.clk_tieoff == 0
    assert testinst.reset_tieoff == 1
    testinst.reset @= 0
    testinst.clk @= 1
    assert testinst.reset_tieoff == 0
    assert testinst.clk_tieoff == 1
    
def test_connect_ports_by_name():
    """Test util function connect_ports_by_name"""
    
    class test_function1(Component):
        def construct(s, n=1):
            # Shorten the module name to the provided name.
            utils.add_n_inputs(s, n, 3, "IN_", 0)
            utils.add_n_outputs(s, n, 3, "OUT_", 0)
            s.YOUT = OutPort(3)
            s.YOUT //= s.IN_0
            for i in range(n):
                ip = getattr(s, "IN_"+str(i))
                op = getattr(s, "OUT_"+str(i))
                op //= ip
                
    class test_function2(Component):
        def construct(s, n=1):
            # Shorten the module name to the provided name.
            s.IN_0 = InPort(3)
            s.IN_2 = InPort(3)
            s.IN_4 = InPort(3)
            utils.add_n_outputs(s, n, 3, "OUT_", 0)
            s.YIN_0 = InPort(3)
            s.YIN_2 = InPort(3)
            s.YIN_4 = InPort(3)
            utils.add_n_outputs(s, n, 3, "YOUT_", 0)
            for i in range(n):
                ip = getattr(s, "IN_"+str(i*2))
                op = getattr(s, "OUT_"+str(i))
                op //= ip
                ip = getattr(s, "YIN_"+str(i*2))
                op = getattr(s, "YOUT_"+str(i))
                op //= ip

    class TestWrapper(Component):
        def construct(s, i_f, o_f):
            s.f1 = test_function1(3)
            s.f2 = test_function2(3)
            utils.connect_in_to_top(s,s.f1.IN_0, "IN0_outer")
            utils.connect_in_to_top(s,s.f1.IN_1, "IN1_outer")
            utils.connect_in_to_top(s,s.f1.IN_2, "IN2_outer")
            utils.connect_out_to_top(s,s.f2.OUT_0, "OUT0_outer")
            utils.connect_out_to_top(s,s.f2.OUT_1, "OUT1_outer")
            utils.connect_out_to_top(s,s.f2.OUT_2, "OUT2_outer")
            utils.connect_ports_by_name(s.f1,"OUT",s.f2,"IN", i_f, o_f)
            utils.connect_out_to_top(s,s.f2.YOUT_0, "YOUT0_outer")
            utils.connect_out_to_top(s,s.f2.YOUT_1, "YOUT1_outer")
            utils.connect_out_to_top(s,s.f2.YOUT_2, "YOUT2_outer")
            utils.connect_ports_by_name(s.f1,"YOUT",s.f2,"YIN", i_f, o_f)
    
    testinst = TestWrapper(1,1)
    with pytest.raises(AssertionError):
        testinst.elaborate()
    testinst = TestWrapper(1,2)
    with pytest.raises(AssertionError):
        testinst.elaborate()
    testinst = TestWrapper(2,1)
    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    testinst.IN0_outer @= 1 
    testinst.IN1_outer @= 2 
    testinst.IN2_outer @= 3 
    testinst.sim_tick()
    testinst.OUT0_outer @= 1 
    testinst.OUT1_outer @= 2 
    testinst.OUT2_outer @= 3 

def test_connect_inst_ports_by_name():
    """Test util function connect_ports_by_name"""
    
    class test_function1(Component):
        def construct(s, n=1):
            # Shorten the module name to the provided name.
            utils.add_n_inputs(s, n, 3, "IN_", 0)
            utils.add_n_outputs(s, n, 3, "OUT_", 0)
            s.YOUT = OutPort(3)
            s.YOUT //= s.IN_0
            for i in range(n):
                ip = getattr(s, "IN_"+str(i))
                op = getattr(s, "OUT_"+str(i))
                op //= ip

    class TestWrapper(Component):
        def construct(s):
            s.f1 = test_function1(3)
            utils.connect_in_to_top(s,s.f1.IN_0, "IN0_outer")
            utils.connect_in_to_top(s,s.f1.IN_1, "IN1_outer")
            utils.connect_in_to_top(s,s.f1.IN_2, "IN2_outer")
            utils.add_n_outputs(s, 3, 3, "OUT_outer_", 0)
            utils.connect_inst_ports_by_name(s, "OUT_outer", s.f1, "OUT")

    testinst = TestWrapper()
    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    testinst.IN0_outer @= 1 
    testinst.IN1_outer @= 2 
    testinst.IN2_outer @= 3 
    testinst.sim_tick()
    testinst.OUT_outer_0 @= 1 
    testinst.OUT_outer_1 @= 2 
    testinst.OUT_outer_2 @= 3 

