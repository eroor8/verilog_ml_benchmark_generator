#!/usr/bin/env python

"""Tests for `verilog_ml_benchmark_generator` pyMTL helper Components."""

import pytest
import random
from pymtl3 import *

from click.testing import CliRunner

from verilog_ml_benchmark_generator import utils
from verilog_ml_benchmark_generator import module_helper_classes
from verilog_ml_benchmark_generator import cli


def test_Register():
    """Test Component class Register"""
    testinst = module_helper_classes.Register(8)
    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    testinst.sim_reset()
    assert testinst.output_data == 0
    testinst.input_data @= 5
    testinst.sim_tick()
    assert testinst.output_data == 0
    testinst.input_data @= 5
    testinst.ena @= 1
    testinst.sim_tick()
    assert testinst.output_data == 5
    
def test_ShiftRegister():
    """Test Component class ShiftRegister"""
    testinst = module_helper_classes.ShiftRegister(8,0)
    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    testinst.sim_reset()
    assert testinst.output_data == 0
    testinst.input_data @= 5
    testinst.ena @= 1
    testinst.sim_tick()
    assert testinst.output_data == 5
    
    testinst = module_helper_classes.ShiftRegister(8,5)
    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    testinst.sim_reset()
    assert testinst.output_data == 0
    testinst.input_data @= 1
    testinst.ena @= 1
    testinst.sim_tick()
    testinst.input_data @= 2
    assert testinst.output_data == 0
    testinst.sim_tick()
    testinst.input_data @= 3
    assert testinst.output_data == 0
    testinst.sim_tick()
    testinst.input_data @= 4
    assert testinst.output_data == 0
    testinst.sim_tick()
    testinst.input_data @= 5
    assert testinst.output_data == 0
    testinst.sim_tick()
    testinst.input_data @= 6
    assert testinst.output_data == 1
    testinst.sim_tick()
    testinst.input_data @= 3
    testinst.ena @= 0
    assert testinst.output_data == 2
    testinst.sim_tick()
    testinst.input_data @= 3
    assert testinst.output_data == 2
    
def test_MAC():
    """Test Component class RELU"""
    
    testinst = module_helper_classes.MAC(8,4,32)
    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    testinst.sim_reset()
    
    # Test weight stationary
    testinst.sim_reset()
    testinst.acc_en @= 0
    assert testinst.weight_out == 0
    assert testinst.input_out == 0
    assert testinst.sum_out == 0
    testinst.weight_in @= 3
    testinst.weight_en @= 1
    testinst.sum_in @= 1
    testinst.sim_tick()
    assert testinst.weight_out == 3
    assert testinst.input_out == 0
    assert testinst.sum_out == 1
    testinst.input_in @= 6
    testinst.input_en @= 1
    testinst.sim_tick()
    assert testinst.weight_out == 3
    assert testinst.input_out == 6
    assert testinst.weight_out_w == 3
    assert testinst.input_out_w == 6
    assert testinst.sum_out == 1+6*3
    testinst.input_in @= 7
    testinst.input_en @= 1
    testinst.sim_tick()
    assert testinst.weight_out == 3
    assert testinst.input_out == 7
    assert testinst.sum_out == 1+7*3
    
    # Test output stationary
    testinst.acc_en @= 1
    testinst.input_en @= 0
    testinst.sim_reset()
    assert testinst.input_out == 0
    assert testinst.sum_out == 0
    testinst.weight_in @= 2
    testinst.input_in @= 6
    testinst.input_en @= 1
    testinst.weight_en @= 1
    testinst.sim_tick()
    assert testinst.weight_out == 2
    assert testinst.input_out == 6
    assert testinst.sum_out == 12
    testinst.input_in @= 3
    testinst.weight_in @= 3
    testinst.input_en @= 1
    testinst.sim_tick()
    assert testinst.weight_out == 3
    assert testinst.input_out == 3
    assert testinst.sum_out == 12+9
            

def test_MLB():
    """Test Component class MLB"""
    # URW chain - weight stationary
    proj_legal = {"name": "test",
                  "stream_info": {"W": 8, "I": 8, "O": 8},
                  "inner_projection": {'URN':{'value':1},'URW':{'value':4},
                                       'UB':{'value':1},'UE':{'value':1},
                                       'UG':{'value':1},
                                       "PRELOAD":[{'dtype':'W','bus_count':1}]}
    }
    testinst = module_helper_classes.MLB(proj_legal)
    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    testinst.sim_reset()
    testinst.W_IN @= 0
    testinst.W_EN @= 0
    testinst.I_IN @= 0
    testinst.I_EN @= 0
    testinst.I_IN @= 0
    testinst.I_EN @= 0
    testinst.ACC_EN @= 0
    
    # Load weights
    testinst.W_EN @= 1
    for cycle in range(8):
        assert testinst.O_OUT == 0
        testinst.W_IN @= 1
        testinst.sim_tick()
    assert testinst.mac_modules.MAC_inst_0.weight_out == 1
    assert testinst.mac_modules.MAC_inst_1.weight_out == 1
    assert testinst.mac_modules.MAC_inst_2.weight_out == 1
    assert testinst.mac_modules.MAC_inst_3.weight_out == 1
    
    # Preload weights, stream inputs, outputs
    testinst.W_EN @= 0
    testinst.I_EN @= 1
    testinst.I_IN @= 6
    testinst.sim_tick()
    assert testinst.mac_modules.MAC_inst_0.input_out == 6
    assert testinst.mac_modules.MAC_inst_0.weight_out == 1
    assert testinst.mac_modules.MAC_inst_3.sum_in == 6
    assert testinst.mac_modules.MAC_inst_3.sum_out == 6
    assert testinst.mac_modules.sum_out_3 == 6
    assert testinst.output_ps_interconnect.inputs_from_mlb_3 == 6
    assert testinst.output_ps_interconnect.outputs_to_afs_0 == 6
    assert testinst.output_interconnect.input_0 == 6
    assert testinst.output_interconnect.output_0 == 6
    assert testinst.O_OUT == 6
    testinst.W_EN @= 0
    testinst.I_IN @= 3
    testinst.sim_tick()
    assert testinst.O_OUT == 6+3
    testinst.W_EN @= 0
    testinst.I_IN @= 2
    testinst.sim_tick()
    assert testinst.O_OUT == 6+3+2
    testinst.W_EN @= 0
    testinst.I_IN @= 1
    testinst.sim_tick()
    assert testinst.O_OUT == 6+3+2+1
    testinst.W_EN @= 0
    testinst.I_IN @= 9
    testinst.sim_tick()
    assert testinst.O_OUT == 3+2+1+9
    testinst.W_EN @= 0
    testinst.I_IN @= 8
    testinst.sim_tick()
    assert testinst.O_OUT == 2+1+9+8


def test_BufferValue():
    """Test Component class BufferValue"""
    
    testinst = module_helper_classes.BufferValue(8,4,800)
    with pytest.raises(AssertionError):
        testinst.elaborate()
        
    testinst = module_helper_classes.BufferValue(8,4,8)
    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    testinst.sim_reset()
    testinst.datain @= 10
    assert testinst.dataout == 0
    testinst.sim_tick()
    assert testinst.dataout == 0
    testinst.wen @= 1
    testinst.sim_tick()
    assert testinst.dataout == 0
    testinst.address @= 9
    testinst.sim_tick()
    assert testinst.dataout == 0
    testinst.address @= 8
    testinst.sim_tick()
    assert testinst.dataout == 10
    testinst.address @= 8
    testinst.datain @= 9
    testinst.sim_tick()
    assert testinst.dataout == 9
    testinst.datain @= 20
    testinst.address @= 7
    testinst.sim_tick()
    assert testinst.dataout == 9

def test_Buffer():
    """Test Component class Buffer"""
        
    testinst = module_helper_classes.Buffer(datawidth=3,length=6)
    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    testinst.sim_reset()
    testinst.datain @= 3
    assert testinst.data == 0
    assert testinst.dataout == 0
    testinst.address @= 0
    testinst.wen @= 1
    testinst.sim_tick()
    testinst.sim_tick()
    assert testinst.data == 3
    assert testinst.dataout == 3
    testinst.address @= 1
    testinst.datain @= 1
    testinst.wen @= 1
    testinst.sim_tick()
    testinst.sim_tick()
    assert testinst.data == 11
    assert testinst.dataout == 1
    testinst.address @= 0
    testinst.datain @= 4
    testinst.wen @= 0
    testinst.sim_tick()
    testinst.sim_tick()
    assert testinst.data == 11
    assert testinst.dataout == 3
    testinst.address @= 2
    testinst.sim_tick()
    testinst.sim_tick()
    assert testinst.data == 11
    assert testinst.dataout == 0

    testinst = module_helper_classes.Buffer(datawidth=3,length=6,startaddr=6,
                                            preload_vector=[5,5,5])
    testinst.elaborate()
    testinst.apply(DefaultPassGroup())
    testinst.sim_reset()
    testinst.datain @= 3
    assert testinst.data == 365
    assert testinst.dataout == 0
    testinst.address @= 0
    testinst.wen @= 1
    testinst.sim_tick()
    testinst.sim_tick()
    assert testinst.data == 365
    assert testinst.dataout == 0
    testinst.address @= 6
    testinst.wen @= 1
    testinst.sim_tick()
    testinst.sim_tick()
    assert testinst.data == 363
    assert testinst.dataout == 3
    testinst.address @= 7
    testinst.wen @= 0
    testinst.sim_tick()
    testinst.sim_tick()
    assert testinst.data == 363
    assert testinst.dataout == 5
    testinst.address @= 6
    testinst.sim_tick()
    testinst.sim_tick()
    assert testinst.data == 363
    assert testinst.dataout == 3
