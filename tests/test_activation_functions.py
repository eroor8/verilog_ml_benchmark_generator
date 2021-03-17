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
from verilog_ml_benchmark_generator import activation_functions
from verilog_ml_benchmark_generator import cli

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_helpers import *

def test_RELU():
    """Test Component class RELU"""
    test_vecs = [
        {"ins":[4,4, True, 0, 0], "outs":[[7,7],[8,0]]}, # 111 -> 111, 1000 -> 0 (same width, qs=0)
        {"ins":[5,4, False, 0, 0], "outs":[[15,15],[17,0]]}, # 01111 -> 1111, 10001 -> 0000 (shorter output, qs=0)
        {"ins":[4,2, False, 0, 0], "outs":[[3,3],[5,0],[9,0]]}, # 0011 -> 11, 0101 --> 0, 1001 --> 0   (even shorter output, qs=0)
        {"ins":[3,5, False, 0, 0], "outs":[[3,3],[4,0]]}, # 011 -> 00011, 100 --> 00000 (shorter input, qs= 0)
        {"ins":[4,4, True, 2, 2], "outs":[[7,7],[8,0]]}, # 111 -> 111, 1000 -> 0 (same width, same qs)
        {"ins":[5,4, False, 2, 2], "outs":[[15,15],[17,0]]}, # 01111 -> 1111, 10001 -> 0000 (shorter output, same qs)
        {"ins":[4,2, False, 2, 1], "outs":[[3,1],[5,2],[9,0]]}, # 0011 -> 01, 0101 --> 10, 1001 --> 0   (shorter output, shorter qs)
        {"ins":[3,5, False, 1, 0], "outs":[[3,1],[4,0]]}, # 011 -> 00001, 100 --> 00000 (longer output, shorter qs)
        {"ins":[3,5, False, 1, 3], "outs":[[3,12],[4,0]]}, # 011 -> 01100, 100 --> 00000 (longer output, longer qs)
        {"ins":[3,6, False, 1, 3], "outs":[[3,12],[4,0]]}, # 011 -> 01100, 100 --> 00000 (longer output, longer qs, longeri)
        {"ins":[3,5, False, 1, 1], "outs":[[3,3],[4,0]]}, # 011 -> 00011, 100 --> 00000 (longer output, same qs)
    ]
    i = 0
    for testvec in test_vecs:
        print("VEC " + str(testvec))
        testinst = activation_functions.RELU(testvec["ins"][0],
                                       testvec["ins"][1],
                                       testvec["ins"][2],
                                       testvec["ins"][3],
                                       testvec["ins"][4])
        testinst.elaborate()
        testinst.apply(DefaultPassGroup()) 
        for pair in testvec["outs"]:
            testinst.sim_reset()
            testinst.activation_function_in @= pair[0]
            testinst.sim_tick()
            assert testinst.activation_function_out == pair[1]
            assert activation_functions.RELU_SW(pair[0], testvec["ins"][0], testvec["ins"][1], testvec["ins"][3], testvec["ins"][4]) == pair[1]
        i = i + 1


def test_CLIPPED_RELU():
    """Test Component class RELU"""
    test_vecs = [
        {"ins":[4,4, True, 0,  0, {'ceil':3}], "outs":[[7,3],[8,0],[2,2]]}, # 111 -> 011, 1000 -> 0 (same width, qs=0)
        {"ins":[5,4, False, 0, 0, {'ceil':4}], "outs":[[15,4],[17,0]]}, # 01111 -> 1111, 10001 -> 0000 (shorter output, qs=0)
        {"ins":[4,2, False, 0, 0, {'ceil':4}], "outs":[[3,3],[5,0],[9,0]]}, # 0011 -> 11, 0101 --> 0, 1001 --> 0   (even shorter output, qs=0)
        {"ins":[3,5, False, 0, 0, {'ceil':4}], "outs":[[3,3],[4,0]]}, # 011 -> 00011, 100 --> 00000 (shorter input, qs= 0)
        {"ins":[4,4, True, 2,  2, {'ceil':1}], "outs":[[7,4],[8,0]]}, # 111 -> 100, 1000 -> 0 (same width, same qs)
        {"ins":[5,4, False, 2, 2, {'ceil':1}], "outs":[[15,4],[17,0]]}, # 01111 -> 1111, 10001 -> 0000 (shorter output, same qs)
        {"ins":[4,2, False, 2, 1, {'ceil':1}], "outs":[[3,1],[5,2],[9,0]]}, # 0011 -> 01, 0101 --> 10, 1001 --> 0   (shorter output, shorter qs)
        {"ins":[3,5, False, 1, 0, {'ceil':1}], "outs":[[3,1],[4,0]]}, # 011 -> 00001, 100 --> 00000 (longer output, shorter qs)
        {"ins":[3,6, False, 1, 3, {'ceil':2}], "outs":[[3,12],[4,0]]}, # 011 -> 01100, 100 --> 00000 (longer output, longer qs)
        {"ins":[3,5, False, 1, 1, {'ceil':1}], "outs":[[3,2],[4,0]]}, # 011 -> 00011, 100 --> 00000 (longer output, same qs)
    ]
    i = 0
    for testvec in test_vecs:
        print("VEC " + str(testvec))
        testinst = activation_functions.CLIPPED_RELU(testvec["ins"][0],
                                       testvec["ins"][1],
                                       testvec["ins"][2],
                                       testvec["ins"][3],
                                       testvec["ins"][4],
                                       testvec["ins"][5])
        testinst.elaborate()
        testinst.apply(DefaultPassGroup()) 
        for pair in testvec["outs"]:
            testinst.sim_reset()
            testinst.activation_function_in @= pair[0]
            testinst.sim_tick()
            assert testinst.activation_function_out == pair[1]
            assert activation_functions.CLIPPED_RELU_SW(pair[0], testvec["ins"][0], testvec["ins"][1], testvec["ins"][3], testvec["ins"][4], testvec["ins"][5]) == pair[1]
        i = i + 1

def test_LEAKY_RELU():
    """Test Component class RELU""" # 10001 - 0001 - 1111
    test_vecs = [ # -1 - 0001 - 1110 - 1111   1100
        {"ins":[4,4, True, 0, 0], "outs":[[7,7],[8,15]]}, # 111 -> 111, 1000 -> 1111 (same width, qs=0)
        {"ins":[5,4, False, 0, 0], "outs":[[15,15],[17,14]]}, # 01111 -> 1111, 10001 -> 1110 (shorter output, qs=0)
        {"ins":[4,2, False, 0, 0], "outs":[[3,3],[9,3]]}, # 0011 -> 11, 0101 --> 0, 1001 --> 11 (even shorter output, qs=0)
        {"ins":[3,5, False, 0, 0], "outs":[[3,3],[4,31]]}, # 011 -> 00011, 100 --> 11111 (shorter input, qs= 0)
        {"ins":[4,4, False, 2, 2], "outs":[[7,7],[8,15]]}, # 111 -> 111, 1000 -> 1111 (same width, same qs)
        {"ins":[5,4, False, 2, 2], "outs":[[15,15],[17,14]]}, # 01111 -> 1111, 10001 -> 1110 (shorter output, same qs)
        {"ins":[4,2, False, 2, 1], "outs":[[3,1],[5,2],[9,3]]}, # 0011 -> 01, 0101 --> 10, 1001 --> 11   (shorter output, shorter qs)
        {"ins":[3,5, False, 1, 0], "outs":[[3,1],[4,31]]}, # 011 -> 00001, 100 --> 11111 (longer output, shorter qs)
        {"ins":[3,6, False, 1, 3], "outs":[[3,12]]}, # 011 -> 01100, 100 --> 11111 (longer output, longer qs)
        {"ins":[3,5, False, 1, 1], "outs":[[3,3],[4,31]]}, # 011 -> 00011, 100 --> 11111 (longer output, same qs)
    ]
    i = 0
    for testvec in test_vecs:
        print("VEC " + str(testvec))
        testinst = activation_functions.LEAKY_RELU(testvec["ins"][0],
                                       testvec["ins"][1],
                                       testvec["ins"][2],
                                       testvec["ins"][3],
                                       testvec["ins"][4])
        testinst.elaborate()
        testinst.apply(DefaultPassGroup()) 
        for pair in testvec["outs"]:
            print("**")
            testinst.sim_reset()
            testinst.activation_function_in @= pair[0]
            testinst.sim_tick()
            testinst.sim_tick()
            assert testinst.activation_function_out == pair[1]
            assert activation_functions.LEAKY_RELU_SW(pair[0], testvec["ins"][0], testvec["ins"][1], testvec["ins"][3], testvec["ins"][4]) == pair[1]
        i = i + 1

def test_NONE():
    """Test Component class RELU"""
    test_vecs = [
        {"ins":[4,4, True, 0, 0], "outs":[[7,7],[8,8]]}, # 111 -> 111, 1000 -> 1000 (same width, qs=0)
        {"ins":[5,4, False, 0, 0], "outs":[[15,15],[17,1]]}, # 01111 -> 1111, 10001 -> 0000 (shorter output, qs=0)
        {"ins":[3,5, False, 0, 0], "outs":[[3,3],[4,28]]}, # 011 -> 00011, 100 --> 11100 (longer output, qs= 0)
        {"ins":[4,4, True, 2, 2], "outs":[[7,7],[8,8]]}, # 111 -> 111, 10.00(8,-2,8) -> 1000 (same width, same qs)
        {"ins":[5,4, False, 2, 2], "outs":[[15,15],[17,1]]}, # 01111 -> 1111, 10001 -> 0001 (shorter output, same qs)
        {"ins":[4,2, False, 2, 1], "outs":[[3,1],[5,2]]}, # 0011 -> 01, 0101 --> 10, 10.01(9,-1.75) --> 0   (shorter output, shorter qs)
        {"ins":[3,5, False, 1, 0], "outs":[[3,1],[4,30]]}, # 011 -> 00001, 100 --> 11110 (longer output, shorter qs)
        {"ins":[3,5, True, 1, 0], "outs":[[3,1],[4,30]]}, # 011 -> 00001, 100 --> 11110 (longer output, shorter qs)
        {"ins":[3,6, False, 1, 3], "outs":[[3,12],[4,48]]}, # 011 -> 01100, 100 --> 110000 (longer output, longer qs)
        {"ins":[3,5, False, 1, 1], "outs":[[3,3],[4,28]]}, # 011 -> 00011, 100 --> 11100 (longer output, same qs)
    ] ## 0110 = -0111 = -1.75 = 011.1 = -3.5 (011.1 - 100.0 - 100.1)
      ## 3...
      ## 1001 = 
    i = 0
    for testvec in test_vecs:
        print("VEC " + str(testvec))
        testinst = activation_functions.NONE(testvec["ins"][0],
                                       testvec["ins"][1],
                                       testvec["ins"][2],
                                       testvec["ins"][3],
                                       testvec["ins"][4])
        testinst.elaborate()
        testinst.apply(DefaultPassGroup()) 
        for pair in testvec["outs"]:
            print(" >> PAIR"  + str(pair))
            testinst.sim_reset()
            testinst.activation_function_in @= pair[0]
            testinst.sim_tick()
            assert testinst.activation_function_out == pair[1]
            assert activation_functions.NONE_SW(pair[0], testvec["ins"][0], testvec["ins"][1], testvec["ins"][3], testvec["ins"][4]) == pair[1]
            
        i = i + 1


def test_SIGMOID_LUT():
    """Test Component class RELU"""
    test_vecs = [
        {"ins":[3,3, True, 0, 0], "outs":[[7,0],[2,0]]}, # 111. -> 0, .1000 -> 0
        {"ins":[3,4, True, 0, 4], "outs":[[2,14],[6,1]]}, # 010 -> .1100 , 110 (-2,6) ->  (.0001)    
        {"ins":[3,4, True, 1, 4], "outs":[[1,9],[5,2]]}, # 001 (0.5) -> 0.622 (1001), -1.5 (101)-> 0.182(.0010) 
    ]
    
    i = 0
    for testvec in test_vecs:
        print("VEC " + str(testvec))
        testinst = activation_functions.SIGMOID_LUT(testvec["ins"][0],
                                       testvec["ins"][1],
                                       testvec["ins"][2],
                                       testvec["ins"][3],
                                       testvec["ins"][4])
        testinst.elaborate()
        testinst.apply(DefaultPassGroup()) 
        for pair in testvec["outs"]:
            testinst.sim_reset()
            testinst.activation_function_in @= pair[0]
            testinst.sim_tick()
            assert testinst.activation_function_out == pair[1]
            assert activation_functions.SIGMOID_LUT_SW(pair[0], testvec["ins"][0], testvec["ins"][1], testvec["ins"][3], testvec["ins"][4]) == testinst.activation_function_out
        i = i + 1

def test_ELU_LUT():
    """Test Component class RELU"""
    test_vecs = [
        {"ins":[3,3, True, 0, 0, {"alpha": 1}], "outs":[[7,7],[2,2]]}, # 111. (-1) -> -0.632 ~0, 10.00 -> 10.00
        {"ins":[3,4, True, 0, 1, {"alpha": 3}], "outs":[[2,4],[6,10]]}, # 010 -> .1100 , 110 (-2,6) -> -2.5 (10.11)    
        {"ins":[3,4, True, 1, 3, {"alpha": 1}], "outs":[[1,4],[5,9]]}, # 001 (0.5) -> 0.622 (1001), -1.5 (10.1)-> -0.77 ~ -0.111 = -1.001 = 1000 
    ]
    
    i = 0
    for testvec in test_vecs:
        print("VEC " + str(testvec))
        testinst = activation_functions.ELU_LUT(testvec["ins"][0],
                                       testvec["ins"][1],
                                       testvec["ins"][2],
                                       testvec["ins"][3],
                                       testvec["ins"][4],
                                       testvec["ins"][5])
        testinst.elaborate()
        testinst.apply(DefaultPassGroup()) 
        for pair in testvec["outs"]:
            testinst.sim_reset()
            testinst.activation_function_in @= pair[0]
            testinst.sim_tick()
            assert testinst.activation_function_out == pair[1]
            assert activation_functions.ELU_LUT_SW(pair[0], testvec["ins"][0], testvec["ins"][1], testvec["ins"][3], testvec["ins"][4], testvec["ins"][5]) == testinst.activation_function_out
        i = i + 1
        
def test_SELU_LUT():
    """Test Component class RELU"""
    test_vecs = [
        {"ins":[3,3, True, 0, 0, {"alpha": 1, "scale":1}], "outs":[[7,7],[2,2]]}, # 111. (-1) -> -0.632 ~-2, 10.00 -> 10.00
        {"ins":[3,4, True, 0, 1, {"alpha": 3, "scale":1}], "outs":[[2,4],[6,10]]}, # 010 -> .1100 , 110 (-2,6) -> -2.5 (10.11)    
        {"ins":[3,4, True, 1, 3, {"alpha": 1, "scale":1}], "outs":[[1,4],[5,9]]}, # 001 (0.5) -> 0.622 (1001), -1.5 (10.1)-> -0.77 ~ -0.111 = -1.001 = 1000 
        {"ins":[3,3, True, 0, 0, {"alpha": 1, "scale":2}], "outs":[[7,6],[2,4]]}, # 111. (-1) -> -0.632 ~1, 10.00 -> 10.00
        {"ins":[3,4, True, 0, 1, {"alpha": 3, "scale":0.25}], "outs":[[2,1],[6,14]]}, # 010 -> .1100 , 110 (-2,6) -> -2.5/4 = -0.62 (111.1)    
        {"ins":[3,4, True, 1, 3, {"alpha": 1, "scale":0.5}], "outs":[[1,2],[5,12]]}, # 001 (0.5) -> 0.622 (1001), -1.5 (10.1)-> -0.39 ~ -0.011 = 1.101 
    ]
    
    i = 0
    for testvec in test_vecs:
        print("VEC " + str(testvec))
        testinst = activation_functions.SELU_LUT(testvec["ins"][0],
                                       testvec["ins"][1],
                                       testvec["ins"][2],
                                       testvec["ins"][3],
                                       testvec["ins"][4],
                                       testvec["ins"][5])
        testinst.elaborate()
        testinst.apply(DefaultPassGroup()) 
        for pair in testvec["outs"]:
            testinst.sim_reset()
            testinst.activation_function_in @= pair[0]
            testinst.sim_tick()
            assert testinst.activation_function_out == pair[1]
            assert activation_functions.SELU_LUT_SW(pair[0], testvec["ins"][0], testvec["ins"][1], testvec["ins"][3], testvec["ins"][4], testvec["ins"][5]) == testinst.activation_function_out
        i = i + 1

        
def test_GENERIC_LUT():
    """Test Component class RELU"""
    
    test_vecs = [
        {"ins":[3,3, True, 0, 0, {'lut':[2,2.5,3,3.5,4,4.5,5,5.5]}], "outs":[[7,5],[2,3]]}, # 111. -> 0, .1000 -> 0
        {"ins":[3,4, True, 0, 4, {'lut':[0,0.5,0.125,0.25,0.75,0.875,0.0625,0]}], "outs":[[2,2],[7,0]]}, # 010 -> .1100 , 110 (-2,6) ->  (.0001)    
        {"ins":[3,4, True, 1, 4, {'lut':[0,0.5,0.125,0.25,0.75,0.875,0.0625,0]}], "outs":[[1,8],[5,14]]}, # 001 (0.5) -> 0.622 (1001), -1.5 (101)-> 0.182(.0010) 
    ]
    
    i = 0
    for testvec in test_vecs:
        print("VEC " + str(testvec))
        testinst = activation_functions.GENERIC_LUT(testvec["ins"][0],
                                       testvec["ins"][1],
                                       testvec["ins"][2],
                                       testvec["ins"][3],
                                       testvec["ins"][4],
                                       testvec["ins"][5])
        testinst.elaborate()
        testinst.apply(DefaultPassGroup()) 
        for pair in testvec["outs"]:
            testinst.sim_reset()
            testinst.activation_function_in @= pair[0]
            testinst.sim_tick()
            assert testinst.activation_function_out == pair[1]
            assert activation_functions.GENERIC_LUT_SW(pair[0], testvec["ins"][0], testvec["ins"][1], testvec["ins"][3], testvec["ins"][4], testvec["ins"][5]) == pair[1]
        i = i + 1

def test_TANH_LUT():
    """Test Component class RELU"""
    test_vecs = [
        {"ins":[3,3, True, 0, 0], "outs":[[7,7],[2,0]]}, # 111. (-1) -> (-0.76 - .1100) 0, .1000 -> 0
        {"ins":[3,4, True, 0, 4], "outs":[[2,15],[1,12]]}, # 010 -> (0.96) .1111 , 110 (-2,6) ->  (.0001)    
        {"ins":[3,4, True, 1, 4], "outs":[[1,7],[3,14]]}, # 001 (0.5) -> 0.622 (1001), -1.5 (101)-> 0.182(.0010) 
    ]
    
    
    i = 0
    for testvec in test_vecs:
        print("VEC " + str(testvec))
        testinst = activation_functions.TANH_LUT(testvec["ins"][0],
                                       testvec["ins"][1],
                                       testvec["ins"][2],
                                       testvec["ins"][3],
                                       testvec["ins"][4])
        testinst.elaborate()
        testinst.apply(DefaultPassGroup()) 
        for pair in testvec["outs"]:
            print(pair)
            testinst.sim_reset()
            testinst.activation_function_in @= pair[0]
            testinst.sim_tick()
            assert testinst.activation_function_out == pair[1]
            assert activation_functions.TANH_LUT_SW(pair[0], testvec["ins"][0], testvec["ins"][1], testvec["ins"][3], testvec["ins"][4]) == pair[1]
        i = i + 1
