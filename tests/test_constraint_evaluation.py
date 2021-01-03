import copy
import pytest
import os
import yaml
import sys
import random
from verilog_ml_benchmark_generator import utils
from verilog_ml_benchmark_generator import constraint_evaluation
from verilog_ml_benchmark_generator import generate_modules
import jsonschema
import subprocess
from jsonschema import validate


def test_find_mappings():
    """Test yaml schema validation"""
    hwb = {
        "block_name": "ml_block",
        "simulation_model": "MLB",
        "MAC_info": { "num_units": 30, "data_widths": {"W":4, "I":4, "O": 8}},
        "possible_projections": {"URW":1, "URN":10, "UE": 3, "UB": 1, "UG": 1},
        "ports": [
            {"name":"a_in", "width":32, "direction": "in", "type":"W"},
            {"name":"b_out", "width":32, "direction": "out", "type":"I"},
            {"name":"res_out", "width":128, "direction": "out", "type":"O"},
        ]
    }
    workload_conv0 = {'B':1,'C':3,
                     'E':32,'PX':224,
                     'PY':224,'RX':3,
                     'RY':3}
    workload_conv1 = {'B':1,'C':64,
                      'E':128,'PX':56,
                      'PY':56,'RX':1,
                      'RY':1}
    workload_fc1 = {'B':1000,'C':1024,
                      'E':1,'PX':1,
                      'PY':1,'RX':1,
                      'RY':1}
    suggested_soln = {'BO':1,'CO':1,
                'EO':11,'PXO':14,
                'PYO':3,'RXO':3,
                'RYO':1,'BI':1,'CI':3,
                'EI': 3,'PXI':1,
                'PYI':1,'RXI':1,
                'RYI':3,'BT':1,'CT':1,
                'ET': 1,'PXT':16,
                'PYT':75,'RXT':1,
                'RYT':1}
    mappings, tp = constraint_evaluation.find_mappings(hwb, workload_conv0, 2000, False, preload_o=32, suggested_solution=suggested_soln)
    print(len(mappings))
    #assert tp == 14340
    #mappings, tp = constraint_evaluation.find_mappings(hwb, workload_conv1, 200, True, suggested_solution=None)
    #print(len(mappings))
    #assert tp == 15512
    #mappings, tp = constraint_evaluation.find_mappings(hwb, workload_fc1, 2000, False, suggested_solution=None)
    #print(len(mappings))
    assert tp == 4090
