import copy
import pytest
import os
import yaml
import sys
import random
import jsonschema
import subprocess
from jsonschema import validate
from verilog_ml_benchmark_generator import utils
from verilog_ml_benchmark_generator import generate_modules

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_helpers import *

VTR_FLOW_PATH = os.getenv('VTR_FLOW_PATH')
NUMX = 1976
NUMI = 988
HEIGHT = 177
WIDTH = 149

filesets = [# EMIF, input, MLB interface same width
            # Preload weights from wide buffer
            # Inner: URW3, URN2, UE1, UB2, UG2
            # Outer: URW1, URN2, UE1, UB2, UG2
              ("mlb_spec_0.yaml","input_spec_0.yaml", "weight_spec_0.yaml",
               "emif_spec_0.yaml", "projection_spec_0.yaml", True),
            # EMIF wider than input, weight buffer
              ("mlb_spec_0.yaml","input_spec_0.yaml", "weight_spec_0.yaml",
               "emif_spec_1.yaml", "projection_spec_0.yaml", True),
            # Input 2x as wide as MLB interface
              ("mlb_spec_0.yaml","input_spec_1.yaml", "weight_spec_0.yaml",
               "emif_spec_1.yaml", "projection_spec_0.yaml", True),
            # Narrower weight buffer
              ("mlb_spec_0.yaml","input_spec_0.yaml", "weight_spec_1.yaml",
              "emif_spec_1.yaml", "projection_spec_0.yaml", True),
            # Inner: URW6, URN1, UE2, UB1, UG1
            # Outer: URW1, URN1, UE1, UB2, UG1
               ("mlb_spec_1.yaml","input_spec_0.yaml", "weight_spec_0.yaml",
                "emif_spec_0.yaml", "projection_spec_1.yaml", True),
            #
            # Narrower I, W, 
            # Inner: URW1, URN2, UE2, UB2, UG1
            # Outer: URW1, URN1, UE3, UB1,UG1
             ("mlb_spec_3.yaml","input_spec_1.yaml", "weight_spec_3.yaml",
                "emif_spec_1.yaml", "projection_spec_3.yaml", True),
            # Narrower I, W, 
            # Outer: URW1, URN2, UE2, UB2, UG1
            # Inner: URW1, URN1, UE3, UB1,UG1
               ("mlb_spec_3.yaml","input_spec_1.yaml", "weight_spec_3.yaml",
                 "emif_spec_1.yaml", "projection_spec_4.yaml", True),
            # Inner: URW6, URN1, UE1, UB2, UG1
            # Outer: URW2, URN2, UE2, UB1, UG1
               ("mlb_spec_3.yaml","input_spec_1.yaml", "weight_spec_3.yaml",
                 "emif_spec_1.yaml", "projection_spec_5.yaml", True),
            # Inner: All2
            # Outer: All2
                ("mlb_spec_3.yaml","input_spec_1.yaml", "weight_spec_3.yaml",
                  "emif_spec_1.yaml", "projection_spec_5.yaml", True),
           ]  # bad: URW2, URN2


def test_fix_ranges():
    """Test width 0 range fixing"""
    input_test = ["some_wire[0:0] = another_wire[9:9] + third_wire[9:1]"]
    outputs = generate_modules.remove_width_0_ranges(input_test)
    assert(outputs == ["some_wire = another_wire[9] + third_wire[9:1]"])
    


def test_yaml_schemas():
    """Test yaml schema validation"""
    hwb_yaml_legal = {
        "block_name": "ml_block",
        "MAC_info": { "num_units": 12, "data_widths": {"W":8, "I":8, "O": 32}},
        "ports": [
            {"name":"a_in", "width":32, "direction": "in", "type":"W"},
            {"name":"b_out", "width":32, "direction": "out", "type":"I"},
            {"name":"res_out", "width":128, "direction": "out", "type":"O"},
        ]
    }
    validate(instance=hwb_yaml_legal, schema=generate_modules.mlb_spec_schema)

    hwb_yaml_illegal = hwb_yaml_legal
    hwb_yaml_illegal.pop("MAC_info")
    validate(instance=hwb_yaml_legal, schema=generate_modules.buffer_spec_schema)

    # Test illegal cases
    hwb_yaml_illegal = hwb_yaml_legal
    hwb_yaml_illegal.pop("ports")
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate(instance=hwb_yaml_illegal, schema=generate_modules.buffer_spec_schema)
    
    proj_legal = {"name": "test",
                  "activation_function": "RELU",
                  "data_widths": {"W": 8,
                                  "I": 8,
                                  "O": 32},
                  "outer_projection": {'RY': 2, 'RX': 1, 'C': 1,
                                       'B': 2, 'E': 4, 'PX':1, 'PY':1,
                                       'G': 1},
                  "inner_projection": {'C': 2, 'RX': 2, 'RY': 1,
                                       'B':2,'E':1, 'PX':1, 'PY':1,
                                       'G':1}
                  }
    validate(instance=proj_legal, schema=generate_modules.proj_schema)
    proj_legal.pop("name")
    validate(instance=proj_legal, schema=generate_modules.proj_schema)

    # Test illegal cases
    proj_illegal = proj_legal
    proj_illegal.pop("activation_function")
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate(instance=proj_illegal, schema=generate_modules.proj_schema)
    proj_illegal = proj_legal
    proj_illegal.pop("data_widths")
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate(instance=proj_illegal, schema=generate_modules.proj_schema)
    proj_illegal = proj_legal
    proj_illegal.pop("outer_projection")
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate(instance=proj_illegal, schema=generate_modules.proj_schema)


@pytest.mark.requiresodin
@pytest.mark.skip
def test_odin_emif_statemachine(mlb_file, ab_file, wb_file, emif_file,
                                projection_file):
    print(VTR_FLOW_PATH)
    assert VTR_FLOW_PATH, "Set environment variable VTR_FLOW_PATH to location " + \
        " of VTR flow scripts"
        
    
    # Make sure that output gets through odin.
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            mlb_file)
    ab_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ab_file)
    wb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            wb_file)
    emif_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            emif_file)
    proj_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            projection_file)
    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "test_odin_emif_sm_odin.v")
    archfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "test_arch.xml")
    with open(mlb_spec) as stream:
        mlb_yaml = yaml.safe_load(stream)
    with open(ab_spec) as stream:
        ab_yaml = yaml.safe_load(stream)
    with open(wb_spec) as stream:
        wb_yaml = yaml.safe_load(stream)
    with open(proj_spec) as stream:
        proj_yaml = yaml.safe_load(stream)
    with open(emif_spec) as stream:
        emif_yaml = yaml.safe_load(stream)
    outtxt = generate_modules.generate_accelerator_given_mapping(module_name="test_odin_emif_sm", 
                                                    mlb_spec=mlb_yaml, wb_spec=wb_yaml,
                                                    ab_spec=ab_yaml, projection=proj_yaml,
                                                    write_to_file=True,
                                                    fast_gen=True,
                                                    emif_spec=emif_yaml,
                                                    waddr=0, iaddr=20, oaddr=90)

    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "test_odin_emif_sm_odin.v")
    command = [VTR_FLOW_PATH, outfile, archfile,
               "-ending_stage", "abc"]
    print("ODIN command ::" + str(command))
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    assert "OK" in str(process.stdout.read())
    
@pytest.mark.requiresodin
def test_odin_emif_statemachine_0():
    test_odin_emif_statemachine("mlb_spec.yaml",
                                "b1_spec.yaml",
                                "b0_spec.yaml",
                                "emif_spec.yaml",
                                "projection_spec.yaml")
    
@pytest.mark.requiresodin
def test_odin_emif_statemachine_mini():
    test_odin_emif_statemachine("mlb_spec.yaml",
                                "b1_spec.yaml",
                                "b0_spec.yaml",
                                "emif_spec.yaml",
                                "projection_spec_3.yaml")
    
@pytest.mark.skip
def test_generate_layer(workload_yaml,
                        mlb_file, ab_file, wb_file, emif_file, proj_file, ws, v=True, sim=True, num_mlbs=1, preload_o=1, preload_i=1, layer_name="test_full_layer_flow", run_odin=True):

    # Make sure that output gets through odin.
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            mlb_file)
    ab_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ab_file)
    wb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            wb_file)
    emif_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             emif_file)
    proj_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             proj_file)
    
    with open(mlb_spec) as stream:
        mlb_yaml = yaml.safe_load(stream)
    with open(ab_spec) as stream:
        ab_yaml = yaml.safe_load(stream)
    with open(wb_spec) as stream:
        wb_yaml = yaml.safe_load(stream)
    with open(proj_spec) as stream:
        proj_yaml = yaml.safe_load(stream)
    with open(emif_spec) as stream:
        emif_yaml = yaml.safe_load(stream)
  
    # Calculate buffer dimensions info
    wvalues_per_buf, wbuf_len, wbuf_count = utils.get_iw_buffer_dimensions(
        wb_yaml, proj_yaml, 'W')
    ivalues_per_buf, ibuf_len, ibuf_count = utils.get_iw_buffer_dimensions(
        ab_yaml, proj_yaml, 'I')
    ovalues_per_buf, obuf_len, obuf_count = utils.get_obuffer_dimensions(
        ab_yaml, proj_yaml)

    # Create random input data arrays to load into EMIF
    wbuf = [[[random.randint(0,(2**proj_yaml["data_widths"]["W"])-1)
            for k in range(wvalues_per_buf)]    # values per word
            for i in range(wbuf_len)]           # words per buffer
            for j in range(wbuf_count)]         # buffer count
    wbuf_flat = [sum((lambda i: inner[i] * \
                      (2**(i*proj_yaml["data_widths"]["W"])))(i) \
                     for i in range(len(inner))) \
                         for outer in wbuf for inner in outer]
    iaddr = len(wbuf_flat)
    ibuf = [[[random.randint(0,(2**proj_yaml["data_widths"]["I"])-1)
             for k in range(ivalues_per_buf)]            # values per word
             for i in range(ibuf_len)]                   # words per buffer
             for j in range (ibuf_count)]                # buffers
    ibuf_flat = [sum((lambda i: inner[i] * \
                (2**(i*proj_yaml["data_widths"]["W"])))(i) \
                     for i in range(len(inner))) \
                          for outer in ibuf for inner in outer]
    emif_data = wbuf_flat + ibuf_flat
    oaddr = len(emif_data)
    
    emif_yaml["fill"] = emif_data
    generate_modules.generate_accelerator_for_layers(
        module_name=layer_name, 
        mlb_spec=mlb_yaml,
        wb_spec=wb_yaml,
        ab_spec=ab_yaml,
        emif_spec=emif_yaml,
        pe_count=num_mlbs,
        layer=workload_yaml,
        waddr=0,
        iaddr=iaddr,
        oaddr=oaddr,
        simulate=sim,
        preload_o=preload_o,
        preload_i=preload_i)

    verilog_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            layer_name + ".v")
    archfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "test_arch_intel.xml")
    command = [VTR_FLOW_PATH, verilog_file, archfile,
               "-ending_stage", "abc"]
    print("ODIN command:" + str(command))
    if (run_odin):
        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        assert "OK" in str(process.stdout.read())

@pytest.mark.longtest 
@pytest.mark.skip
def test_generate_layer_example_intel_l1(layer_name="test_full_layer_flow"):
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":32},
        "loop_dimensions": {'B':1, 'C':1024, 
                            'E':1000, 'PX':1,
                            'PY':1, 'RX':1,
                            'RY':1},
        "activation_function": 'RELU'
       }
    #assert 5==7
    test_generate_layer(workload, "mlb_spec_intel.yaml",
                        "input_spec_intel_16.yaml",
                        "input_spec_intel_16.yaml",
                        "emif_spec_intel.yaml",
                        "projection_spec_cs.yaml", True,
                        False, False, NUMI, -1, 2, layer_name=layer_name, run_odin=False)
    x_locs = list(range(8,WIDTH-2,7))+[81,95,67]
    print(x_locs)
    print(sorted(x_locs))
    gen_constraint_file("chain_list_for_placement.yaml", "full_layer_l1.constraints", sorted(x_locs), list(range(2,HEIGHT-2-4,4)))
    
@pytest.mark.longtest
@pytest.mark.skip
def test_generate_layer_example_intel_l2(layer_name="test_full_layer_flow"):
    
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":32},
        "loop_dimensions": {'B':1, 'C':64,
                            'E':128, 'PX':56,
                            'PY':56, 'RX':1,
                            'RY':1},
        "activation_function": 'RELU'
       }
    test_generate_layer(workload, "mlb_spec_intel.yaml",
                        "input_spec_intel_8.yaml",
                        "input_spec_intel_8.yaml",
                        "emif_spec_intel.yaml",
                        "projection_spec_cs.yaml", True,
                        False, False, NUMI, -1, 2, layer_name=layer_name, run_odin=False)
    gen_constraint_file("chain_list_for_placement.yaml", "full_layer_l2.constraints", list(range(8,WIDTH-2,7))+[81,95,67], list(range(2,HEIGHT-2-4,4)))

@pytest.mark.longtest
@pytest.mark.skip
def test_generate_layer_example_intel_l3(layer_name="test_full_layer_flow"):
    
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":32},
        "loop_dimensions": {'B':1,'C':3,
                            'E':32,'PX':224,
                            'PY':224,'RX':3,
                            'RY':3},
        "activation_function": 'RELU'
       }
    test_generate_layer(workload, "mlb_spec_intel.yaml",
                        "input_spec_intel_8.yaml",
                        "input_spec_intel_8.yaml",
                        "emif_spec_intel.yaml",
                        "projection_spec_cs.yaml", True,
                        False, False, NUMI, -1, 2, layer_name=layer_name, run_odin=False)
    
    gen_constraint_file("chain_list_for_placement.yaml", "full_layer_l3.constraints", list(range(8,WIDTH-2,7))+[81,95,67], list(range(2,HEIGHT-2-4,4)))


def test_generate_layer_intel():
    test_generate_layer_example_intel_l1(layer_name="test_full_layer_flow_l1")
    test_generate_layer_example_intel_l2(layer_name="test_full_layer_flow_l2")
    test_generate_layer_example_intel_l3(layer_name="test_full_layer_flow_l3")

@pytest.mark.longtest
@pytest.mark.skip
def test_generate_layer_xilinx_l1(layer_name="test_full_layer_flow_x1"):
    
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':1024, 
                            'E':1000, 'PX':1,
                            'PY':1, 'RX':1,
                            'RY':1},
        "activation_function": 'RELU'
       }
    test_generate_layer(workload, "mlb_spec_xilinx_mode2.yaml",
                        "input_spec_intel_16.yaml",
                        "input_spec_intel_36.yaml",
                        "emif_spec_intel.yaml",
                        "projection_spec_cs.yaml", True,
                        False, False, NUMX, -1, -1, layer_name=layer_name, run_odin=False)
    gen_constraint_file("chain_list_for_placement.yaml", "full_layer_x1.constraints", list(range(8,WIDTH-2,7))+[81,95,67], list(range(2,HEIGHT-2-2,2)), portname="P_cout")
    #assert 4==9

@pytest.mark.longtest
@pytest.mark.skip
def test_generate_layer_xilinx_l2(layer_name="test_full_layer_flow_x2"):
    
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':64,
                            'E':128, 'PX':56,
                            'PY':56, 'RX':1,
                            'RY':1},
        "activation_function": 'RELU'
       }
    test_generate_layer(workload, "mlb_spec_xilinx_mode2.yaml",
                        "input_spec_intel_8.yaml",
                        "input_spec_intel_8.yaml",
                        "emif_spec_intel.yaml",
                        "projection_spec_cs.yaml", True,
                        False, False, NUMX, -1, -1, layer_name=layer_name, run_odin=False)
    gen_constraint_file("chain_list_for_placement.yaml", "full_layer_x2.constraints", list(range(8,WIDTH-2,7))+[81,95,67], list(range(2,HEIGHT-2-2,2)), portname="P_cout")
    
@pytest.mark.longtest
@pytest.mark.skip
def test_generate_layer_xilinx_l3(layer_name="test_full_layer_flow_x3"):
    
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1,'C':3,
                            'E':32,'PX':224,
                            'PY':224,'RX':3,
                            'RY':3},
        "activation_function": 'RELU'
       }
    test_generate_layer(workload, "mlb_spec_xilinx_mode2.yaml",
                        "input_spec_intel_8.yaml",
                        "input_spec_intel_8.yaml",
                        "emif_spec_intel.yaml",
                        "projection_spec_cs.yaml", True,
                        False, False, NUMX, -1, -1, layer_name=layer_name, run_odin=False)
    gen_constraint_file("chain_list_for_placement.yaml", "full_layer_x3.constraints", list(range(8,WIDTH-2,7))+[81,95,67], list(range(2,HEIGHT-2-2,2)), portname="P_cout")


def test_generate_layer_xilinx():
    test_generate_layer_xilinx_l1(layer_name="test_full_layer_flow_x1")
    test_generate_layer_xilinx_l2(layer_name="test_full_layer_flow_x2")
    test_generate_layer_xilinx_l3(layer_name="test_full_layer_flow_x3")


@pytest.mark.longtest
@pytest.mark.skip
def test_generate_layer_intel_soft(layer_name="test_full_layer_flow_soft"):
    
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":32},
        "loop_dimensions": {'B':1,'C':3,
                            'E':32,'PX':224,
                            'PY':224,'RX':3,
                            'RY':3},
        "activation_function": 'RELU'
       }
    test_generate_layer(workload, "mlb_spec_intel_v2.yaml",
                        "input_spec_intel_8.yaml",
                        "input_spec_intel_8.yaml",
                        "emif_spec_intel.yaml",
                        "projection_spec_cs.yaml", True,
                        False, False, NUMI, -1, 2, layer_name=layer_name, run_odin=False)
    
    gen_constraint_file("chain_list_for_placement.yaml", "full_layer_soft.constraints", list(range(8,WIDTH-2,7))+[81,95,67], list(range(2,HEIGHT-2-4,4)))


@pytest.mark.longtest
@pytest.mark.skip
def test_generate_layer_intel_soft_small(layer_name="test_full_layer_flow_soft_small"):
    
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":32},
        "loop_dimensions": {'B':1,'C':3,
                            'E':32,'PX':224,
                            'PY':224,'RX':3,
                            'RY':3},
        "activation_function": 'RELU'
       }
    test_generate_layer(workload, "mlb_spec_intel_v2.yaml",
                        "input_spec_intel_8.yaml",
                        "input_spec_intel_8.yaml",
                        "emif_spec_intel.yaml",
                        "projection_spec_cs.yaml", True,
                        False, False,494, -1, 2, layer_name=layer_name, run_odin=False)
    
    gen_constraint_file("chain_list_for_placement.yaml", "full_layer_soft_small.constraints", list(range(8,WIDTH-2,7))+[81,95,67], list(range(2,HEIGHT-2-4,4)))
   
@pytest.mark.longtest
def test_generate_layer_xilinx_test(layer_name="test_full_layer_flow_x3"):
    
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1,'C':3,
                            'E':32,'PX':224,
                            'PY':224,'RX':3,
                            'RY':3},
        "activation_function": 'RELU'
       }
    test_generate_layer(workload, "mlb_spec_xilinx_mode2.yaml",
                        "input_spec_intel.yaml",
                        "input_spec_intel.yaml",
                        "emif_spec_intel.yaml",
                        "projection_spec_cs.yaml", True,
                        False, False, 50, -1, -1, layer_name=layer_name, run_odin=False)
    gen_constraint_file("chain_list_for_placement.yaml", "full_layer_x3.constraints", list(range(8,100,7)), list(range(2,100,2)), portname="P_cout")

    
@pytest.mark.longtest
def test_generate_layer_example_intel_test(layer_name="test_full_layer_flow"):
    
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":32},
        "loop_dimensions": {'B':1,'C':3,
                            'E':32,'PX':224,
                            'PY':224,'RX':3,
                            'RY':3},
        "activation_function": 'RELU'
       }
    test_generate_layer(workload, "mlb_spec_intel.yaml",
                        "input_spec_intel_8.yaml",
                        "input_spec_intel_8.yaml",
                        "emif_spec_intel.yaml",
                        "projection_spec_cs.yaml", True,
                        False, False, 25, 1, 1, layer_name=layer_name, run_odin=False)
    
    gen_constraint_file("chain_list_for_placement.yaml", "full_layer_l3.constraints", list(range(8,45,7)), list(range(2,45,4)))
    
