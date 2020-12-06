import copy
import pytest
import os
import yaml
import sys
import random
from verilog_ml_benchmark_generator import utils
from verilog_ml_benchmark_generator import generate_modules
import jsonschema
import subprocess
from jsonschema import validate

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_helpers import *

VTR_FLOW_PATH = os.getenv('VTR_FLOW_PATH')

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

def test_yaml_schemas():
    """Test yaml schema validation"""
    hwb_yaml_legal = {
        "block_name": "ml_block",
        "simulation_model": "MLB",
        "MAC_info": { "num_units": 12, "data_widths": {"W":8, "I":8, "O": 32}},
        "ports": [
            {"name":"a_in", "width":32, "direction": "in", "type":"W"},
            {"name":"b_out", "width":32, "direction": "out", "type":"I"},
            {"name":"res_out", "width":128, "direction": "out", "type":"O"},
        ]
    }
    validate(instance=hwb_yaml_legal, schema=generate_modules.mlb_spec_schema)

    # Test a few more ok cases.
    hwb_yaml_legal.pop("block_name")
    validate(instance=hwb_yaml_legal, schema=generate_modules.buffer_spec_schema)
    hwb_yaml_legal.pop("simulation_model")
    validate(instance=hwb_yaml_legal, schema=generate_modules.buffer_spec_schema)
    hwb_yaml_illegal = hwb_yaml_legal
    hwb_yaml_illegal.pop("MAC_info")
    validate(instance=hwb_yaml_legal, schema=generate_modules.buffer_spec_schema)

    # Test illegal cases
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate(instance=hwb_yaml_illegal, schema=generate_modules.mlb_spec_schema)
    hwb_yaml_illegal = hwb_yaml_legal
    hwb_yaml_illegal.pop("ports")
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate(instance=hwb_yaml_illegal, schema=generate_modules.buffer_spec_schema)
    
    proj_legal = {"name": "test",
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
    validate(instance=proj_legal, schema=generate_modules.proj_schema)
    proj_legal.pop("name")
    validate(instance=proj_legal, schema=generate_modules.proj_schema)

    # Test illegal cases
    proj_illegal = proj_legal
    proj_illegal.pop("activation_function")
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate(instance=proj_illegal, schema=generate_modules.proj_schema)
    proj_illegal = proj_legal
    proj_illegal.pop("stream_info")
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate(instance=proj_illegal, schema=generate_modules.proj_schema)
    proj_illegal = proj_legal
    proj_illegal.pop("outer_projection")
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate(instance=proj_illegal, schema=generate_modules.proj_schema)


@pytest.mark.skip
def test_odinify_statemachine():
    assert VTR_FLOW_PATH, "Set environment variable VTR_FLOW_PATH to " + \
        "location of VTR flow scripts"
        
    
    # Make sure that output gets through odin.
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mlb_spec.yaml")
    ab_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "b1_spec.yaml")
    wb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "b0_spec.yaml")
    proj_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "projection_spec.yaml")
    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "test_odin_sm.v")
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
    outtxt = generate_modules.generate_statemachine("test_odin_sm", 
                                            mlb_yaml, wb_yaml,
                                            ab_yaml, proj_yaml, False)
    with open(outfile, 'w') as file:
        file.write(outtxt[1])
    command = [VTR_FLOW_PATH, outfile, archfile,
               "-ending_stage", "abc"]
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    assert "OK" in str(process.stdout.read())



@pytest.mark.parametrize(
    "mlb_file,ab_file,wb_file,emif_file,proj_file,ws", filesets
)
@pytest.mark.full_simulations
@pytest.mark.skip
def test_simulate_emif_statemachine(
        mlb_file, ab_file, wb_file, emif_file, proj_file, ws, v=True):
    
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
    wbuf = [[[random.randint(0,(2**proj_yaml["stream_info"]["W"])-1)
            for k in range(wvalues_per_buf)]    # values per word
            for i in range(wbuf_len)]           # words per buffer
            for j in range(wbuf_count)]         # buffer count
    wbuf_flat = [sum((lambda i: inner[i] * \
                      (2**(i*proj_yaml["stream_info"]["W"])))(i) \
                     for i in range(len(inner))) \
                         for outer in wbuf for inner in outer]
    iaddr = len(wbuf_flat)
    ibuf = [[[random.randint(0,(2**proj_yaml["stream_info"]["I"])-1)
             for k in range(ivalues_per_buf)]            # values per word
             for i in range(ibuf_len)]                   # words per buffer
             for j in range (ibuf_count)]                # buffers
    ibuf_flat = [sum((lambda i: inner[i] * \
                (2**(i*proj_yaml["stream_info"]["I"])))(i) \
                     for i in range(len(inner))) \
                          for outer in ibuf for inner in outer]
    emif_data = wbuf_flat + ibuf_flat
    print(wbuf_flat)
    oaddr = len(emif_data)
    
    emif_yaml["parameters"]["fill"] = copy.deepcopy(emif_data)
    outvals, testinst = generate_modules.simulate_statemachine(
        module_name="test_odin_emif_sm", 
                                                    mlb_spec=mlb_yaml,
                                                    wb_spec=wb_yaml,
                                                    ab_spec=ab_yaml,
                                                    emif_spec=emif_yaml,
                                                    projection=proj_yaml,
                                                    write_to_file=True,
                                                    randomize=False,
                                                    waddr=0,
                                                    iaddr=iaddr,
                                                    oaddr=oaddr,
                                                    ws=ws,
                                                    validate_output=v)
    print("done simulating")
    # Check that EMIFs have the right data
    emif_vals = utils.read_out_stored_values_from_emif(
        testinst.emif_inst.sim_model.buf, wvalues_per_buf, iaddr,
        proj_yaml["stream_info"]["W"], 0)
    print(emif_vals)
    print(wbuf)
    for k in range(len(wbuf)):
        for j in range(len(wbuf[k])):
            for i in range(len(wbuf[k][j])):
                assert emif_vals[k*len(wbuf[k])+j][i] == wbuf[k][j][i]
                
    emif_vals = utils.read_out_stored_values_from_emif(
        testinst.emif_inst.sim_model.buf, ivalues_per_buf, oaddr-iaddr,
        proj_yaml["stream_info"]["I"], iaddr)
    print("\n\nCOMPARE")
    print(emif_vals)
    print("WITH")
    print(ibuf)
    for k in range(len(ibuf)):
        for j in range(len(ibuf[k])):
            for i in range(len(ibuf[k][j])):
                assert emif_vals[k*len(ibuf[k])+j][i] == ibuf[k][j][i]

    # Check that the right data got into the on-chip buffers
    check_buffers(testinst.datapath, testinst.datapath.weight_modules,
                  "ml_block_weights_inst_{}",
                  wbuf, proj_yaml["stream_info"]["W"], testinst)
    check_buffers(testinst.datapath, testinst.datapath.input_act_modules,
                  "ml_block_inputs_inst_{}",
                  ibuf, proj_yaml["stream_info"]["I"], testinst)
    # Check that the right data is in the MLBs
    #if (ws):
    print("okkkk...")
    print(testinst.datapath.mlb_modules.ml_block_inst_0.sim_model.mac_modules.input_out_0)
    print(testinst.datapath.mlb_modules.ml_block_inst_0.sim_model.mac_modules.sum_out_0)
    #if (ws):
    #    assert(check_weight_contents(
    #        testinst.datapath, proj_yaml,
    #        "ml_block_inst_{}", "weight_out_{}", wbuf))

    print("\n\n\n\nHERE")
    obuf = [[[0 for i in range(ovalues_per_buf)]
             for i in range(obuf_len)]
             for j in range (obuf_count)]
    obuf = utils.get_expected_outputs(obuf, ovalues_per_buf,
                                wbuf,
                                ibuf, ivalues_per_buf,
                                proj_yaml)
    print("EXPECTED OUT")
    print(obuf)
    print("\nACTUAL OUT")

    with open("final_offchip_data_contents.yaml") as outfile:
        outvals_yaml = yaml.safe_load(outfile)
    print(outvals_yaml)
    print(obuf_count)
    print(obuf_len)
    for bufi in range(obuf_count):
        for olen in range(min(obuf_len,ibuf_len)-1): 
            assert obuf[bufi][olen] == outvals_yaml[bufi*min(obuf_len,ibuf_len) + olen]

def test_simulate_emif_statemachine_unit_ws_pl():
    test_simulate_emif_statemachine("mlb_spec_3.yaml",
                               "input_spec_1.yaml",
                               "weight_spec_3.yaml",
                               "emif_spec_1.yaml",
                                    "projection_spec_5.yaml", True, False)
    
def test_simulate_emif_statemachine_unit_ws_bc():
    test_simulate_emif_statemachine("mlb_spec_3.yaml",
                               "input_spec_1.yaml",
                               "weight_spec_3.yaml",
                               "emif_spec_1.yaml",
                                    "projection_spec_6.yaml", True, False)
   # assert 1==0
    
def test_simulate_emif_statemachine_unit_os_bc():
    test_simulate_emif_statemachine("mlb_spec_3.yaml",
                               "input_spec_1.yaml",
                               "weight_spec_3.yaml",
                               "emif_spec_1.yaml",
                               "projection_spec_7.yaml", False, False)
   # assert 1==0
    
def test_simulate_random_emif_statemachine():

    # Make sure that output gets through odin.
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mlb_spec_0.yaml")
    ab_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "input_spec_0.yaml")
    wb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "weight_spec_0.yaml")
    emif_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "emif_spec_0.yaml")
    proj_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "projection_spec_0.yaml")
    
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
    wbuf = [[[random.randint(0,(2**proj_yaml["stream_info"]["W"])-1)
            for k in range(wvalues_per_buf)]    # values per word
            for i in range(wbuf_len)]           # words per buffer
            for j in range(wbuf_count)]         # buffer count
    wbuf_flat = [sum((lambda i: inner[i] * \
                      (2**(i*proj_yaml["stream_info"]["W"])))(i) \
                     for i in range(len(inner))) \
                         for outer in wbuf for inner in outer]
    iaddr = len(wbuf_flat)
    ibuf = [[[random.randint(0,(2**proj_yaml["stream_info"]["I"])-1)
             for k in range(ivalues_per_buf)]            # values per word
             for i in range(ibuf_len)]                   # words per buffer
             for j in range (ibuf_count)]                # buffers
    ibuf_flat = [sum((lambda i: inner[i] * \
                (2**(i*proj_yaml["stream_info"]["W"])))(i) \
                     for i in range(len(inner))) \
                          for outer in ibuf for inner in outer]
    emif_data = wbuf_flat + ibuf_flat
    oaddr = len(emif_data)
    
    emif_yaml["parameters"]["fill"] = emif_data
    outvals, testinst = generate_modules.simulate_statemachine(module_name="test_odin_emif_sm", 
                                                    mlb_spec=mlb_yaml,
                                                    wb_spec=wb_yaml,
                                                    ab_spec=ab_yaml,
                                                    emif_spec=emif_yaml,
                                                    projection=proj_yaml,
                                                    write_to_file=False,
                                                    randomize=True,
                                                    waddr=0,
                                                    iaddr=iaddr,
                                                    oaddr=oaddr)

            
def test_odinify_emif_statemachine():
    assert VTR_FLOW_PATH, "Set environment variable VTR_FLOW_PATH to location " + \
        " of VTR flow scripts"
        
    
    # Make sure that output gets through odin.
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mlb_spec.yaml")
    ab_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "b1_spec.yaml")
    wb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "b0_spec.yaml")
    emif_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "emif_spec.yaml")
    proj_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "projection_spec.yaml")
    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "test_odin_emif_sm.v")
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
    outtxt = generate_modules.generate_statemachine(module_name="test_odin_emif_sm", 
                                                    mlb_spec=mlb_yaml, wb_spec=wb_yaml,
                                                    ab_spec=ab_yaml, projection=proj_yaml,
                                                    write_to_file=False,
                                                    emif_spec=emif_yaml,
                                                    waddr=0, iaddr=20, oaddr=90)
    with open(outfile, 'w') as file:
        file.write(outtxt[1])
    command = [VTR_FLOW_PATH, outfile, archfile,
               "-ending_stage", "abc"]
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    assert "OK" in str(process.stdout.read())

def test_odinify():
    assert VTR_FLOW_PATH, "Set environment variable VTR_FLOW_PATH to location " + \
        "of VTR flow scripts"
        
    # Make sure that output gets through odin.
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mlb_spec.yaml")
    ab_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "b1_spec.yaml")
    wb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "b0_spec.yaml")
    proj_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "projection_spec.yaml")
    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "test_odin.v")
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
    outtxt = generate_modules.generate_full_datapath("test_odin", 
                                            mlb_yaml, wb_yaml,
                                                     ab_yaml, proj_yaml, True)
    with open(outfile, 'w') as file:
        file.write(outtxt[1])
    command = [VTR_FLOW_PATH, outfile, archfile,
               "-ending_stage", "abc"]
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    assert "OK" in str(process.stdout.read())

    
