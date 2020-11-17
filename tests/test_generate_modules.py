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

filesets = [#("mlb_spec_0.yaml","input_spec_0.yaml", "weight_spec_0.yaml",
            # "emif_spec_0.yaml", "projection_spec_0.yaml"),
            ("mlb_spec_0.yaml","input_spec_0.yaml", "weight_spec_0.yaml",
             "emif_spec_1.yaml", "projection_spec_0.yaml"),
            ("mlb_spec_0.yaml","input_spec_1.yaml", "weight_spec_0.yaml",
             "emif_spec_1.yaml", "projection_spec_0.yaml"),
            ("mlb_spec_0.yaml","input_spec_0.yaml", "weight_spec_1.yaml",
             "emif_spec_1.yaml", "projection_spec_0.yaml"),
            ("mlb_spec_1.yaml","input_spec_0.yaml", "weight_spec_0.yaml",
             "emif_spec_0.yaml", "projection_spec_1.yaml")
           ]

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
    "mlb_file,ab_file,wb_file,emif_file,proj_file", filesets
)
@pytest.mark.full_simulations
def test_simulate_emif_statemachine(
        mlb_file, ab_file, wb_file, emif_file, proj_file):
    
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
    oaddr = len(emif_data)
    
    print("\n\nOriginally")
    print(emif_data)
    
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
                                                    oaddr=oaddr)

    # Check that EMIFs have the right data
    #wvalues_per_emif_word = math.ceil( \
    #    utils.get_sum_datatype_width(wb_yaml, "DATA", ["in"]) / \
    #    proj_yaml["stream_info"]["W"])
    emif_vals = utils.read_out_stored_values_from_emif(
        testinst.emif_inst.sim_model.buf, wvalues_per_buf, iaddr,
        proj_yaml["stream_info"]["W"], 0)
    for k in range(len(wbuf)):
        for j in range(len(wbuf[k])):
            for i in range(len(wbuf[k][j])):
                assert emif_vals[k*len(wbuf[k])+j][i] == wbuf[k][j][i]
    
    #ivalues_per_emif_word = math.ceil( \
    #    utils.get_sum_datatype_width(ab_yaml, "DATA", ["in"]) / \
    #    proj_yaml["stream_info"]["I"])
    emif_vals = utils.read_out_stored_values_from_emif(
        testinst.emif_inst.sim_model.buf, ivalues_per_buf, oaddr-iaddr,
        proj_yaml["stream_info"]["I"], iaddr)
    print("\n\nCOMPARE")
    print(emif_vals)
    print("WITH")
    print(ibuf)
    print("WITH")
    print(ibuf_flat)
    print("WITH")
    print(emif_data)
    for k in range(len(ibuf)):
        for j in range(len(ibuf[k])):
            for i in range(len(ibuf[k][j])):
                assert emif_vals[k*len(ibuf[k])+j][i] == ibuf[k][j][i]
    check_buffers(testinst.datapath, testinst.datapath.weight_modules,
                  "ml_block_weights_inst_{}",
                  wbuf, proj_yaml["stream_info"]["W"], testinst)
    check_buffers(testinst.datapath, testinst.datapath.input_act_modules,
                  "ml_block_inputs_inst_{}",
                  ibuf, proj_yaml["stream_info"]["I"], testinst)
    
    # Now load the weights into the MLBsejr
    inner_ub = proj_yaml["inner_projection"]["UB"]["value"]
    outer_ub = proj_yaml["outer_projection"]["UB"]["value"]
    wbi_section_length = proj_yaml["inner_projection"]["UE"]["value"] * \
                        proj_yaml["inner_projection"]["URN"]["value"] * \
                        proj_yaml["inner_projection"]["URW"]["value"]
    wbo_section_length = proj_yaml["outer_projection"]["UE"]["value"] * \
                        proj_yaml["outer_projection"]["URN"]["value"] * \
                        proj_yaml["outer_projection"]["URW"]["value"] *\
                        proj_yaml["inner_projection"]["UG"]["value"] *  wbi_section_length
    # Calculate required buffers etc.
    mlb_count = utils.get_mlb_count(proj_yaml["outer_projection"])
    mac_count = utils.get_mlb_count(proj_yaml["inner_projection"])
    assert(check_mlb_chains_values(testinst.datapath, mlb_count, mac_count, 1, 1,
                            "ml_block_inst_{}", "weight_out_{}", wbuf,
                            proj_yaml["stream_info"]["W"],
                            wbo_section_length, outer_ub,
                            wbi_section_length, inner_ub))

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
    for bufi in range(obuf_count):
        for olen in range(min(obuf_len,ibuf_len)-1): 
            assert obuf[bufi][olen] == outvals_yaml[bufi*min(obuf_len,ibuf_len) + olen]

def test_simulate_emif_statemachine_unit():
    test_simulate_emif_statemachine("mlb_spec_0.yaml",
                               "input_spec_0.yaml",
                               "weight_spec_0.yaml",
                               "emif_spec_0.yaml",
                               "projection_spec_0.yaml")
    
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

    
