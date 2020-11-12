
import pytest
import os
import yaml
from verilog_ml_benchmark_generator import generate_modules
import jsonschema
import subprocess
from jsonschema import validate
VTR_FLOW_PATH = "/home/esther/VTR/vpr_with_edits2/vpr_with_edits/vtr_flow/scripts/run_vtr_flow.pl"

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

def test_odinify():
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
                                            ab_yaml, proj_yaml, False)
    with open(outfile, 'w') as file:
        file.write(outtxt[1])
    command = [VTR_FLOW_PATH, outfile, archfile,
               "-ending_stage", "abc"]
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    assert "OK" in str(process.stdout.read())

    
