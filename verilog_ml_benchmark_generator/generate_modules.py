"""
    This file contains the user-facing functions that perform main tasks...
       - Generate pyMTL statemachine modules
       - Simulate using cycle-accurate pyMTL simulations
       - Generate verilog and perform post processing
    Also some helper functions for post processing verilog to make it
    odin compatible.
"""
import re
import random
import yaml
import constraint_evaluation
from jsonschema import validate
from pymtl3 import DefaultPassGroup
from pymtl3.passes.backends.verilog import VerilogTranslationPass
import utils
import state_machine_classes
il = 1
currstep = 0

# Schemas to validate input yamls
supported_activations = ["RELU"]
datatype_mlb = ["I", "O", "W", "C", "CLK", "RESET", "I_EN", "W_EN", "ACC_EN",
                "WEN"]
datatype_emif = ["AVALON_ADDRESS", "AVALON_READ", "AVALON_WRITE",
                 "AVALON_READDATA", "AVALON_WRITEDATA", "AVALON_READDATAVALID",
                 "AVALON_WAITREQUEST", "MODE"]
datatype_buffer = ["ADDRESS", "DATA"]
datatype_any = datatype_mlb+datatype_buffer+datatype_emif
port_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "width": {"type": "number", "minimum": 1},
        "direction": {"type": "string", "enum": ["in", "out"]},
        "type": {"type": "string", "enum": datatype_any}},
    "required": ["name", "width", "direction", "type"],
    "additionalProperties": False
}
access_pattern_schema = {
    "type": "object",
    "properties": {
        "AP1": {"value": {"type": "int", "minimum": 1}},
        "AP2": {"value": {"type": "int", "minimum": 1}},
        "AP3": {"value": {"type": "int", "minimum": 1}},
        "AP4": {"value": {"type": "int", "minimum": 1}},
        "AP5": {"value": {"type": "int", "minimum": 1}},
        "PRELOAD": {"type": "array",
                    "items": {"type": "object",
                              "properties":
                              {"dtype": {"type": "string",
                                         "enum": ["W", "I", "O"]},
                               "bus_count": {"type": "number",
                                             "minimum": 1}},
                              "required": ["dtype"]}},
    },
    "required": ["AP2", "AP3", "AP4", "AP5"],
    "additionalProperties": False
}
inner_proj_schema = {
    "type": "object",
    "properties": {
        "RX": {"type": "number", "minimum": 1},
        "RY": {"type": "number", "minimum": 1},
        "C": {"type": "number", "minimum": 1},
        "E": {"type": "number", "minimum": 1},
        "B": {"type": "number", "minimum": 1},
        "PX": {"type": "number", "minimum": 1},
        "PY": {"type": "number", "minimum": 1},
        "G": {"type": "number", "minimum": 1},
        "PRELOAD": {"type": "array",
                    "items": {"type": "object",
                              "properties":
                              {"dtype": {"type": "string",
                                         "enum": ["W", "I", "O"]},
                               "bus_count": {"type": "number",
                                             "minimum": 1}},
                              "required": ["dtype"]}},
    },
    "required": ["RY", "B", "E", "G"],
    "additionalProperties": False
}
datawidth_schema = {
    "type": "object",
    "properties": {
        "W": {"type": "number"},
        "I": {"type": "number"},
        "O": {"type": "number"}},
    "required": ["W", "I", "O"],
    "additionalProperties": False
}
proj_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "activation_function": {"type": "string", "enum": ["RELU"]},
        "outer_projection": inner_proj_schema,
        "inner_projection": inner_proj_schema,
        "temporal_projection": inner_proj_schema,
        "data_widths": datawidth_schema,
        "dilation": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"}
            },
            "required": ["x", "y"],
            "additionalProperties": False
        },
        "stride": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"}
            },
            "required": ["x", "y"],
            "additionalProperties": False
        }
    },
    "required": ["activation_function", "outer_projection",
                 "inner_projection"],
    "additionalProperties": False
}
MAC_info_schema = {"type": "object",
                   "properties": {
                       "num_units": {"type": "number"},
                       "data_widths": datawidth_schema},
                   "additionalProperties": False}
buffer_spec_schema = {
    "type": "object",
    "properties": {
        "simulation_model": {"type": "string"},
        "block_name": {"type": "string"},
        "ports": {"type": "array", "items": port_schema}},
    "required": ["ports", "block_name"],
    "additionalProperties": False
}
emif_schema = {
    "type": "object",
    "properties": {
        "block_name": {"type": "string"},
        "simulation_model": {"type": "string"},
        "pipelined": {"type": "string", "enum": ["True", "False"]},
        "ports": {"type": "array", "items": port_schema}},
    "required": ["ports", "block_name"]
}
mlb_spec_schema = {
    "type": "object",
    "properties": {
        "block_name": {"type": "string"},
        "MAC_info": MAC_info_schema,
        "simulation_model": {"type": "string"},
        "ports": {"type": "array", "items": port_schema},
        "access_patterns": access_pattern_schema,
        "output_accumulator": {"type": "boolean"},
    },
    "required": ["ports", "block_name"],
    "additionalProperties": False
}


def validate_inputs(wb_spec=None, ab_spec=None, mlb_spec=None,
                    emif_spec=None, projections=None, sim=True):
    """ Call pyMTL elaboration methods to elaborate a Component instance,
        and post-process generated verilog.

    :param wb_spec: Weight buffer input info.
    :param ab_spec: Activation buffer input info.
    :param mlb_spec: MLB input info.
    :param projection: Projection input.
    """
    if (wb_spec):
        validate(instance=wb_spec, schema=buffer_spec_schema)
    if (ab_spec):
        validate(instance=ab_spec, schema=buffer_spec_schema)
    if (mlb_spec):
        validate(instance=mlb_spec, schema=mlb_spec_schema)
        assert(("simulation_model" not in mlb_spec) or
               (mlb_spec["simulation_model"] == "MLB")), \
            "MLB yaml input has invalid simulation model:" + \
            " it should be either 'MLB' or nothing."
        mlb_spec["simulation_model"] = "MLB"

        # Ensure that the MLB specified is sufficient for all
        # specified projections.
        if not ("MAC_info" in mlb_spec):
            mlb_spec["MAC_info"] = {}
        i_data_widths = {
            dtype: max([proj_spec['data_widths'][dtype]
                        for proj_spec in projections])
            for dtype in ['W', 'I', 'O']}
        if not ("data_widths" in mlb_spec["MAC_info"]):
            mlb_spec["MAC_info"]["data_widths"] = i_data_widths
        else:
            for dtype in ['W', 'I', 'O']:
                assert(i_data_widths[dtype] <=
                       mlb_spec["MAC_info"]["data_widths"][dtype]), \
                       "MLB precision is insufficient (" + \
                       str(i_data_widths[dtype]) + " > " + \
                       str(mlb_spec["MAC_info"]["data_widths"][dtype]) \
                       + ")"

        if not ("num_units" in mlb_spec["MAC_info"]):
            inner_projs = [proj_spec['inner_projection']
                           for proj_spec in projections]
            MAC_counts = [utils.get_mlb_count(inner_proj)
                          for inner_proj in inner_projs]
            mlb_spec["MAC_info"]["num_units"] = max(MAC_counts)

    if (projections):
        for projection in projections:
            validate(instance=projection, schema=proj_schema)

            # Does the projection make sense?
            total_px = projection["inner_projection"].get('PX', 1) * \
                projection["outer_projection"].get('PX', 1) * \
                projection["temporal_projection"].get('PX', 1)
            total_rx = projection["inner_projection"].get('RX', 1) * \
                projection["outer_projection"].get('RX', 1) * \
                projection["temporal_projection"].get('RX', 1)
            assert (total_px >= total_rx), \
                "Image size (" + str(total_px) + \
                " should be greater than filter size (" + \
                str(total_rx)

            # This is for test coverage purposes
            if (sim):
                if (projection["inner_projection"].get('RX', 1) > 1):
                    assert(True)
                if (projection["inner_projection"].get('RX', 1) == 1):
                    assert(True)
                if (projection["inner_projection"].get('RY', 1) > 1):
                    assert(True)
                if (projection["inner_projection"].get('RY', 1) == 1):
                    assert(True)
                if (projection["inner_projection"].get('C', 1) > 1):
                    assert(True)
                if (projection["inner_projection"].get('C', 1) == 1):
                    assert(True)
                if (projection["inner_projection"].get('E', 1) > 1):
                    assert(True)
                if (projection["inner_projection"].get('E', 1) == 1):
                    assert(True)
                if (projection["inner_projection"].get('PX', 1) > 1):
                    assert(True)   # unused
                if (projection["inner_projection"].get('PX', 1) == 1):
                    assert(True)
                if (projection["inner_projection"].get('PY', 1) > 1):
                    assert(True)
                if (projection["inner_projection"].get('PY', 1) == 1):
                    assert(True)
                if (projection["inner_projection"].get('B', 1) > 1):
                    assert(True)
                if (projection["inner_projection"].get('B', 1) == 1):
                    assert(True)
                if (projection["inner_projection"].get('G', 1) > 1):
                    assert(True)
                if (projection["inner_projection"].get('G', 1) == 1):
                    assert(True)
                if (projection["outer_projection"].get('RX', 1) > 1):
                    assert(True)
                if (projection["outer_projection"].get('RX', 1) == 1):
                    assert(True)
                if (projection["outer_projection"].get('RY', 1) > 1):
                    assert(True)
                if (projection["outer_projection"].get('RY', 1) == 1):
                    assert(True)
                if (projection["outer_projection"].get('C', 1) > 1):
                    assert(True)
                if (projection["outer_projection"].get('C', 1) == 1):
                    assert(True)
                if (projection["outer_projection"].get('E', 1) > 1):
                    assert(True)
                if (projection["outer_projection"].get('E', 1) == 1):
                    assert(True)
                if (projection["outer_projection"].get('PX', 1) > 1):
                    assert(True)
                if (projection["outer_projection"].get('PX', 1) == 1):
                    assert(True)
                if (projection["outer_projection"].get('PY', 1) > 1):
                    assert(True)
                if (projection["outer_projection"].get('PY', 1) == 1):
                    assert(True)
                if (projection["outer_projection"].get('B', 1) > 1):
                    assert(True)
                if (projection["outer_projection"].get('B', 1) == 1):
                    assert(True)
                if (projection["outer_projection"].get('G', 1) > 1):
                    assert(True)
                if (projection["outer_projection"].get('G', 1) == 1):
                    assert(True)
                if (projection["temporal_projection"].get('RX', 1) > 1):
                    assert(True)
                if (projection["temporal_projection"].get('RX', 1) == 1):
                    assert(True)
                if (projection["temporal_projection"].get('RY', 1) > 1):
                    assert(True)
                if (projection["temporal_projection"].get('RY', 1) == 1):
                    assert(True)
                if (projection["temporal_projection"].get('C', 1) > 1):
                    assert(True)
                if (projection["temporal_projection"].get('C', 1) == 1):
                    assert(True)
                if (projection["temporal_projection"].get('E', 1) > 1):
                    assert(True)
                if (projection["temporal_projection"].get('E', 1) == 1):
                    assert(True)
                if (projection["temporal_projection"].get('PX', 1) > 1):
                    assert(True)
                if (projection["temporal_projection"].get('PX', 1) == 1):
                    assert(True)
                if (projection["temporal_projection"].get('PY', 1) > 1):
                    assert(True)
                if (projection["temporal_projection"].get('PY', 1) == 1):
                    assert(True)
                if (projection["temporal_projection"].get('B', 1) > 1):
                    assert(True)
                if (projection["temporal_projection"].get('B', 1) == 1):
                    assert(True)
                if (projection["temporal_projection"].get('G', 1) > 1):
                    assert(True)
                if (projection["temporal_projection"].get('G', 1) == 1):
                    assert(True)
                if (projection.get("stride", {}).get('x', 1) == 1):
                    assert(True)
                if (projection.get("stride", {}).get('x', 1) > 1):
                    assert(True)
                if (projection.get("stride", {}).get('y', 1) == 1):
                    assert(True)
                if (projection.get("stride", {}).get('y', 1) > 1):
                    assert(True)
                if (projection.get("dilation", {}).get('x', 1) == 1):
                    assert(True)
                if (projection.get("dilation", {}).get('x', 1) > 1):
                    assert(True)
                if (projection.get("dilation", {}).get('y', 1) == 1):
                    assert(True)
                if (projection.get("dilation", {}).get('y', 1) > 1):
                    assert(True)

    if (emif_spec):
        validate(instance=emif_spec, schema=emif_schema)
        if "pipelined" not in emif_spec:
            emif_spec["pipelined"] = "True"
        if "fill" not in emif_spec:
            emif_spec["fill"] = []


def generate_verilog(component, write_to_file, module_name,
                     include_sim_models):
    """ Call pyMTL elaboration methods to elaborate a Component instance,
        and post-process generated verilog.

    :param component: PyMTL Component to elaborate
    :param write_to_file: Whether or not to write resulting verilog to file.
    :param component: Name of generated verilog file
    """
    utils.printi(1, "{:=^60}".format("> Generating verilog from pymtl " +
                                     "models: " + module_name + "_pymtl.v" +
                                     " (not synthesizable) <"))
    # Create the pymtl modules and generate verilog using pymtl
    component.set_metadata(VerilogTranslationPass.enable, True)
    component.set_metadata(VerilogTranslationPass.explicit_file_name,
                           module_name + "_pymtl.v")
    component.set_metadata(VerilogTranslationPass.explicit_module_name,
                           module_name)
    component.apply(VerilogTranslationPass())

    # Post-process generated file so that it is compatible with odin :(
    outtxt_o = postprocess_verilog_odin(module_name + "_pymtl.v")
    if (write_to_file):
        with open(module_name + "_odin.v", 'w') as file:
            file.write(outtxt_o)

    # Post-process generated file into system verilog
    outtxt_sv = postprocess_verilog_sv(module_name + "_pymtl.v",
                                       include_sim_models)
    if (write_to_file):
        with open(module_name + "_quartus_vivado.sv", 'w') as file:
            file.write(outtxt_sv)

    utils.printi(1, "{: ^10}".format("> Final output files: "))
    utils.printi(2, "{: ^10}".format(module_name + "_odin.sv"))
    utils.printi(2, "{: ^10}".format(module_name + "_quartus_vivado.sv"))
    return [component, outtxt_o]


def remove_sim_block_defs(line_list, sim_blocks):
    """ Remove definitions of listed sim_blocks from line_list
        (necessary because sim models shoudn't be included in generated .v)

    :param line_list: Original file, split into lines
    :param sim_blocks: List of modules to remove
    """
    lineidx = 0
    skipCopy = False
    while lineidx < len(line_list):
        # Remove definitions of hard blocks
        for sim_block in sim_blocks:
            found_hardmod = re.search(r"module .*" + sim_block,
                                      line_list[lineidx])
            found_hardmod2 = re.search(r"Full name.*" + sim_block,
                                       line_list[lineidx])
            if found_hardmod or found_hardmod2:
                skipCopy = True
        if skipCopy:
            if (re.search(r"endmodule", line_list[lineidx])):
                skipCopy = False
            line_list[lineidx] = "//" + line_list[lineidx]
        lineidx += 1
    return line_list


def remove_non_existant_ports(line_list, non_existant_ports):
    """ Remove references to list of ports that shouldn't exist
        (necessary because reset and clock are added to all modules, even
        when they aren't required)

    :param line_list: Original file, split into lines
    :param non_existant_ports: List of orts to remove
    """
    lineidx = 0
    while lineidx < len(line_list):
        # Remove nonexistant ports
        for nep in non_existant_ports:
            found_nep = re.search(r"\(.*" + nep + r"\s*\)",
                                  line_list[lineidx])
            if found_nep:
                # If there was no comma at the end of the line,
                # remove comma from prev line.
                curr_comma = re.search(r"\)\s*,", line_list[lineidx])
                if not curr_comma:
                    prev_line = re.search("(.*),", line_list[lineidx - 1])
                    line_list[lineidx - 1] = prev_line.group(1)
                line_list[lineidx] = "//" + line_list[lineidx]
        lineidx += 1

    return line_list


def move_ios_into_module_body(line_list):
    """ Moves input and output declarations out of the module definition
        and into the module body.
        (necessary because ODIN requires for inputs and outputs to be
        declared in this way)

    :param line_list: Original file, split into lines
    """
    lineidx = 0
    stored_ios = []
    while lineidx < len(line_list):
        # Move inputs out of the module definition
        origline = line_list[lineidx]
        foundio = re.findall(r"(in|out)put\s+wire\s+\[\d+:\d+\]\s+\S+\s*,*",
                             origline)
        foundmodstart = re.search(";", origline)
        assert len(foundio) < 2
        if len(foundio) > 0:
            match = re.search(
                r"(in|out)(put)(\s+)(wire)(\s+)(\[\d+:\d+\])" +
                r"(\s+)(\S+)(\s+)(,*)", origline)
            assert match
            line_list[lineidx] = match.group(8) + match.group(10)
            stored_ios.append(match.group(1) + match.group(2) +
                              match.group(3) + match.group(6) +
                              match.group(7) + match.group(8) +
                              match.group(9) + ';')
            lineidx += 1
        elif len(stored_ios) > 0 and foundmodstart:
            p1 = origline[0:foundmodstart.end()]
            line_list[lineidx] = p1
            for storedio in stored_ios:
                lineidx += 1
                line_list.insert(lineidx, storedio)
            lineidx += 1
            line_list.insert(lineidx, origline[foundmodstart.end():])
            stored_ios = []
        else:
            lineidx += 1

    return line_list


def remove_parameter_references(line_list):
    """ Replace all references to parameters with their values.
        (necessary because ODIN doesn't support parameters)

    :param line_list: Original file, split into lines
    """
    lineidx = 0
    localparams = {}
    while lineidx < len(line_list):
        # Keep track of local params
        foundlp = re.search(
            r"localparam\s+wire\s+\[\d+:\d+\]\s+(\S+)\s*=\s*(\S+);",
            line_list[lineidx])
        if foundlp:
            line_list[lineidx] = "//" + line_list[lineidx]
            localparams[foundlp.group(1)] = foundlp.group(2)
        else:
            for localparam in localparams:
                line_list[lineidx] = re.sub(r"\d\'\(\s*" + localparam +
                                            r"\s*\)",
                                            localparams[localparam],
                                            line_list[lineidx])
        lineidx += 1

    return line_list


def remove_width_0_ranges(line_list):
    """ Replace all references to parameters with their values.
        (necessary because ODIN doesn't like seeing [x:x] for example)

    :param line_list: Original file, split into lines
    """
    lineidx = 0
    while lineidx < len(line_list):
        line_list[lineidx] = re.sub(r"\[0:0\]", "", line_list[lineidx])
        foundlp = re.search(r"\S\[(\d):(\d)\]", line_list[lineidx])
        if foundlp:
            if (foundlp.group(1) == foundlp.group(2)):
                line_list[lineidx] = re.sub(r"\[" + str(foundlp.group(1)) +
                                            ":" +
                                            str(foundlp.group(1)) + r"\]",
                                            "[" + str(foundlp.group(1)) + "]",
                                            line_list[lineidx])
        lineidx += 1

    return line_list


def postprocess_verilog_sv(filename_in, include_sim_models=False):
    """ Make verilog file compatible with odin.
        This is a bit of a hack, but necessary to run VTR.

    :param filename_in: Name of file to fix up
    """
    utils.printi(1, "{:=^60}".format("> Post-processing generated verilog " +
                                     "to .sv <"))

    # Post-process generated file so that it is compatible with odin :(
    with open(filename_in, 'r') as file:
        filedata = file.read()

    # Replace some system verilog syntax etc
    filedata = filedata.replace('1\'d1', '2\'d1')
    filedata = filedata.replace('{}', 'empty')

    # pyMTL adds clk and reset to everything... but we dont want it
    # in some cases.
    non_existant_ports = []

    # Rename ML blocks to correct name
    line_list = filedata.splitlines()
    line_list = move_ios_into_module_body(line_list)
    if (not include_sim_models):
        line_list = remove_sim_block_defs(line_list, ["sim_True"])
    line_list = remove_non_existant_ports(line_list, non_existant_ports)
    line_list = remove_parameter_references(line_list)
    line_list = remove_width_0_ranges(line_list)
    filedata = '\n'.join(line_list)

    # replace HW block component names with actual names
    if (not include_sim_models):
        filedata = re.sub(r"(MLB_Wrapper__spec_)(\S*)(__projs_\S*)(\s+)(.*)",
                          r"\2\4\5",
                          filedata)
        filedata = re.sub(r"(HWB_Sim__spec_)(\S*)(__projs_\S*)(\s+)(.*)",
                          r"\2\4\5",
                          filedata)
        filedata = re.sub(r"(HWB_Sim__)(\S*)(\s+)(\S+)_inst(.*)",
                          r"\4\3\4_inst\5",
                          filedata)
    filedata = re.sub(r'\s+\[0:0\]\s+', r" ", filedata)
    return filedata


def postprocess_verilog_odin(filename_in):
    """ Make verilog file compatible with odin.
        This is a bit of a hack, but necessary to run VTR.

    :param filename_in: Name of file to fix up
    """
    utils.printi(1, "{:=^60}".format("> Post-processing generated " +
                                     "verilog for ODIN <"))

    # Post-process generated file so that it is compatible with odin :(
    with open(filename_in, 'r') as file:
        filedata = file.read()

    # Replace some system verilog syntax
    filedata = filedata.replace('logic', 'wire')  # so far this works...
    filedata = filedata.replace('always_comb', 'always@(*)')
    filedata = filedata.replace('always_ff', 'always')
    filedata = filedata.replace('1\'d1', '2\'d1')
    filedata = filedata.replace('{}', 'empty')

    # pyMTL adds clk and reset to everything... but we dont want it
    # in some cases.
    non_existant_ports = [r"ml_block_weight\S*_inst_\d+__clk",
                          r"ml_block_input\S*_inst_\d+__clk",
                          r"ml_block_weight\S*_inst_\d+__reset",
                          r"ml_block_input\S*_inst_\d+__clk",
                          r"ml_block_input\S*_inst_\d+__reset"]

    # Rename ML blocks to correct name
    line_list = filedata.splitlines()
    line_list = move_ios_into_module_body(line_list)
    line_list = remove_sim_block_defs(line_list, ["sim_True"])
    line_list = remove_non_existant_ports(line_list, non_existant_ports)
    line_list = remove_parameter_references(line_list)
    line_list = remove_width_0_ranges(line_list)
    filedata = '\n'.join(line_list)

    # replace HW block component names with actual names
    filedata = re.sub(r"(MLB_Wrapper__spec_)(\S*)(__projs_\S*)(\s+)(.*)",
                      r"\2\4\5",
                      filedata)
    filedata = re.sub(r"(HWB_Sim__spec_)(\S*)(__projs_\S*)(\s+)(.*)",
                      r"\2\4\5",
                      filedata)
    filedata = re.sub(r"(HWB_Sim__)(\S*)(\s+)(\S+)_inst(.*)",
                      r"\4\3\4_inst\5",
                      filedata)
    filedata = re.sub(r'\s+\[0:0\]\s+', r" ", filedata)
    return filedata


def generate_accelerator_given_mapping(module_name, mlb_spec, wb_spec, ab_spec,
                                       projection, write_to_file, emif_spec={},
                                       waddr=0, iaddr=0, oaddr=0, ws=True,
                                       fast_gen=True):
    """ Validate input specifications, generate  a system including both
        the statemachines and datapath.

    :param module_name: Top level module name and filename
    :param mlb_spec: Hardware definition of ML block
    :param wb_spec: Hardware definition of weight buffer
    :param ab_spec: Hardware definition of input buffer
    :param emif: Hardware definition of EMIF
    :param projection: Projection information
    :param write_to_file: Whether or not to write resulting verilog to file.
    """
    validate_inputs(wb_spec=wb_spec, ab_spec=ab_spec, mlb_spec=mlb_spec,
                    emif_spec=emif_spec, projections=[projection])
    emif_spec["simulation_model"] = "EMIF"
    t = state_machine_classes.MultipleLayerSystem(mlb_spec, wb_spec, ab_spec,
                                                  ab_spec, emif_spec,
                                                  projection, waddr, iaddr,
                                                  oaddr, ws, fast_gen)
    t.elaborate()
    return generate_verilog(t, write_to_file, module_name, not fast_gen)


def run_simulation(module, num_cycles, n=-1):
    """ Run simulation, and ensure that done is asserted

    :param module: pyMTL module to simulate
    :param num_cycles: number of cycles to simulate
    """
    # Start simulation and wait for done to be asserted
    utils.print_heading("Begin cycle accurate simulation", currstep + 2)
    if (n > -1):
        module.sel @= n
    module.sm_start @= 1
    module.sim_tick()
    module.sm_start @= 0
    for i in range(num_cycles):
        if (module.done):
            utils.printi(il, "Simulation complete (done asserted)", "GREEN")
            break
        module.sim_tick()
    module.sim_tick()
    assert(module.done)

    for i in range(20):
        # Just make sure that the EMIF finishes reading in data...
        module.sim_tick()


def simulate_accelerator_with_random_input(module_name, mlb_spec, wb_spec,
                                           ab_spec, emif_spec, projection,
                                           write_to_file,
                                           oaddr=0, iaddr=0, waddr=0,
                                           ws=True, validate_output=True):
    """
    Generate an accelerator.
    Fill the off-chip memory with random data (or assume initial data is
    included in the spec).

    :param module_name: Top level module name and filename
    :param mlb_spec: Hardware definition of ML block
    :param wb_spec: Hardware definition of weight buffer
    :param ab_spec: Hardware definition of input buffer
    :param emif_spec: Hardware definition of EMIF
    :param projections: Projection information
    :param write_to_file: Whether or not to write resulting verilog to file.
    :param oaddrs: Address of output actiavations in off-chip data
    :param iaddrs: Address of input actiavations in off-chip data
    :param waddrs: Address of weights in off-chip data
    :param ws: Weight stationary (or output stationary)
    :param validate_output: Whether to check the output correctness
    :param layer_sel: Which layers to simulate
    """
    validate_inputs(wb_spec=wb_spec, ab_spec=ab_spec, mlb_spec=mlb_spec,
                    emif_spec=emif_spec, projections=[projection])
    wb_spec["simulation_model"] = "Buffer"
    ab_spec["simulation_model"] = "Buffer"
    emif_spec["simulation_model"] = "EMIF"

    # Calculate buffer counts and dimensions
    wvalues_per_buf, wbuf_len, wbuf_count = utils.get_iw_buffer_dimensions(
        wb_spec, projection, 'W')
    ivalues_per_buf, ibuf_len, ibuf_count = utils.get_iw_buffer_dimensions(
        ab_spec, projection, 'I')
    ovalues_per_buf, obuf_len, obuf_count = utils.get_obuffer_dimensions(
        ab_spec, projection)

    # Fill EMIF with random data
    utils.print_heading("Generating random data to initialize " +
                        "off-chip memory", currstep + 0)
    wbuf = [[[random.randint(0, (2 ** projection["data_widths"]["W"]) - 1)
              for k in range(wvalues_per_buf)]
             for i in range(wbuf_len)]
            for j in range(wbuf_count)]
    wbuf_flat = utils.flatten_array(wbuf, projection["data_widths"]["W"])
    iaddr = len(wbuf_flat)
    ibuf = [[[random.randint(0, (2 ** projection["data_widths"]["I"]) - 1)
              for k in range(ivalues_per_buf)]
             for i in range(ibuf_len)]
            for j in range(ibuf_count)]
    ibuf_flat = utils.flatten_array(ibuf, projection["data_widths"]["I"])
    emif_data = wbuf_flat + ibuf_flat
    utils.printi(il, "\tGenerated random initial data: " + str(emif_data))
    emif_spec["fill"] = emif_data
    oaddr = len(emif_data)

    # Generate the accelerator
    utils.print_heading("Generating pyMTL model of accelerator", currstep + 1)
    t = state_machine_classes.MultipleLayerSystem(mlb_spec, wb_spec, ab_spec,
                                                  ab_spec, emif_spec,
                                                  projection, w_address=0,
                                                  i_address=iaddr,
                                                  o_address=oaddr, ws=ws)
    t.elaborate()
    t.apply(DefaultPassGroup())

    # Start simulation and wait for done to be asserted
    t.sim_reset()
    run_simulation(t, 2000)

    # Collect final EMIF data, and write it to a file.
    emif_vals = utils.read_out_stored_values_from_emif(
        t.emif_inst.sim_model.bufi, ovalues_per_buf,
        min(obuf_len, ibuf_len) * obuf_count,
        projection["data_widths"]["I"], oaddr)
    if (write_to_file):
        with open("final_offchip_data_contents.yaml", 'w') as file:
            yaml.dump(emif_vals, file, default_flow_style=False)

    # Check that the outputs are right!
    if (validate_output):
        utils.print_heading("Comparing final off-chip buffer contents " +
                            " with expected results", currstep + 3)
        obuf = [[[0 for i in range(ovalues_per_buf)]
                 for i in range(obuf_len)]
                for j in range(obuf_count)]
        obuf = utils.get_expected_outputs(obuf, ovalues_per_buf,
                                          wbuf, ibuf, ivalues_per_buf,
                                          projection)
        utils.printi(il, "Expected " + str(obuf))
        utils.printi(il, "Actual " + str(emif_vals))
        for bufi in range(obuf_count):
            for olen in range(min(obuf_len, ibuf_len) - 1):
                assert obuf[bufi][olen] == emif_vals[bufi *
                                                     min(obuf_len, ibuf_len)
                                                     + olen]
        utils.printi(il, "Simulation outputs were correct.", "GREEN")
    return emif_vals, t


def simulate_accelerator(module_name, mlb_spec, wb_spec, ab_spec, emif_spec,
                         projections, write_to_file,
                         oaddrs=[0], iaddrs=[0], waddrs=[0], ws=True,
                         validate_output=True, layer_sel=[0], simulate=True,
                         gen_ver=False, include_sim_models=False):
    """
    Generate an accelerator
    Fill the off-chip memory with random data (or assume initial data is
    included in the spec).

    :param module_name: Top level module name and filename
    :param mlb_spec: Hardware definition of ML block
    :param wb_spec: Hardware definition of weight buffer
    :param ab_spec: Hardware definition of input buffer
    :param emif_spec: Hardware definition of EMIF
    :param projections: Projection information
    :param write_to_file: Whether or not to write resulting verilog to file.
    :param oaddrs: Address of output actiavations in off-chip data
    :param iaddrs: Address of input actiavations in off-chip data
    :param waddrs: Address of weights in off-chip data
    :param ws: Weight stationary (or output stationary)
    :param validate_output: Whether to check the output correctness
    :param layer_sel: Which layers to simulate
    """
    if "inner_projection" in projections:
        projections = [projections]

    validate_inputs(wb_spec=wb_spec, ab_spec=ab_spec, mlb_spec=mlb_spec,
                    emif_spec=emif_spec, projections=projections, sim=simulate)
    ab_spec["simulation_model"] = "Buffer"
    wb_spec["simulation_model"] = "Buffer"
    emif_spec["simulation_model"] = "EMIF"

    utils.print_heading("Generating pyMTL model of accelerator", currstep + 1)
    if (gen_ver):
        generate_accelerator_given_mapping(module_name, mlb_spec, wb_spec,
                                           ab_spec, projections[0],
                                           write_to_file, emif_spec, waddrs[0],
                                           iaddrs[0], oaddrs[0], ws,
                                           fast_gen=(not include_sim_models))
        if (not simulate):
            return

    # Generate the statemachine
    utils.print_heading("Generating pyMTL model of accelerator", currstep + 1)
    t = state_machine_classes.MultipleLayerSystem(mlb_spec, wb_spec, ab_spec,
                                                  ab_spec, emif_spec,
                                                  projections,
                                                  w_address=waddrs,
                                                  i_address=iaddrs,
                                                  o_address=oaddrs, ws=ws)
    t.elaborate()
    if (simulate):
        t.apply(DefaultPassGroup())

    # Simulate each of the layers
    output_vals = [[] for m in range(max(layer_sel) + 1)]
    t.sim_reset()

    # Initial data is included in the EMIF spec.
    assert("fill" in emif_spec)
    for n in layer_sel:
        # Calculate buffer counts and dimensions
        wvalues_per_buf, wbuf_len, wbuf_count = utils.get_iw_buffer_dimensions(
            wb_spec, projections[n], 'W')
        ivalues_per_buf, ibuf_len, ibuf_count = utils.get_iw_buffer_dimensions(
            ab_spec, projections[n], 'I')
        ovalues_per_buf, obuf_len, obuf_count = utils.get_obuffer_dimensions(
            ab_spec, projections[n])

        # Run the simulation for 2000 cycles
        if (simulate):
            print("Simulate layer " + str(n))
            wbuf = utils.read_out_stored_values_from_array(
                emif_spec["fill"], wvalues_per_buf,
                wbuf_len * wbuf_count, projections[n]["data_widths"]["W"],
                waddrs[n], wbuf_len)
            ibuf = utils.read_out_stored_values_from_array(
                emif_spec["fill"], ivalues_per_buf,
                ibuf_len * ibuf_count, projections[n]["data_widths"]["I"],
                iaddrs[n], ibuf_len)
            run_simulation(t, 2000, n)

            # Collect final EMIF data (and later write to file)
            # buffer_vals = utils.read_out_stored_values_from_emif(
            #    t.datapath.input_act_modules.ml_block_input_inst_0.
            #     sim_model_inst0, ivalues_per_buf,
            #    10,
            #    projections[n]["data_widths"]["I"], 0)
            # print(buffer_vals)
            # buffer_vals = utils.read_out_stored_values_from_emif(
            #    t.datapath.input_act_modules.ml_block_input_inst_1.
            #     sim_model_inst0, ivalues_per_buf,
            #    10,
            #    projections[n]["data_widths"]["I"], 0)
            # print(buffer_vals)

    if (simulate):
        # Collect final EMIF data (and later write to file)
        emif_vals = utils.read_out_stored_values_from_emif(
            t.emif_inst.sim_model.bufi, ovalues_per_buf,
            min(obuf_len, ibuf_len) * obuf_count,
            projections[n]["data_widths"]["I"], oaddrs[n])
        output_vals[n] = emif_vals

    # Check that the outputs are right!
    if (simulate and validate_output):
        obuf = [[[0 for i in range(ovalues_per_buf)]
                for i in range(obuf_len)]
                for j in range(obuf_count)]
        obuf = utils.get_expected_outputs(obuf, ovalues_per_buf, wbuf,
                                          ibuf, ivalues_per_buf,
                                          projections[-1])
        utils.print_heading("Comparing final off-chip buffer contents" +
                            "with expected results", currstep + 3)
        utils.printi(il, "Expected " + str(obuf))
        utils.printi(il, "Actual " + str(emif_vals))
        for bufi in range(obuf_count):
            for olen in range(min(obuf_len, ibuf_len) - 1):
                assert obuf[bufi][olen] == \
                    emif_vals[bufi * min(obuf_len, ibuf_len) + olen]
        utils.printi(il, "Simulation outputs were correct.", "GREEN")

    if (write_to_file):
        with open("final_offchip_data_contents.yaml", 'w') as file:
            yaml.dump(output_vals, file, default_flow_style=False)

    return output_vals, t


def generate_accelerator_for_layers(module_name, mlb_spec, wb_spec,
                                    ab_spec, emif_spec,
                                    pe_count, layer,
                                    oaddr=0, iaddr=0, waddr=0,
                                    simulate=True,
                                    preload_o=1,
                                    preload_i=1,
                                    ws=True, fast_gen=True):
    """
    Generate an accelerator for a given set of layers.

    :param module_name: Top level module name and filename
    :param mlb_spec: Hardware definition of ML block
    :param wb_spec: Hardware definition of weight buffer
    :param ab_spec: Hardware definition of input buffer
    :param emif_spec: Hardware definition of EMIF
    :param pe_count: Number of PEs available
    :param oaddr: Address of output actiavations in off-chip data
    :param iaddr: Address of input actiavations in off-chip data
    :param waddr: Address of weights in off-chip data
    :param ws: Weight stationary (or output stationary)
    """
    global currstep
    utils.print_heading("Find an appropriate mapping vector for given layer " +
                        "specification", 1)
    currstep = 1
    soft_logic_required = False
    if (mlb_spec['access_patterns']['AP1'] == 0):
        soft_logic_required = True

    mappings, mapping_score = constraint_evaluation.find_mappings(
        mlb_spec,
        layer["loop_dimensions"],
        pe_count,
        preload_o=preload_o,
        preload_i=preload_i,
        suggested_solution=None,
        enable_soft_logic=soft_logic_required
    )
    assert(len(mappings) == 1)
    proj = {}
    proj["activation_function"] = layer["activation_function"]
    proj["stride"] = layer["stride"]
    proj["dilation"] = layer["dilation"]
    proj["data_widths"] = layer["data_widths"]
    proj["temporal_projection"] = {'RY': (mappings[0]["RXT"] *
                                          mappings[0]["RYT"]),
                                   'C': mappings[0]["CT"],
                                   'E': mappings[0]["ET"],
                                   'PX': mappings[0]["PXT"],
                                   'PY': mappings[0]["PYT"],
                                   'B': mappings[0]["BT"],
                                   'G': 1}
    proj["outer_projection"] = {'RY': mappings[0]["RYO"],
                                'C': mappings[0]["CO"],
                                'RX': mappings[0]["RXO"],
                                'E': mappings[0]["EO"],
                                'PX': 1,
                                'PY': 1,
                                'B': (mappings[0]["BO"] *
                                      mappings[0]["PXO"] *
                                      mappings[0]["PYO"]),
                                'G': 1}
    proj["inner_projection"] = {'RY': mappings[0]["RYI"],
                                'C': mappings[0]["CI"],
                                'RX': mappings[0]["RXI"],
                                'E': mappings[0]["EI"],
                                'PX': 1,
                                'PY': 1,
                                'B': (mappings[0]["BI"] *
                                      mappings[0]["PXI"] *
                                      mappings[0]["PYI"]),
                                'G': 1}
    if (preload_o > 0):
        proj['outer_projection']['PRELOAD'] = [{'dtype': 'W',
                                                'bus_count': preload_o}]
    if (preload_i > 0):
        proj['inner_projection']['PRELOAD'] = [{'dtype': 'W',
                                                'bus_count': preload_i}]
    print(proj)

    simulate_accelerator(
        module_name, mlb_spec, wb_spec,  ab_spec, emif_spec, proj, True,
        [oaddr], [iaddr], [waddr], ws, simulate=simulate,
        gen_ver=True, include_sim_models=(not fast_gen))
