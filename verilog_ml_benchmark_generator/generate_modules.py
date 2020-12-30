"""
    This file contains the user-facing functions that perform main tasks...
       - Generate pyMTL statemachine modules
       - Simulate using cycle-accurate pyMTL simulations
       - Generate verilog and perform post processing
    Also some helper functions for post processing verilog to make it
    odin compatible.
"""
import sys
import os
import re
import random
import yaml
import constraint_evaluation
from jsonschema import validate
from pymtl3 import *
from pymtl3.passes.backends.verilog import *
from pymtl3.passes.backends.verilog import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import module_classes
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
    "required": ["name", "width", "direction", "type"]
}
inner_proj_schema = {
    "type": "object",
    "properties": {
        "URW": {"value": {"type": "int", "minimum": 1}},
        "URN": {"value": {"type": "int", "minimum": 1}},
        "UE": {"value": {"type": "int", "minimum": 1}},
        "UB": {"value": {"type": "int", "minimum": 1}},
        "UG": {"value": {"type": "int", "minimum": 1}},
        "PRELOAD": {"type": "array",
                    "items": {"type": "object",
                              "properties":
                              {"dtype": {"type": "string",
                                         "enum": ["W", "I", "O"]},
                               "bus_count": {"type": "number",
                                             "minimum": 1}},
                              "required": ["dtype"]}},
    },
    "required": ["URW", "URN", "UB", "UE", "UG"]
}
datawidth_schema = {
    "type": "object",
    "properties": {
        "W": {"type": "number"},
        "I": {"type": "number"},
        "O": {"type": "number"}},
    "required": ["W", "I", "O"]
}
proj_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "activation_function": {"type": "string", "enum": ["RELU"]},
        "outer_projection": inner_proj_schema,
        "inner_projection": inner_proj_schema,
        "stream_info": datawidth_schema
    },
    "required": ["activation_function", "outer_projection",
                 "inner_projection", "stream_info"]
}
MAC_info_schema = {"type": "object",
                   "properties": {
                       "num_units": {"type": "number"},
                       "data_widths": datawidth_schema},
                   "required": ["num_units", "data_widths"]}
buffer_spec_schema = {
    "type": "object",
    "properties": {
        "block_name": {"type": "string"},
        "simulation_model": {"type": "string",
                             "enum": ["MLB", "Buffer", "EMIF"]},
        "MAC_info": MAC_info_schema,
        "ports": {"type": "array", "items": port_schema}},
    "required": ["ports"]
}
mlb_spec_schema = {
    "type": "object",
    "properties": {
        "block_name": {"type": "string"},
        "simulation_model": {"type": "string"},
        "MAC_info": MAC_info_schema,
        "ports": {"type": "array", "items": port_schema}},
    "required": ["ports", "MAC_info"]
}


def generate_verilog(component, write_to_file, module_name):
    """ Call pyMTL elaboration methods to elaborate a Component instance,
        and post-process generated verilog.

    :param component: PyMTL Component to elaborate
    :param write_to_file: Whether or not to write resulting verilog to file.
    :param component: Name of generated verilog file
    """
    # Generate the outer module containing many MLBs
    component.set_metadata(VerilogTranslationPass.enable, True)
    component.set_metadata(VerilogTranslationPass.explicit_file_name,
                           module_name + "_pymtl.v")
    component.set_metadata(VerilogTranslationPass.explicit_module_name,
                           module_name)
    component.apply(VerilogTranslationPass())

    # Post-process generated file so that it is compatible with odin :(
    outtxt = odinify(module_name + "_pymtl.v")
    if (write_to_file):
        with open(module_name + ".v", 'w') as file:
            file.write(outtxt)
    return [component, outtxt]


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
                                            str(foundlp.group(1)) + "\]",
                                            "[" + str(foundlp.group(1)) + "]",
                                            line_list[lineidx])
        lineidx += 1

    return line_list


def odinify(filename_in):
    """ Make verilog file compatible with odin.
        This is a bit of a hack, but necessary to run VTR.

    :param filename_in: Name of file to fix up
    """

    # Post-process generated file so that it is compatible with odin :(
    with open(filename_in, 'r') as file:
        filedata = file.read()

    # Replace some system verilog syntax
    filedata = filedata.replace('logic', 'wire')  # so far this works...
    filedata = filedata.replace('always_comb', 'always@(*)')
    filedata = filedata.replace('always_ff', 'always')
    filedata = filedata.replace('1\'d1', '2\'d1')
    filedata = filedata.replace('{}', 'empty')

    # Odin can't handle wide values
    filedata = filedata.replace('64\'d', '62\'d')
    filedata = filedata.replace('128\'d', '62\'d')
    filedata = filedata.replace('210\'d', '62\'d')
    filedata = filedata.replace('416\'d', '62\'d')
    filedata = filedata.replace('216\'d', '62\'d')

    # pyMTL adds clk and reset to everything... but we dont want it
    # in some cases.
    non_existant_ports = [r"ml_block_weights_inst_\d+__clk",
                          r"ml_block_input_inst_\d+__clk",
                          r"ml_block_weights_inst_\d+__reset",
                          r"ml_block_input_inst_\d+__clk",
                          r"ml_block_input_inst_\d+__reset"]

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
    filedata = re.sub(r'\s+\[0:0\]\s+', r" ", filedata)
    return filedata


def generate_full_datapath(module_name, mlb_spec, wb_spec, ab_spec,
                           projections, write_to_file):
    """ Validate input specifications, generate just the datapath and then
        write resulting verilog to file ``module_name``.v

    :param module_name: Top level module name and filename
    :param mlb_spec: Hardware definition of ML block
    :param wb_spec: Hardware definition of weight buffer
    :param ab_spec: Hardware definition of input buffer
    :param projection: Projection information
    :param write_to_file: Whether or not to write resulting verilog to file.
    """
    validate(instance=wb_spec, schema=buffer_spec_schema)
    validate(instance=ab_spec, schema=buffer_spec_schema)
    validate(instance=mlb_spec, schema=mlb_spec_schema)
    for proj in projections:
        validate(instance=proj, schema=proj_schema)

    # Generate the outer module containing many MLBs
    t = module_classes.Datapath(mlb_spec, wb_spec, ab_spec, ab_spec,
                                projections)
    t.elaborate()
    return generate_verilog(t, write_to_file, module_name)


def generate_statemachine(module_name, mlb_spec, wb_spec, ab_spec,
                          projection, write_to_file, emif_spec={},
                          waddr=0, iaddr=0, oaddr=0, ws=True):
    """ Validate input specifications, generate a system including both
        the statemachines and datapath.

    :param module_name: Top level module name and filename
    :param mlb_spec: Hardware definition of ML block
    :param wb_spec: Hardware definition of weight buffer
    :param ab_spec: Hardware definition of input buffer
    :param emif: Hardware definition of EMIF
    :param projection: Projection information
    :param write_to_file: Whether or not to write resulting verilog to file.
    """
    validate(instance=wb_spec, schema=buffer_spec_schema)
    validate(instance=ab_spec, schema=buffer_spec_schema)
    validate(instance=mlb_spec, schema=mlb_spec_schema)
    validate(instance=projection, schema=proj_schema)
    validate(instance=projection, schema=emif_spec)
    t = state_machine_classes.MultipleLayerSystem(mlb_spec, wb_spec, ab_spec,
                                                  ab_spec, emif_spec,
                                                  projection, waddr, iaddr,
                                                  oaddr, ws)
    t.elaborate()
    return generate_verilog(t, write_to_file, module_name)


def run_simulation(module, num_cycles, n=-1):
    """ Run simulation, and ensure that done is asserted

    :param module: pyMTL module to simulate
    :param num_cycles: number of cycles to simulate
    """
    # Start simulation and wait for done to be asserted
    utils.print_heading("Begin cycle accurate simulation", currstep + 2)
    module.sim_reset()
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
                                           write_to_file, randomize=True,
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
    :param randomize: Whether to randomize input data or read from emif spec.
    :param oaddrs: Address of output actiavations in off-chip data
    :param iaddrs: Address of input actiavations in off-chip data
    :param waddrs: Address of weights in off-chip data
    :param ws: Weight stationary (or output stationary)
    :param validate_output: Whether to check the output correctness
    :param layer_sel: Which layers to simulate
    """
    emif_spec["simulation_model"] = "EMIF"
    wb_spec["simulation_model"] = "Buffer"
    ab_spec["simulation_model"] = "Buffer"
    mlb_spec["simulation_model"] = "MLB"
    validate(instance=wb_spec, schema=buffer_spec_schema)
    validate(instance=ab_spec, schema=buffer_spec_schema)
    validate(instance=mlb_spec, schema=mlb_spec_schema)
    validate(instance=projection, schema=proj_schema)

    # Calculate buffer counts and dimensions
    wvalues_per_buf, wbuf_len, wbuf_count = utils.get_iw_buffer_dimensions(
        wb_spec, projection, 'W')
    ivalues_per_buf, ibuf_len, ibuf_count = utils.get_iw_buffer_dimensions(
        ab_spec, projection, 'I')
    ovalues_per_buf, obuf_len, obuf_count = utils.get_obuffer_dimensions(
        ab_spec, projection)

    # Either randomize input data, or get it from the EMIF yaml file.
    if (randomize):
        # Fill EMIF with random data
        utils.print_heading("Generating random data to initialize " +
                            "off-chip memory", currstep + 0)
        wbuf = [[[random.randint(0, (2 ** projection["stream_info"]["W"]) - 1)
                  for k in range(wvalues_per_buf)]
                 for i in range(wbuf_len)]
                for j in range(wbuf_count)]
        wbuf_flat = utils.flatten_array(wbuf, projection["stream_info"]["W"])
        iaddr = len(wbuf_flat)
        ibuf = [[[random.randint(0, (2 ** projection["stream_info"]["I"]) - 1)
                  for k in range(ivalues_per_buf)]
                 for i in range(ibuf_len)]
                for j in range(ibuf_count)]
        ibuf_flat = utils.flatten_array(ibuf, projection["stream_info"]["I"])
        emif_data = wbuf_flat + ibuf_flat
        utils.printi(il, "\tGenerated random initial data: " + str(emif_data))
        emif_spec["parameters"]["fill"] = emif_data
        oaddr = len(emif_data)
    else:
        # Initial data is included in the EMIF spec.
        assert("fill" in emif_spec["parameters"])
        wbuf = utils.read_out_stored_values_from_array(
            emif_spec["parameters"]["fill"], wvalues_per_buf,
            wbuf_len * wbuf_count, projection["stream_info"]["W"],
            waddr, wbuf_len)
        ibuf = utils.read_out_stored_values_from_array(
            emif_spec["parameters"]["fill"], ivalues_per_buf,
            ibuf_len * ibuf_count, projection["stream_info"]["I"],
            iaddr, ibuf_len)

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
    run_simulation(t, 2000)

    # Collect final EMIF data, and write it to a file.
    emif_vals = utils.read_out_stored_values_from_emif(
        t.emif_inst.sim_model.buf, ovalues_per_buf,
        min(obuf_len, ibuf_len) * obuf_count,
        projection["stream_info"]["I"], oaddr)
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
                         projections, write_to_file, randomize=True,
                         oaddrs=[0], iaddrs=[0], waddrs=[0], ws=True,
                         validate_output=True, layer_sel=[0], simulate=True,
                         gen_ver=False):
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
    :param randomize: Whether to randomize input data or read from emif spec.
    :param oaddrs: Address of output actiavations in off-chip data
    :param iaddrs: Address of input actiavations in off-chip data
    :param waddrs: Address of weights in off-chip data
    :param ws: Weight stationary (or output stationary)
    :param validate_output: Whether to check the output correctness
    :param layer_sel: Which layers to simulate
    """
    mlb_spec["simulation_model"] = "MLB"
    if "inner_projection" in projections:
        projections = [projections]
    validate(instance=wb_spec, schema=buffer_spec_schema)
    validate(instance=ab_spec, schema=buffer_spec_schema)
    validate(instance=mlb_spec, schema=mlb_spec_schema)

    if (gen_ver):
        generate_statemachine(module_name, mlb_spec, wb_spec, ab_spec,
                              projections[0], write_to_file, emif_spec,
                              waddrs[0], iaddrs[0], oaddrs[0], ws)
    emif_spec["simulation_model"] = "EMIF"
    wb_spec["simulation_model"] = "Buffer"
    ab_spec["simulation_model"] = "Buffer"
    for projection in projections:
        validate(instance=projection, schema=proj_schema)

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
    output_vals = [[] for l in range(max(layer_sel) + 1)]
    for n in layer_sel:
        # Calculate buffer counts and dimensions
        wvalues_per_buf, wbuf_len, wbuf_count = utils.get_iw_buffer_dimensions(
            wb_spec, projections[n], 'W')
        ivalues_per_buf, ibuf_len, ibuf_count = utils.get_iw_buffer_dimensions(
            ab_spec, projections[n], 'I')
        ovalues_per_buf, obuf_len, obuf_count = utils.get_obuffer_dimensions(
            ab_spec, projections[n])

        # Initial data is included in the EMIF spec.
        assert("fill" in emif_spec["parameters"])

        # Run the simulation for 2000 cycles
        if (simulate):
            wbuf = utils.read_out_stored_values_from_array(
                emif_spec["parameters"]["fill"], wvalues_per_buf,
                wbuf_len * wbuf_count, projections[n]["stream_info"]["W"],
                waddrs[n], wbuf_len)
            ibuf = utils.read_out_stored_values_from_array(
                emif_spec["parameters"]["fill"], ivalues_per_buf,
                ibuf_len * ibuf_count, projections[n]["stream_info"]["I"],
                iaddrs[n], ibuf_len)
            run_simulation(t, 2000, n)

            # Collect final EMIF data (and later write to file)
            emif_vals = utils.read_out_stored_values_from_emif(
                t.emif_inst.sim_model.buf, ovalues_per_buf,
                min(obuf_len, ibuf_len) * obuf_count,
                projections[n]["stream_info"]["I"], oaddrs[n])
            output_vals[n] = emif_vals

        # Check that the outputs are right!
        if (simulate and validate_output):
            obuf = [[[0 for i in range(ovalues_per_buf)]
                    for i in range(obuf_len)]
                    for j in range(obuf_count)]
            obuf = utils.get_expected_outputs(obuf, ovalues_per_buf, wbuf,
                                              ibuf, ivalues_per_buf,
                                              projections[n])
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
                                    ws=True):
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
    mappings, mapping_score = constraint_evaluation.find_mappings(
        mlb_spec,
        layer["loop_dimensions"],
        pe_count
    )
    assert(len(mappings) == 1)
    proj = {}
    proj["activation_function"] = layer["activation_function"]
    proj["stride"] = layer["stride"]
    proj["dilation"] = layer["dilation"]
    proj["stream_info"] = layer["stream_info"]
    proj["temporal_projection"] = {'URN': {'value': (mappings[0]["RXT"] *
                                                     mappings[0]["RYT"] *
                                                     mappings[0]["CT"]),
                                           'x': mappings[0]["RXT"],
                                           'y': mappings[0]["RYT"],
                                           'chans': mappings[0]["CT"]},
                                   'UE': {'value': mappings[0]["ET"]},
                                   'UB': {'value': (mappings[0]["BT"] *
                                                    mappings[0]["PXT"] *
                                                    mappings[0]["PYT"]),
                                          'x': mappings[0]["PXT"],
                                          'y': mappings[0]["PYT"],
                                          'batches': mappings[0]["BT"]},
                                   'UG': {'value': 1}}
    proj["outer_projection"] = {'URN': {'value': (mappings[0]["RYO"] *
                                                  mappings[0]["CO"]),
                                        'x': 1,
                                        'y': mappings[0]["RYO"],
                                        'chans': mappings[0]["CO"]},
                                'URW': {'value': mappings[0]["RXO"],
                                        'x': mappings[0]["RXO"],
                                        'y': 1},
                                'UE': {'value': mappings[0]["EO"]},
                                'UB': {'value': (mappings[0]["BO"] *
                                                 mappings[0]["PXO"] *
                                                 mappings[0]["PYO"]),
                                       'x': mappings[0]["PXO"],
                                       'y': mappings[0]["PYO"],
                                       'batches': mappings[0]["BO"]},
                                'UG': {'value': 1}}
    proj["inner_projection"] = {'URN': {'value': (mappings[0]["RYI"] *
                                                  mappings[0]["CI"]),
                                        'x': 1,
                                        'y': mappings[0]["RYI"],
                                        'chans': mappings[0]["CI"]},
                                'URW': {'value': mappings[0]["RXI"],
                                        'x': mappings[0]["RXI"],
                                        'y': 1},
                                'UE': {'value': mappings[0]["EI"]},
                                'UB': {'value': (mappings[0]["BI"] *
                                                 mappings[0]["PXI"] *
                                                 mappings[0]["PYI"]),
                                       'x': mappings[0]["PXI"],
                                       'y': mappings[0]["PYI"],
                                       'batches': mappings[0]["BI"]},
                                'UG': {'value': 1}}
    
    proj['inner_projection']['PRELOAD'] = [{'dtype':'W', 'bus_count':1}]
    proj['outer_projection']['PRELOAD'] = [{'dtype':'W', 'bus_count':1}]
    print(proj)
    
    outvals, testinst = simulate_accelerator(
        module_name, mlb_spec, wb_spec,  ab_spec, emif_spec, proj, True,
        False, [oaddr], [iaddr], [waddr], ws, simulate=simulate,
        gen_ver=True)

    # Calculate buffer counts and dimensions
    return outvals, testinst
