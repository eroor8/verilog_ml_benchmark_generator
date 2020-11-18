"""<later>."""
import sys
import os
import inspect
import fileinput
import re
import random
import math
import yaml
from jsonschema import validate
from pymtl3 import *
from pymtl3.examples.ex00_quickstart import FullAdder
from pymtl3.passes.backends.verilog import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import module_classes
import utils
import state_machine_classes

# Schemas to validate input yamls
supported_activations=["RELU"]
datatype_mlb = ["I","O","W","C","CLK","RESET", "I_EN", "W_EN", "ACC_EN", "WEN"]
datatype_emif = ["AVALON_ADDRESS", "AVALON_READ", "AVALON_WRITE",
                 "AVALON_READDATA", "AVALON_WRITEDATA", "AVALON_READDATAVALID",
                 "AVALON_WAITREQUEST"]
datatype_buffer = ["ADDRESS", "DATA"]
datatype_any = datatype_mlb+datatype_buffer+datatype_emif
port_schema = {
    "type" : "object",
    "properties" : {
        "name" : {"type" : "string", "minLength":1},
        "width" : {"type" : "number", "minimum":1},
        "direction" : {"type" : "string", "enum":["in", "out"]},
        "type" : {"type":"string", "enum":datatype_any}
    },
    "required" : ["name","width","direction","type"]
}
inner_proj_schema = {
    "type" : "object",
    "properties" : {
        "URW" : {"value": {"type":"int", "minimum":1}},
        "URN" : {"value": {"type":"int", "minimum":1}},
        "UE" : {"value": {"type":"int", "minimum":1}},
        "UB" : {"value": {"type":"int", "minimum":1}},
        "UG" : {"value": {"type":"int", "minimum":1}},
        "PRELOAD" : {"type":"array",
                   "items": {"type":"object",
                             "properties":
                                {"dtype": {"type":"string", "enum":["W","I","O"]},
                                 "bus_count": {"type":"number", "minimum":1}},
                             "required": ["dtype"]
                    }
         },
    },
    "required" : ["URW","URN","UB","UE","UG"]
}
datawidth_schema = {
    "type":"object",
    "properties": {
        "W":{"type":"number"},
        "I":{"type":"number"},
        "O":{"type":"number"}
    },
    "required" : ["W","I","O"]
}
proj_schema = {
    "type":"object",
    "properties": {
        "name" : {"type":"string"},
        "activation_function": {"type":"string", "enum":["RELU"]},
        "outer_projection" : inner_proj_schema,  
        "inner_projection" : inner_proj_schema,
        "stream_info" : datawidth_schema,
    },
    "required" : ["activation_function","outer_projection",
                  "inner_projection", "stream_info"]
}
MAC_info_schema = {"type":"object",
                   "properties": {
                       "num_units" : {"type":"number"},
                       "data_widths" : datawidth_schema,
                   },
                   "required" : ["num_units", "data_widths"]}
buffer_spec_schema = {
    "type" : "object",
    "properties" : {
        "block_name" : {"type":"string"},
        "simulation_model" : {"type":"string", "enum":["MLB","Buffer", "EMIF"]},
        "MAC_info" : MAC_info_schema,
        "ports" : {"type":"array",
                   "items": port_schema
                  },
        },
    "required" : ["ports"]
}
mlb_spec_schema = {
    "type" : "object",
    "properties" : {
        "block_name" : {"type":"string"},
        "simulation_model" : {"type":"string"},
        "MAC_info" : MAC_info_schema,
        "ports" : {"type":"array",
                   "items": port_schema
                  },
        },
    "required" : ["ports", "MAC_info"]
}

def elab_and_write(component, write_to_file, module_name):   
    """ Call pyMTL elobration methods to elaborate a Component instance,
        and post-process generated verilog. 
        
    :param component: PyMTL Component to elaborate
    :type component: Component class
    :param write_to_file: Whether or not to write resulting verilog to file.
    :type  write_to_file: bool
    :param component: Name of generated verilog file
    :type component: string
    """
    # Generate the outer module containing many MLBs
    component.elaborate()
    component.set_metadata(VerilogTranslationPass.enable, True)
    component.set_metadata(VerilogTranslationPass.explicit_file_name,
                           module_name+"_pymtl.v")
    component.set_metadata(VerilogTranslationPass.explicit_module_name,
                           module_name)
    component.apply(VerilogTranslationPass())

    # Post-process generated file so that it is compatible with odin :(
    outtxt = odinify(module_name+"_pymtl.v")
    if (write_to_file):
        with open(module_name+".v", 'w') as file:
            file.write(outtxt)
    return [component, outtxt]

def remove_sim_block_defs(line_list, sim_blocks):
    """ Remove definitions of given modules from list of lines
        
    :param line_list: Original file
    :type  line_list: array of strings
    :param sim_blocks: Modules to remove
    :type  line_list: array of strings
    """
    lineidx = 0
    skipCopy=False
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
        lineidx+= 1
    return line_list

def remove_non_existant_ports(line_list, non_existant_ports):
    """ Remove references to list of ports that shouldn't exist
        
    :param line_list: Original file
    :type  line_list: array of strings
    :param non_existant_ports: Ports to remove
    :type  non_existant_ports: array of strings
    """
    lineidx = 0
    while lineidx < len(line_list):
        # Remove nonexistant ports
        for nep in non_existant_ports:
            found_nep_def = re.search(r"\.*"+nep+r"\s*;*,*\)*$",
                                      line_list[lineidx])
            found_nep = re.search(r"\(.*"+nep+r"\s*\)", line_list[lineidx])
            if found_nep:
                # If there was no comma at the end of the line,
                # remove comma from prev line.
                curr_comma = re.search(r"\)\s*,", line_list[lineidx])
                if not curr_comma:
                    prev_line = re.search("(.*),", line_list[lineidx-1])
                    line_list[lineidx-1] = prev_line.group(1)
                line_list[lineidx] = "//" + line_list[lineidx]
        lineidx+= 1
        
    return line_list


def move_ios_into_module_body(line_list):
    """ Moves input and output declarations out of the module
    definition and into the module body.
        
    :param line_list: Original file
    :type  line_list: array of strings
    """
    lineidx = 0
    stored_ios = []
    while lineidx < len(line_list):
        # Move inputs out of the module definition
        origline = line_list[lineidx]
        foundio = re.findall(r"(in|out)put\s+wire\s+\[\d+:\d+\]\s+\S+\s*,*",
                             origline)
        foundmodstart = re.search(";",origline)
        assert len(foundio) < 2
        if len(foundio)>0:
            match = re.search(r"(in|out)(put)(\s+)(wire)(\s+)(\[\d+:\d+\])" +
                              r"(\s+)(\S+)(\s+)(,*)", origline)
            assert match
            line_list[lineidx] = match.group(8) + match.group(10)
            stored_ios.append(match.group(1) + match.group(2) + match.group(3) + 
                              match.group(6) + match.group(7) + match.group(8) +
                              match.group(9) + ';')
            lineidx+=1;
        elif len(stored_ios) > 0 and foundmodstart:
            p1 = origline[0:foundmodstart.end()]
            line_list[lineidx] = p1
            for storedio in stored_ios:
                lineidx+=1;
                line_list.insert(lineidx, storedio)
            lineidx+=1;
            line_list.insert(lineidx, origline[foundmodstart.end():])
            stored_ios = []
        else:
            lineidx+=1;

    return line_list

def remove_parameter_references(line_list):
    """ Replace all references to parameters with their values
        
    :param line_list: Original file
    :type  line_list: array of strings
    """
    lineidx = 0
    localparams ={}
    while lineidx < len(line_list):
        # Keep track of local params
        foundlp = re.search(r"localparam\s+wire\s+\[\d+:\d+\]\s+(\S+)\s*=\s*(\S+);",
                            line_list[lineidx])
        if foundlp: 
            line_list[lineidx] = "//" + line_list[lineidx]
            localparams[foundlp.group(1)] = foundlp.group(2)
        else:
            for localparam in localparams:
                line_list[lineidx] = re.sub(r"\d\'\(\s*"+localparam+r"\s*\)",
                                            localparams[localparam],
                                            line_list[lineidx])
        lineidx+= 1

    return line_list

def odinify(filename_in):
    """ Make verilog file compatible with odin. 
        This is a bit of a hack, but necessary to run VTR
        
    :param filename_in: Name of file to fix up
    :type  filename_in: string
    """
    
    # Post-process generated file so that it is compatible with odin :(
    with open(filename_in, 'r') as file:
        filedata = file.read()
        
    # Replace some system verilog syntax
    filedata = filedata.replace('logic', 'wire') # so far this works...
    filedata = filedata.replace('always_comb', 'always@(*)')
    filedata = filedata.replace('always_ff', 'always')
    filedata = filedata.replace('{}', 'empty')
    
    # Odin can't handle wide values
    filedata = filedata.replace('64\'d', '62\'d')
    filedata = filedata.replace('128\'d', '62\'d')

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
    filedata =  '\n'.join(line_list)
    
    # replace HW block component names with actual names
    filedata = re.sub(r"(HWB_Sim__spec_)(\S*)(__proj_\S*)(\s+)(.*)", r"\2\4\5",
                      filedata)
    filedata = re.sub(r'\s+\[0:0\]\s+', r" ", filedata)
    return filedata

def generate_full_datapath(module_name, mlb_spec, wb_spec, ab_spec, \
                           projection, write_to_file):
    """ Validate input specifications, generate the datapath and then write
        resulting verilog to file ``module_name``.v

    :param module_name: Top level module name and filename
    :type module_name: string
    :param mlb_spec: Hardware definition of ML block
    :type mlb_spec: dict
    :param wb_spec: Hardware definition of weight buffer
    :type wb_spec: dict
    :param ab_spec: Hardware definition of input buffer
    :type ab_spec: dict
    :param projection: Projection information
    :type projection: dict
    :param write_to_file: Whether or not to write resulting verilog to file.
    :type  write_to_file: bool
    """
    validate(instance=wb_spec, schema=buffer_spec_schema)
    validate(instance=ab_spec, schema=buffer_spec_schema)
    validate(instance=mlb_spec, schema=mlb_spec_schema)
    validate(instance=projection, schema=proj_schema)
    
    # Generate the outer module containing many MLBs
    t = module_classes.Datapath(mlb_spec, wb_spec, ab_spec, ab_spec, projection)
    return elab_and_write(t, write_to_file, module_name)

def generate_statemachine(module_name, mlb_spec, wb_spec, ab_spec, \
                           projection, write_to_file, emif_spec={},
                           waddr=0, iaddr=0, oaddr=0):
    """ Validate input specifications, generate the datapath and then write
        resulting verilog to file ``module_name``.v

    :param module_name: Top level module name and filename
    :type module_name: string
    :param mlb_spec: Hardware definition of ML block
    :type mlb_spec: dict
    :param wb_spec: Hardware definition of weight buffer
    :type wb_spec: dict
    :param ab_spec: Hardware definition of input buffer
    :type ab_spec: dict
    :param projection: Projection information
    :type projection: dict
    :param write_to_file: Whether or not to write resulting verilog to file.
    :type  write_to_file: bool
    """
    validate(instance=wb_spec, schema=buffer_spec_schema)
    validate(instance=ab_spec, schema=buffer_spec_schema)
    validate(instance=mlb_spec, schema=mlb_spec_schema)
    validate(instance=projection, schema=proj_schema)
    
    if (emif_spec == {}):
        # Generate the outer module containing many MLBs
        t = state_machine_classes.StateMachine(mlb_spec, wb_spec, ab_spec,
                                               ab_spec, projection)
    else:
        validate(instance=projection, schema=emif_spec)
        t = state_machine_classes.StateMachineEMIF(mlb_spec, wb_spec, ab_spec,
                                                   ab_spec, emif_spec,
                                               projection, waddr, iaddr, oaddr)
    return elab_and_write(t, write_to_file, module_name)

def simulate_statemachine(module_name, mlb_spec, wb_spec, ab_spec, emif_spec, \
                           projection, write_to_file, randomize=True,
                           oaddr=0, iaddr=0, waddr=0, validate_output=True):
    """
    Generate a Statemachine with an EMIF interface
    Fill the off-chip memory with random data (or assume initial data is
    included in the spec).
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

    if (randomize):
        # Fill EMIF with random data
        utils.print_heading("Generating random data to initialize off-chip memory",0)
        wbuf = [[[random.randint(0,(2**projection["stream_info"]["W"])-1)
                for k in range(wvalues_per_buf)]    
                for i in range(wbuf_len)]           
                for j in range(wbuf_count)]         
        wbuf_flat = [sum((lambda i: inner[i] * \
                          (2**(i*projection["stream_info"]["W"])))(i)
                         for i in range(len(inner))) 
                     for outer in wbuf for inner in outer]
        iaddr = len(wbuf_flat)
        ibuf = [[[random.randint(0,(2**projection["stream_info"]["I"])-1)
                 for k in range(ivalues_per_buf)]            
                 for i in range(ibuf_len)]                   
                 for j in range (ibuf_count)]                
        ibuf_flat = [sum((lambda i: inner[i] * \
                          (2**(i*projection["stream_info"]["I"])))(i)
                         for i in range(len(inner))) 
                     for outer in ibuf for inner in outer]
        emif_data = wbuf_flat + ibuf_flat
        print("\tGenerated Data: " + str(emif_data))
        emif_spec["parameters"]["fill"] = emif_data
        oaddr = len(emif_data)
    else:
        wbuf = utils.read_out_stored_values_from_array(
            emif_spec["parameters"]["fill"],
            wvalues_per_buf,
            wbuf_len*wbuf_count,
            projection["stream_info"]["W"],
            waddr,
            wbuf_len
            )
        ibuf = utils.read_out_stored_values_from_array(
            emif_spec["parameters"]["fill"],
            ivalues_per_buf,
            ibuf_len*ibuf_count,
            projection["stream_info"]["I"],
            iaddr,
            ibuf_len
            )
        
    # Generate the statemachine
    utils.print_heading("Generating pyMTL model of network and statemachine",1)
    t = state_machine_classes.StateMachineEMIF(mlb_spec, wb_spec, ab_spec,
                                               ab_spec, emif_spec, projection,
                                               w_address=0, i_address=iaddr, o_address=oaddr)
    
    # Start simulation and wait for done to be asserted
    t.elaborate()
    t.apply(DefaultPassGroup())
    
    utils.print_heading("Begin cycle accurate simulation",2)
    t.sim_reset()
    t.sm_start @= 1
    t.sim_tick()
    t.sm_start @= 0
    for i in range(2000):
        if (t.done):
            print("Simulation complete (done asserted)")
            break
        t.sim_tick()
    t.sim_tick()
    assert(t.done)
    
    for i in range(20): # Just make sure that the EMIF finishes reading in data...
        t.sim_tick()

    emif_vals = utils.read_out_stored_values_from_emif(t.emif_inst.sim_model.buf,
                                                 ovalues_per_buf,
                                                 min(obuf_len,ibuf_len)*obuf_count,
                                                 projection["stream_info"]["I"],
                                                 oaddr)
    if (write_to_file):
        with open("final_offchip_data_contents.yaml", 'w') as file:
            yaml.dump(emif_vals, file, default_flow_style=False)

    # Check that the outputs are right!
    obuf = [[[0 for i in range(ovalues_per_buf)]
         for i in range(obuf_len)]
         for j in range (obuf_count)]
    obuf = utils.get_expected_outputs(obuf, ovalues_per_buf,
        wbuf,
        ibuf, ivalues_per_buf, projection)
    
    if (validate_output):
        utils.print_heading("Comparing final off-chip buffer contents with expected results",3)
        print("Expected " + str(obuf))
        print("Actual " + str(emif_vals))
        for bufi in range(obuf_count):
            for olen in range(min(obuf_len,ibuf_len)-1):
                assert obuf[bufi][olen] == emif_vals[bufi*min(obuf_len,ibuf_len) + olen]  
        print("Simulation outputs were correct.")
            
    return emif_vals, t
