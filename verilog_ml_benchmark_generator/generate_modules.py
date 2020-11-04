"""<later>."""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import module_classes
from pymtl3 import *
import inspect
from pymtl3.examples.ex00_quickstart import FullAdder
from pymtl3.passes.backends.verilog import *
import fileinput
import re
from jsonschema import validate

# Schemas to validate input yamls
supported_activations=["RELU"]
datatype_mlb = ["I","O","W","C","CLK","RESET"]
datatype_buffer = ["ADDRESS", "DATAIN", "DATAOUT"]
datatype_any = datatype_mlb+datatype_buffer
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
        "simulation_model" : {"type":"string", "enum":["MLB","Buffer"]},
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
    component.set_metadata(VerilogTranslationPass.explicit_file_name, module_name+"_pymtl.v")
    component.set_metadata(VerilogTranslationPass.explicit_module_name, module_name)
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
            found_hardmod = re.search(r"module " + sim_block, line_list[lineidx])
            if found_hardmod:
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
            found_nep_def = re.search(r"\.*"+nep+r"\s*;*,*\)*$", line_list[lineidx])
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
        foundio = re.findall(r"(in|out)put\s+wire\s+\[\d+:\d+\]\s+\S+\s*,*",origline)
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
    line_list = remove_sim_block_defs(line_list, ["HWB_Sim__spec_"]) 
    line_list = remove_non_existant_ports(line_list, non_existant_ports)
    line_list = remove_parameter_references(line_list)         
    filedata =  '\n'.join(line_list)
    
    # replace HW block component names with actual names
    filedata = re.sub(r"(HWB_Sim__spec_)(\S*)(__inner_proj_\S*)(\s+)(.*)", r"\2\4\5", filedata)
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
    :param ib_spec: Hardware definition of input buffer
    :type ib_spec: dict
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



    
