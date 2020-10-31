"""<later>."""
import module_classes
from pymtl3 import *
import inspect
from pymtl3.examples.ex00_quickstart import FullAdder
from pymtl3.passes.backends.verilog import *
import fileinput
import re

def odinify(filename_in):
    # Post-process generated file so that it is compatible with odin :(
    with open(filename_in, 'r') as file:
        filedata = file.read()
    # Rename logic to wire
    filedata = filedata.replace('logic', 'wire')
    filedata = filedata.replace('always_comb', 'always@(*)')
    filedata = filedata.replace('always_ff', 'always')

    # Rename ML blocks to correct name
    line_list = filedata.splitlines()
    lineidx = 0
    stored_ios = []
    localparams = {}
    while lineidx < len(line_list):
        # Rename ML blocks to correct name
        line_list[lineidx] = re.sub(r"(HWB__spec_)(.*)", r"\2", line_list[lineidx])

        # Keep track of local params
        foundlp = re.search("localparam\s+wire\s+\[\d+:\d+\]\s+(\S+)\s*=\s*(\S+);",line_list[lineidx])
        if foundlp: 
            line_list[lineidx] = "//" + line_list[lineidx]
            localparams[foundlp.group(1)] = foundlp.group(2)
        else:
            for localparam in localparams:
                line_list[lineidx] = re.sub("\d\'\(\s*"+localparam+"\s*\)",localparams[localparam], line_list[lineidx])
        
        # Move inputs out of the module definition
        origline = line_list[lineidx]
        foundio = re.findall("(in|out)put\s+wire\s+\[\d+:\d+\]\s+\S+\s*,*",origline)
        foundmodstart = re.search(";",origline)
        assert len(foundio) < 2
        if len(foundio)>0:
            match = re.search("(in|out)(put)(\s+)(wire)(\s+)(\[\d+:\d+\])(\s+)(\S+)(\s+)(,*)",origline)
            assert match
            line_list[lineidx] = match.group(8) + match.group(10)
            stored_ios.append(match.group(1) + match.group(2) + match.group(3) + match.group(6) + 
                              match.group(7) + match.group(8) + match.group(9) + ';')
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
        
    return '\n'.join(line_list)

def generate_activation_function(module_name, func_type, input_width, output_width, registered, write_to_file):
    """Make an activation function"""
    print("Generating activation function")
    # Generate the outer module containing many MLBs
    if not (func_type == "RELU"):
        printf("Warning: Only RELU is currently supported")
        return
    
    t = module_classes.RELU(input_width, output_width, registered)
    #t.apply(DefaultPassGroup())
    #t.sim_reset()
    #t.input_sum @= -1
    #t.sim_tick()
    #print(t.output_act)
    #assert t.output_act == 0
    return elab_and_write(t, write_to_file, module_name)

def generate_activation_functions(module_name, func_type, count, input_width, output_width, registered, write_to_file):
    """Make an activation function"""
    print("Generating activation function")
    # Generate the outer module containing many MLBs
    if not (func_type == "RELU"):
        printf("Warning: Only RELU is currently supported")
        return
    
    t = module_classes.Activation_Wrapper(count, func_type, input_width, output_width, registered)
    return elab_and_write(t, write_to_file, module_name)
    
def generate_block_wrapper(module_name, hw_count, hw_spec, write_to_file):
    """Funct to generate MLBs."""
    print("Generating modules")
    # Generate the outer module containing many MLBs
    t = module_classes.HWB_Wrapper(hw_spec, hw_count)
    return elab_and_write(t, write_to_file, module_name)

def generate_full_datapath(module_name, mlb_spec, wb_spec, ab_spec, projection, write_to_file):
    """Funct to generate MLBs."""
    print("Generating modules")
    # Generate the outer module containing many MLBs
    t = module_classes.Datapath(mlb_spec, wb_spec, ab_spec, ab_spec, projection)
    return elab_and_write(t, write_to_file, module_name)

def elab_and_write(t, write_to_file, module_name):
    """Funct to generate MLBs."""
    # Generate the outer module containing many MLBs
    t.elaborate()
    t.set_metadata(VerilogTranslationPass.enable, True)
    t.set_metadata(VerilogTranslationPass.explicit_file_name, module_name+"_pymtl.v")
    t.set_metadata(VerilogTranslationPass.explicit_module_name, module_name)
    t.apply(VerilogTranslationPass())

    # Post-process generated file so that it is compatible with odin :(
    outtxt = odinify(module_name+"_pymtl.v")
    if (write_to_file):
        with open(module_name+".v", 'w') as file:
            file.write(outtxt)
    return t


    
