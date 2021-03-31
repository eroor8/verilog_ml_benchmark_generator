import copy
import pytest
import os
import yaml
import sys
import re
import random
import jsonschema
import subprocess
from jsonschema import validate
from verilog_ml_benchmark_generator import utils
from verilog_ml_benchmark_generator import generate_modules

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_helpers import *

VSIM_PATH = os.getenv('VSIM_PATH')
NUMX = 1976
NUMI = 988
HEIGHT = 177
WIDTH = 149



@pytest.mark.full_simulations
@pytest.mark.skip
def test_simulate_emif_statemachine_sv(
        mlb_file, ab_file, wb_file, emif_file, proj_file, ws=False, v=False, mod_name="test_sv",
        simulate_pymtl=True):

    assert VSIM_PATH, "Set environment variable VSIM_PATH to the location of the modelsim executables"
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
    wbuf_len = min(wbuf_len, utils.get_weight_buffer_len(proj_yaml))
    wbuf = [[[random.randint(0,(2**proj_yaml["data_widths"]["W"])-1)
            for k in range(wvalues_per_buf)]    # values per word
            for i in range(wbuf_len)]           # words per buffer
            for j in range(wbuf_count)]         # buffer count
    wbuf_flat = [sum((lambda i: inner[i] * \
                      (2**(i*proj_yaml["data_widths"]["W"])))(i) \
                     for i in range(len(inner))) \
                         for outer in wbuf for inner in outer]
    iaddr = len(wbuf_flat)
    ibuf_len = min(ibuf_len, utils.get_input_buffer_len(proj_yaml))
    ibuf = [[[random.randint(0,(2**proj_yaml["data_widths"]["I"])-1)
             for k in range(ivalues_per_buf)]            # values per word
             for i in range(ibuf_len)]                   # words per buffer
             for j in range (ibuf_count)]                # buffers

    ibuf_flat = [sum((lambda i: inner[i] * \
                (2**(i*proj_yaml["data_widths"]["I"])))(i) \
                     for i in range(len(inner))) \
                          for outer in ibuf for inner in outer]
    emif_data = wbuf_flat + ibuf_flat
    print(wbuf_flat)
    oaddr = len(emif_data)
    
    emif_yaml["fill"] = copy.deepcopy(emif_data)
    mem_lines = []
    for val in emif_data:
        mem_lines = mem_lines + [str(hex(val)[2:])]
    print(emif_data)
    with open("orig_emif_contents.mem", 'w') as file:
        file.write('\n'.join(mem_lines))
        
    obuf = [[['X' for i in range(ovalues_per_buf)]
                 for i in range(obuf_len)]
                 for j in range (obuf_count)]
    obuf = utils.get_expected_outputs(obuf, ovalues_per_buf,
                                    wbuf,
                                    ibuf, ivalues_per_buf,
                                    proj_yaml)
        
 
    outvals, testinst = generate_modules.simulate_accelerator(
            module_name=mod_name, 
                                                        mlb_spec=mlb_yaml,
                                                        wb_spec=wb_yaml,
                                                        ab_spec=ab_yaml,
                                                        emif_spec=emif_yaml,
                                                        projections=proj_yaml,
                                                        write_to_file=True,
                                                        waddrs=[0],
                                                        iaddrs=[iaddr],
                                                        oaddrs=[oaddr],
                                                        ws=ws,
                                                        validate_output=v,
                                                        gen_ver=True,
                                                        #include_sim_models=['emif', 'emif_inner', 'ml_block_weights', 'ml_block_input', 'mlb_model'],
                                                        include_sim_models=['emif', 'mlb_model'],
                                                        simulate=simulate_pymtl)
    if (simulate_pymtl):
        print("done simulating")
        # Check that EMIFs have the right data
        emif_vals = utils.read_out_stored_values_from_emif(
            testinst.emif_inst.sim_model.emif_inner_inst, wvalues_per_buf, iaddr,
            proj_yaml["data_widths"]["W"], 0)
        print(emif_vals)
        print(wbuf)
        for k in range(len(wbuf)):
            for j in range(len(wbuf[k])):
                for i in range(len(wbuf[k][j])):
                    if not (emif_vals[k*len(wbuf[k])+j][i] == wbuf[k][j][i]):
                        print(k, j, i)
                    assert emif_vals[k*len(wbuf[k])+j][i] == wbuf[k][j][i]
                    
        emif_vals = utils.read_out_stored_values_from_emif(
            testinst.emif_inst.sim_model.emif_inner_inst, ivalues_per_buf, oaddr-iaddr,
            proj_yaml["data_widths"]["I"], iaddr)
        print("\n\nCOMPARE")
        print(emif_vals)
        print("WITH")
        print(ibuf)
        for k in range(len(ibuf)):
            for j in range(len(ibuf[k])):
                for i in range(len(ibuf[k][j])):
                    assert emif_vals[k*len(ibuf[k])+j][i] == ibuf[k][j][i]
        
        # Check that the right data got into the on-chip buffers
        print("Check weight buffer")
        check_buffers(testinst.datapath, testinst.datapath.weight_modules,
                      "ml_block_weights_inst_{}",
                      wbuf, proj_yaml["data_widths"]["W"], testinst)
        print("Check input buffer")
        check_buffers(testinst.datapath, testinst.datapath.input_act_modules,
                      "ml_block_input_inst_{}",
                      ibuf, proj_yaml["data_widths"]["I"], testinst)
        
        # Check that the right data is in the MLBs
        #print("Check MLB weights")
        #print(testinst.datapath.mlb_modules.ml_block_inst_0.curr_inst.sim_model.mac_modules.input_out_0)
        #print(testinst.datapath.mlb_modules.ml_block_inst_0.curr_inst.sim_model.mac_modules.sum_out_0)

        print(" -- EXPECTED OUT")
        print(obuf)
        print("\n -- ACTUAL OUT")
        
        with open("final_offchip_data_contents.yaml") as outfile:
            outvals_yaml = yaml.safe_load(outfile)[0]
        print(outvals_yaml)
        
        for bufi in range(obuf_count):
            for olen in range(min(obuf_len,ibuf_len)-1):
                print("Compare with pymtl output")
                print(obuf[bufi][olen])
                assert obuf[bufi][olen] == outvals_yaml[bufi*min(obuf_len,ibuf_len) + olen]

    # Now test with modelsim
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.join(curr_dir, "sv_project")
    print("cp " + mod_name + "_quartus_vivado.sv " + os.path.join(proj_dir, mod_name + "_quartus_vivado.sv"))
    os.system("rm -rf " + os.path.join(proj_dir, "*_quartus_vivado.sv"))
    os.system("cp " + mod_name + "_quartus_vivado.sv " + os.path.join(proj_dir, mod_name + "_quartus_vivado.sv"))
    
    print("\n -- vlib") 
    vlib = os.path.join(VSIM_PATH, "vlib")
    process = subprocess.Popen([vlib, "work"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    outtxt = str(process.stdout.read())
    print(outtxt)
    assert "Error" not in outtxt

    print("\n -- vlog") 
    vlog = os.path.join(VSIM_PATH, "vlog")
    
    process = subprocess.Popen([vlog, os.path.join(proj_dir, "*.sv")],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    outtxt = str(process.stdout.read())
    if ("Errors: 0" not in outtxt):
        print(outtxt)
    assert "Errors: 0" in outtxt
    
    print("\n -- vsim") 
    vsim = os.path.join(VSIM_PATH, "vsim")
    final_addr = oaddr + obuf_count*obuf_len
    #extract_buffer = " /full_layer_testbench/dut0/emif_inst/sim_model/bufi/data"
    vsim_script = "run -all; mem display -startaddress " + str(oaddr) + \
                  " -endaddress " + str(min(final_addr, oaddr+995-1)) + \
                  " /full_layer_testbench/dut0/emif_inst/sim_model/emif_inner_inst/data;"
    n = 1
    while (final_addr >  oaddr+n*995):
        vsim_script = vsim_script + " mem display -startaddress " + str(oaddr+n*995) + \
                  " -endaddress " + str(min(oaddr + obuf_count*obuf_len, oaddr+(n+1)*995-1)) + \
                  " /full_layer_testbench/dut0/emif_inst/sim_model/emif_inner_inst/data;"
        n = n + 1
    print(vsim_script)
    #vsim_script = "run -all"
    process = subprocess.Popen([vsim, "-c", "-do", vsim_script, "full_layer_testbench"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    outtxt = str(process.stdout.read())
    #print(outtxt)
    print("parsing output")
    with open('vsim_out.txt', 'w') as f:
        print(outtxt, file=f)
    assert "Errors: 0" in outtxt
    assert "Starting system verilog simulation" in outtxt
    assert "Done asserted!" in outtxt
    dump_line = outtxt.find(vsim_script)
    assert dump_line > -1
    mem_dump = outtxt[dump_line:].split("\\n")
    with open('memdump.txt', 'w') as f:
        print(mem_dump, file=f)
    verilog_out = []
    for dump_line in mem_dump:
        #print(dump_line)
        n = ovalues_per_buf
        while(n > 0):
            m = re.match(r'#\s+[0-9]+: x*[01]*' + '([01]{8})'*n + '$', dump_line)
            if (m):
                #print(dump_line)
                curr_word = [0]*(ovalues_per_buf-n)
                for val in range(n):
                    #print("val", val)
                    #print(int(m.group(val+1), base=2))
                    curr_word = [int(m.group(val+1), base=2)] + curr_word
                        
                verilog_out = verilog_out + [curr_word]
                break
            n = n - 1

    print(">> EXPECTED OUT")
    print(obuf)
    print(wbuf)
    print(ibuf)
    print("\n>> ACTUAL OUT")
    print(verilog_out)

    emif_idx = 0
    for bufi in range(obuf_count):
        #print("BUF", bufi)
        for olen in range(obuf_len):
            #print("compare")
            #print(obuf[bufi][olen])
            empty = True
            for v in range(len(obuf[bufi][olen])):
                #print(emif_idx, v)
                if not (obuf[bufi][olen][v] == 'X'):
                    #print(obuf[bufi][olen][v], verilog_out[emif_idx][v])
                    assert obuf[bufi][olen][v] == verilog_out[emif_idx][v]
                    empty = False
            if not empty:
                emif_idx = emif_idx + 1


@pytest.mark.requiresodin
def test_modelsim_emif_statemachine_0():
    test_simulate_emif_statemachine_sv("mlb_model.yaml",
                                       "buffer_spec_8.yaml",
                                       "buffer_spec_8.yaml",
                                       "emif_spec_large.yaml",
                                       "projection_spec_3.yaml", simulate_pymtl=False, ws=False)
    
@pytest.mark.requiresodin
def test_modelsim_emif_statemachine_1():
    test_simulate_emif_statemachine_sv("mlb_model.yaml",
                                       "buffer_spec_8.yaml",
                                       "buffer_spec_8.yaml",
                                       "emif_spec_large.yaml",
                                       "projection_spec_3.yaml", simulate_pymtl=False, ws=True)
#    assert(0)

    
@pytest.mark.full_simulations
@pytest.mark.skip
def test_simulate_layer_sv(
        workload, mlb_file, ab_file, wb_file, emif_file, ws=True, v=False, mod_name="test_sv",
        simulate_pymtl=True, pe_count=4):

    assert VSIM_PATH, "Set environment variable VSIM_PATH to the location of the modelsim executables"
    # Make sure that output gets through odin.
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            mlb_file)
    ab_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ab_file)
    wb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            wb_file)
    emif_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            emif_file)
    with open(mlb_spec) as stream:
        mlb_yaml = yaml.safe_load(stream)
    with open(ab_spec) as stream:
        ab_yaml = yaml.safe_load(stream)
    with open(wb_spec) as stream:
        wb_yaml = yaml.safe_load(stream)
    with open(emif_spec) as stream:
        emif_yaml = yaml.safe_load(stream)

    #iaddr = 1024
    #oaddr = 2048
    iaddr=2048
    oaddr=3000
    proj_yaml, sim_out = generate_modules.generate_accelerator_for_layers(
        module_name=mod_name, 
        mlb_spec=mlb_yaml,
        wb_spec=wb_yaml,
        ab_spec=ab_yaml,
        emif_spec=emif_yaml,
        pe_count=pe_count,
        layer=workload,
        waddr=0,
        iaddr=iaddr,
        oaddr=oaddr,
        simulate=False,
        preload_o=-1, #1 if ws else -1,
        preload_i=-1, #1 if ws else -1,
        ws=ws,
        fast_gen=['emif','mlb_model'])
    outvals = sim_out[0]
    testinst = sim_out[1]
    
    # Calculate buffer dimensions info
    wvalues_per_buf, wbuf_len, wbuf_count = utils.get_iw_buffer_dimensions(
        wb_yaml, proj_yaml, 'W')
    ivalues_per_buf, ibuf_len, ibuf_count = utils.get_iw_buffer_dimensions(
        ab_yaml, proj_yaml, 'I')
    ovalues_per_buf, obuf_len, obuf_count = utils.get_obuffer_dimensions(
        ab_yaml, proj_yaml)

    # Figure out the layer dimensions based on the projection...
    layer = {"group": workload["loop_dimensions"].get("G",1),
             "batches": workload["loop_dimensions"]["B"],
             "out_chans": workload["loop_dimensions"]["E"],
             "in_chans": workload["loop_dimensions"]["C"],
             "image_x": workload["loop_dimensions"]["PX"],
             "image_y": workload["loop_dimensions"]["PY"],
             "filter_x": workload["loop_dimensions"]["RX"],
             "filter_y": workload["loop_dimensions"]["RY"],
             "stridex": workload["stride"]["x"],
             "stridey": workload["stride"]["y"],
             "dilx": workload["dilation"]["x"],
             "dily": workload["dilation"]["y"]
    }
    print("==> Layer information:")
    print(layer)
    weights = [[[[[random.randint(1,4) #(2**proj_yaml["data_widths"]["W"])-1)
                   for k in range(layer["filter_x"])]    # x
                   for i in range(layer["filter_y"])]    # y    
                   for j in range(layer["in_chans"])]    # ichans
                   for l in range(layer["out_chans"])]   # ochans
                   for t in range(layer["group"])]       # group
    inputs = [[[[[random.randint(0,5) #(2**proj_yaml["data_widths"]["I"])-1)
                   for k in range(layer["image_x"])]     # x
                   for i in range(layer["image_y"])]     # y    
                   for j in range(layer["in_chans"])]    # chans
                   for l in range(layer["batches"])]     # batch
                   for t in range(layer["group"])]       # group
    layer_type = mlb_yaml.get('MAC_info', {}).get('type', 'MAC')
    layer_outputs = utils.compute_layer(inputs, weights, layer, layer_type,
                                        output_width=proj_yaml["data_widths"]["O"],
                                        final_width=proj_yaml["data_widths"]["I"],
                                        activation_function=proj_yaml["activation_function"])
    layer_outputs = [[[[[layer_outputs[t][l][j][i][k]%(2**proj_yaml["data_widths"]["I"])
                         for k in range(len(layer_outputs[t][l][j][i]))]  # x
                         for i in range(len(layer_outputs[t][l][j]))]      # y    
                         for j in range(len(layer_outputs[t][l]))]       # chans
                         for l in range(len(layer_outputs[t]))]       # batch
                         for t in range(len(layer_outputs))]       # group o%(2**proj_yaml["data_widths"]["I"])
    actual_outputs = [[[[['x'
                     for k in range(len(layer_outputs[t][l][j][i]))]  # x
                     for i in range(len(layer_outputs[t][l][j]))]      # y    
                     for j in range(len(layer_outputs[t][l]))]       # chans
                     for l in range(len(layer_outputs[t]))]       # batch
                     for t in range(len(layer_outputs))]       # group o%(2**proj_yaml["data_widths"]["I"])   
    
    # Move the weights and inputs into the EMIF in the expected order
    wbuf = reorder_weight_array(weights,proj_yaml, wb_yaml)
    ibuf = reorder_input_array(inputs,proj_yaml, ab_yaml, obuf_len)
    print(inputs)
    print(ibuf)

    wbuf_len = min(wbuf_len, utils.get_weight_buffer_len(proj_yaml))
    wbuf_flat = [sum((lambda i: inner[i] * \
                      (2**(i*proj_yaml["data_widths"]["W"])))(i) \
                     for i in range(len(inner))) \
                         for outer in wbuf for inner in outer]
    assert(len(wbuf_flat) <= iaddr)
    if (len(wbuf_flat) < iaddr):
        wbuf_flat = wbuf_flat + [0]*(iaddr-len(wbuf_flat))
    iaddr = len(wbuf_flat)
    ibuf_len = min(ibuf_len, utils.get_input_buffer_len(proj_yaml))
    ibuf_flat = [sum((lambda i: inner[i] * \
                (2**(i*proj_yaml["data_widths"]["I"])))(i) \
                     for i in range(len(inner))) \
                          for outer in ibuf for inner in outer]  
    assert(len(ibuf_flat) <= (oaddr-iaddr))
    emif_data = wbuf_flat + ibuf_flat
    print(wbuf_flat)
    #oaddr = len(emif_data)
    
    emif_yaml["fill"] = copy.deepcopy(emif_data)
    mem_lines = []
    for val in emif_data:
        mem_lines = mem_lines + [str(hex(val)[2:])]
    print(emif_data)
    with open("orig_emif_contents.mem", 'w') as file:
        file.write('\n'.join(mem_lines))
        
    obuf = [[['X' for i in range(ovalues_per_buf)]
                 for i in range(obuf_len)]
                 for j in range (obuf_count)]
    obuf = utils.get_expected_outputs(obuf, ovalues_per_buf,
                                    wbuf,
                                    ibuf, ivalues_per_buf,
                                    proj_yaml)
    #print(obuf)
    #print(wbuf)
    #print(ibuf)
    #assert(0)
    if (simulate_pymtl):
        print("done simulating")
        # Check that EMIFs have the right data
        emif_vals = utils.read_out_stored_values_from_emif(
            testinst.emif_inst.sim_model.emif_inner_inst, wvalues_per_buf, iaddr,
            proj_yaml["data_widths"]["W"], 0)
        print(emif_vals)
        print(wbuf)
        for k in range(len(wbuf)):
            for j in range(len(wbuf[k])):
                for i in range(len(wbuf[k][j])):
                    if not (emif_vals[k*len(wbuf[k])+j][i] == wbuf[k][j][i]):
                        print(k, j, i)
                    assert emif_vals[k*len(wbuf[k])+j][i] == wbuf[k][j][i]
                    
        emif_vals = utils.read_out_stored_values_from_emif(
            testinst.emif_inst.sim_model.emif_inner_inst, ivalues_per_buf, oaddr-iaddr,
            proj_yaml["data_widths"]["I"], iaddr)
        print("\n\nCOMPARE")
        print(emif_vals)
        print("WITH")
        print(ibuf)
        for k in range(len(ibuf)):
            for j in range(len(ibuf[k])):
                for i in range(len(ibuf[k][j])):
                    assert emif_vals[k*len(ibuf[k])+j][i] == ibuf[k][j][i]
        
        # Check that the right data got into the on-chip buffers
        print("Check weight buffer")
        check_buffers(testinst.datapath, testinst.datapath.weight_modules,
                      "ml_block_weights_inst_{}",
                      wbuf, proj_yaml["data_widths"]["W"], testinst)
        print("Check input buffer")
        check_buffers(testinst.datapath, testinst.datapath.input_act_modules,
                      "ml_block_input_inst_{}",
                      ibuf, proj_yaml["data_widths"]["I"], testinst)
    
        print(" -- EXPECTED OUT")
        print(obuf)
        print("\n -- ACTUAL OUT")
        
        with open("final_offchip_data_contents.yaml") as outfile:
            outvals_yaml = yaml.safe_load(outfile)[0]
        print(outvals_yaml)
        
        for bufi in range(obuf_count):
            for olen in range(min(obuf_len,ibuf_len)-1):
                print("Compare with pymtl output")
                print(obuf[bufi][olen])
                print()
                assert obuf[bufi][olen] == outvals_yaml[bufi*min(obuf_len,ibuf_len) + olen]

    # Now test with modelsim
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.join(curr_dir, "sv_project")
    print("cp " + mod_name + "_quartus_vivado.sv " + os.path.join(proj_dir, mod_name + "_quartus_vivado.sv"))
    os.system("rm -rf " + os.path.join(proj_dir, "*_quartus_vivado.sv"))
    os.system("cp " + mod_name + "_quartus_vivado.sv " + os.path.join(proj_dir, mod_name + "_quartus_vivado.sv"))
    
    print("\n -- vlib") 
    vlib = os.path.join(VSIM_PATH, "vlib")
    process = subprocess.Popen([vlib, "work"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    outtxt = str(process.stdout.read())
    print(outtxt)
    assert "Error" not in outtxt

    print("\n -- vlog") 
    vlog = os.path.join(VSIM_PATH, "vlog")
    print([vlog, os.path.join(proj_dir, "*.sv")])
    #assert(0)
    process = subprocess.Popen([vlog, os.path.join(proj_dir, "*.sv")],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    outtxt = str(process.stdout.read())
    if ("Errors: 0" not in outtxt):
        print(outtxt)
    assert "Errors: 0" in outtxt
    
    print("\n -- vsim") 
    vsim = os.path.join(VSIM_PATH, "vsim")
    obuf_len = utils.get_output_buffer_len(proj_yaml)
    print("HEREs the len we were given:", utils.get_output_buffer_len(proj_yaml))
    print(obuf_len)
    print(obuf_count)
    #assert(0)
    final_addr = oaddr + obuf_count*obuf_len
    #extract_buffer = " /full_layer_testbench/dut0/emif_inst/sim_model/bufi/data"
    vsim_script = "run -all; mem display -startaddress " + str(oaddr) + \
                  " -endaddress " + str(min(final_addr, oaddr+995-1)) + \
                  " /full_layer_testbench/dut0/emif_inst/sim_model/emif_inner_inst/data;"
    n = 1
    print(oaddr)
    print(final_addr)
    print(oaddr+n*995)
    while (final_addr >  oaddr+n*995):
        vsim_script = vsim_script + " mem display -startaddress " + str(oaddr+n*995) + \
                  " -endaddress " + str(min(oaddr + obuf_count*obuf_len, oaddr+(n+1)*995-1)) + \
                  " /full_layer_testbench/dut0/emif_inst/sim_model/emif_inner_inst/data;"
        n = n + 1
    print(vsim_script)
    #vsim_script = "run -all"
    process = subprocess.Popen([vsim, "-c", "-do", vsim_script, "full_layer_testbench"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    outtxt = str(process.stdout.read())
    #print(outtxt)
    print("parsing output")
    with open('vsim_out.txt', 'w') as f:
        print(outtxt, file=f)
    print(outtxt)
    assert "Errors: 0" in outtxt
    assert "Starting system verilog simulation" in outtxt
    assert "Done asserted!" in outtxt
    dump_line = outtxt.find(vsim_script)
    assert dump_line > -1
    mem_dump = outtxt[dump_line:].split("\\n")
    with open('memdump.txt', 'w') as f:
        print(mem_dump, file=f)
    verilog_out = []
    for dump_line in mem_dump:
        n = ovalues_per_buf
        while(n > 0):
            m = re.match(r'#\s+[0-9]+: .*[01]*' + '([01]{8})'*n + '$', dump_line)
            if (m):
                curr_word = [0]*(ovalues_per_buf-n)
                for val in range(n):
                    print("val", val)
                    print(int(m.group(val+1), base=2))
                    curr_word = [int(m.group(val+1), base=2)] + curr_word
                        
                verilog_out = verilog_out + [curr_word]
                break
            n = n - 1

    #for kk in range(len(verilog_out), obuf_count*obuf_len):
    #    verilog_out = verilog_out + [verilog_out[-1]]
    print("raw output", verilog_out)
    print("weights", weights)
    print("reordered weights", wbuf)
    print("inputs", inputs)
    print("reordered inputs", ibuf)
    print("expected obuf", layer_outputs)
    actual_outputs = reorder_output_array(verilog_out, proj_yaml, ab_yaml, actual_outputs, ibuf_len)
    print("expected obuf", layer_outputs)
    print("reordered obuf", actual_outputs)
    
    not_all_x = False
    for j in range(len(actual_outputs)):
        for k in range(len(actual_outputs[j])):
            for l in range(len(actual_outputs[j][k])):
                for m in range(len(actual_outputs[j][k][l])):
                    for n in range(len(actual_outputs[j][k][l][m])):
                        if not (actual_outputs[j][k][l][m][n] == 'x'):
                            not_all_x = True
                            assert(actual_outputs[j][k][l][m][n] == layer_outputs[j][k][l][m][n])
    #assert(99==0)

def test_sim_layer_c_e_os():
            
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':128, 
                            'E':32, 'PX':1,
                            'PY':1, 'RX':1,
                            'RY':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained_os.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=False, pe_count=20)
    
def test_sim_layer_c_e_ws():
            
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':128, 
                            'E':32, 'PX':1,
                            'PY':1, 'RX':1,
                            'RY':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=True, pe_count=129)


def test_sim_layer_c_e_b_os():
            
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':16, 'C':32, 
                            'E':16, 'PX':1,
                            'PY':1, 'RX':1,
                            'RY':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained_os.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=False, pe_count=64)

    
def test_sim_layer_c_e_b_ws():
            
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':16, 'C':32, 
                            'E':16, 'PX':1,
                            'PY':1, 'RX':1,
                            'RY':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=True, pe_count=129)


def test_sim_layer_c_px_os():
            
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':16, 
                            'E':1, 'PX':32,
                            'PY':1, 'RX':1,
                            'RY':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained_os.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=False, pe_count=32)

    
def test_sim_layer_c_px_ws():
            
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':32, 
                            'E':1, 'PX':32,
                            'PY':1, 'RX':1,
                            'RY':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=True, pe_count=33)


def test_sim_layer_px_rx_os():
            
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':1, 
                            'E':1, 'PX':64,
                            'PY':1, 'RX':4,
                            'RY':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained_os.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=False, pe_count=17)
    
def test_sim_layer_px_rx_ws():
            
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':1, 
                            'E':1, 'PX':16,
                            'PY':1, 'RX':2,
                            'RY':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=True, pe_count=17)


def test_sim_layer_px_py_os():
            
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':1, 
                            'E':1, 'PX':28,
                            'PY':28, 'RX':1,
                            'RY':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained_os.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=False, pe_count=17)
    
def test_sim_layer_px_py_ws():
            
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':1, 
                            'E':1, 'PX':28,
                            'PY':28, 'RX':1,
                            'RY':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=True, pe_count=17)

def test_sim_layer_px_py_rx_os():
            
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':1, 
                            'E':1, 'PX':28,
                            'PY':28, 'RX':2,
                            'RY':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained_os.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=False, pe_count=17)
    
def test_sim_layer_px_py_rx_ws():
            
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':1, 
                            'E':1, 'PX':28,
                            'PY':28, 'RX':2,
                            'RY':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=True, pe_count=17)
    
def test_sim_layer_py_ry_os():
            
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':1, 
                            'E':1, 'PX':1,
                            'PY':28, 'RX':1,
                            'RY':2},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained_os.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=False, pe_count=4)
    
def test_sim_layer_py_ry_ws():
            
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':1, 
                            'E':1, 'PX':1,
                            'PY':28, 'RX':1,
                            'RY':2},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=True, pe_count=4)
    #assert(0)
    
def test_sim_layer_py_px_ry_ws():
            
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':1, 
                            'E':1, 'PX':16,
                            'PY':28, 'RX':1,
                            'RY':2},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=True, pe_count=16)
    
def test_sim_layer_py_px_ry_os():
            
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':1, 
                            'E':1, 'PX':16,
                            'PY':28, 'RX':1,
                            'RY':2}, 
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained_os.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=False, pe_count=16)
    
def test_sim_layer_py_px_ry_rx_ws():
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':1, 
                            'E':1, 'PX':16,
                            'PY':28, 'RX':2,
                            'RY':2},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=True, pe_count=16)
    #assert(0)
    
def test_sim_layer_py_px_ry_rx_os():
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':1, 'C':1, 
                            'E':1, 'PX':16,
                            'PY':28, 'RX':2,
                            'RY':2},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained_os.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=False, pe_count=16)
    
def test_sim_layer_all_ws():
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':2, 'C':2, 
                            'E':2, 'PX':4,
                            'PY':8, 'RX':2,
                            'RY':2, 'G':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=True, pe_count=8)
    
    
def test_sim_layer_all_ws_funny_dimensions_strides():
    workload = {
        "stride": {"x":2, "y":2},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':2, 'C':2, 
                            'E':2, 'PX':4,
                            'PY':8, 'RX':2,
                            'RY':2, 'G':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=True, pe_count=8)
    
    
def test_sim_layer_all_ws_funny_dimensions_strides2():
    workload = {
        "stride": {"x":3, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':2, 'C':2, 
                            'E':2, 'PX':8,
                            'PY':4, 'RX':2,
                            'RY':2, 'G':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=True, pe_count=8)
    
    
def test_sim_layer_all_ws_funny_dimensions_strides3():
    workload = {
        "stride": {"x":2, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':2, 'C':2, 
                            'E':2, 'PX':8,
                            'PY':8, 'RX':2,
                            'RY':3, 'G':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=True, pe_count=8)
    
    
def test_sim_layer_all_ws_funny_dimensions_strides4():
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":3, "y":3},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':2, 'C':1, 
                            'E':2, 'PX':4,
                            'PY':8, 'RX':2,
                            'RY':3, 'G':1},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=True, pe_count=8)
    #assert(0==5)
    # Limitations:
    #  stridey < ry


def test_sim_layer_act_fn():
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':2, 'C':2, 
                            'E':2, 'PX':4,
                            'PY':8, 'RX':2,
                            'RY':2, 'G':1},
        "activation_function": 'RELU'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=True, pe_count=16)
    

def test_sim_layer_all_os():
    workload = {
        "stride": {"x":1, "y":1},
        "dilation": {"x":1, "y":1},
        "data_widths": {"W":8, "I":8, "O":16},
        "loop_dimensions": {'B':2, 'C':1, 
                            'E':1, 'PX':8,
                            'PY':8, 'RX':2,
                            'RY':2},
        "activation_function": 'NONE'
       }
    test_simulate_layer_sv(workload, "mlb_model_constrained_os.yaml",
                           "buffer_spec_8.yaml",
                           "buffer_spec_8.yaml",
                           "emif_spec_large.yaml", simulate_pymtl=False,
                           ws=False, pe_count=8)
