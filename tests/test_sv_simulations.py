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
        mlb_file, ab_file, wb_file, emif_file, proj_file, ws=True, v=False, mod_name="test_sv",
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
        
    obuf = [[[0 for i in range(ovalues_per_buf)]
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
                                                        ws=False,
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
                print()
                assert obuf[bufi][olen] == outvals_yaml[bufi*min(obuf_len,ibuf_len) + olen]
    #assert(0)
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
        print(dump_line)
        n = ovalues_per_buf
        while(n > 0):
            m = re.match(r'#\s+[0-9]+: x*[01]*' + '([01]{8})'*n + '$', dump_line)
            if (m):
                print(dump_line)
                curr_word = [0]*(ovalues_per_buf-n)
                for val in range(n):
                    print("val", val)
                    print(int(m.group(val+1), base=2))
                    curr_word = [int(m.group(val+1), base=2)] + curr_word
                        
                verilog_out = verilog_out + [curr_word]
                break
            n = n - 1

    print(">> EXPECTED OUT")
    print(obuf)
    print("\n>> ACTUAL OUT")
    print(verilog_out)

    for bufi in range(obuf_count):
        for olen in range(min(obuf_len,ibuf_len)-1):
            #print("compare")
            #print(obuf[bufi][olen])
            assert obuf[bufi][olen] == verilog_out[bufi*obuf_len+olen]


@pytest.mark.requiresodin
def test_modelsim_emif_statemachine_mini():
    test_simulate_emif_statemachine_sv("mlb_model.yaml",
                                "buffer_spec_8.yaml",
                                "b0_spec.yaml",
                                "emif_spec_large.yaml",
                                       "projection_spec_3.yaml", simulate_pymtl=False)
