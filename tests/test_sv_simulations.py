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
        mlb_file, ab_file, wb_file, emif_file, proj_file, ws=True, v=False, mod_name="test_sv"):

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
        mem_lines = mem_lines + [str(val)]
    with open(mod_name + ".mem", 'w') as file:
        file.write('\n'.join(mem_lines))

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
                                                    include_sim_models=True)
    print("done simulating")
    # Check that EMIFs have the right data
    emif_vals = utils.read_out_stored_values_from_emif(
        testinst.emif_inst.sim_model.bufi, wvalues_per_buf, iaddr,
        proj_yaml["data_widths"]["W"], 0)
    print(emif_vals)
    print(wbuf)
    for k in range(len(wbuf)):
        for j in range(len(wbuf[k])):
            for i in range(len(wbuf[k][j])):
                assert emif_vals[k*len(wbuf[k])+j][i] == wbuf[k][j][i]
                
    emif_vals = utils.read_out_stored_values_from_emif(
        testinst.emif_inst.sim_model.bufi, ivalues_per_buf, oaddr-iaddr,
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
    check_buffers(testinst.datapath, testinst.datapath.weight_modules,
                  "ml_block_weights_inst_{}",
                  wbuf, proj_yaml["data_widths"]["W"], testinst)
    check_buffers(testinst.datapath, testinst.datapath.input_act_modules,
                  "ml_block_input_inst_{}",
                  ibuf, proj_yaml["data_widths"]["I"], testinst)
    
    # Check that the right data is in the MLBs
    print("okkkk...")
    print(testinst.datapath.mlb_modules.ml_block_inst_0.curr_inst.sim_model.mac_modules.input_out_0)
    print(testinst.datapath.mlb_modules.ml_block_inst_0.curr_inst.sim_model.mac_modules.sum_out_0)

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
        outvals_yaml = yaml.safe_load(outfile)[0]

    for bufi in range(obuf_count):
        for olen in range(min(obuf_len,ibuf_len)-1):
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
    #print(outtxt)
    assert "Errors: 0" in outtxt
    
    print("\n -- vsim") 
    vsim = os.path.join(VSIM_PATH, "vsim")

    vsim_script = "run -all; mem display -startaddress " + str(oaddr) + \
                  " -endaddress " + str(oaddr + obuf_count*obuf_len) + \
                  " /full_layer_testbench/dut0/emif_inst/sim_model/bufi/data"
    process = subprocess.Popen([vsim, "-c", "-do", vsim_script, "full_layer_testbench"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    outtxt = str(process.stdout.read())
    #print(outtxt)
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
        n = ovalues_per_buf
        m = re.match(r'#\s+[0-9]+: x*[01]*' + '([01]{8})'*n + '$', dump_line)
        if (m):
            curr_word = []
            for val in range(n):
                print(val)
                print(int(m.group(val+1), base=2))
                curr_word = [int(m.group(val+1), base=2)] + curr_word
            verilog_out = verilog_out + [curr_word]

    print("EXPECTED OUT")
    print(obuf)
    print("\nACTUAL OUT")
    print(verilog_out)

    for bufi in range(obuf_count):
        for olen in range(min(obuf_len,ibuf_len)-1):
            print(obuf[bufi][olen])
            assert obuf[bufi][olen] == verilog_out[bufi*obuf_len+olen]

            
@pytest.mark.requiresodin
def test_modelsim_emif_statemachine_mini():
    test_simulate_emif_statemachine_sv("mlb_spec.yaml",
                                "b1_spec.yaml",
                                "b0_spec.yaml",
                                "emif_spec_small.yaml",
                                "projection_spec_3.yaml")

