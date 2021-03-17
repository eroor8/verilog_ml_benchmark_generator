#!/usr/bin/env python

"""Tests for `verilog_ml_benchmark_generator` package."""

import pytest
import os
import subprocess

from click.testing import CliRunner
from verilog_ml_benchmark_generator import cli
VTR_FLOW_PATH = os.getenv('VTR_FLOW_PATH')
VSIM_PATH = os.getenv('VSIM_PATH')

def test_generate_v_odin():
    """Test generate statemachine"""
    runner = CliRunner()
    assert VTR_FLOW_PATH, "Set environment variable VTR_FLOW_PATH to location " + \
        " of VTR flow scripts"
        
    # Check successful output
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mlb_spec.yaml")
    ab_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "b1_spec.yaml")
    wb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "b0_spec.yaml")
    proj = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "projection_spec.yaml")
    emif_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "emif_spec.yaml")
    result = runner.invoke(cli.generate_accelerator_verilog,
                           ['--module_name', "test",
                            '--eb_definition', mlb_spec,
                            '--act_buffer_definition', ab_spec,
                            '--weight_buffer_definition', wb_spec,
                            '--emif_definition', emif_spec,
                            '--mapping_vector_definition', proj])
    assert result.exit_code == 0
    assert 'Final output files' in result.output

    
    outfile = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           ".."), "test_odin.v")
    archfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "test_arch.xml")
    command = [VTR_FLOW_PATH, outfile, archfile,
               "-ending_stage", "abc"]
    print("ODIN command ::" + str(command))
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    assert "OK" in str(process.stdout.read())
    
def test_generate_simulate_sv_accelerator():
    """Test generate statemachine"""
    runner = CliRunner()
    
    assert VSIM_PATH, "Set environment variable VSIM_PATH to the location of the modelsim executables"
    # Check successful output
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mlb_spec.yaml")
    ab_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "b1_spec.yaml")
    wb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "b0_spec.yaml")
    proj = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "projection_spec.yaml")
    emif_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "emif_spec.yaml")
    result = runner.invoke(cli.generate_accelerator_verilog,
                           ['--module_name', "test",
                            '--eb_definition', mlb_spec,
                            '--act_buffer_definition', ab_spec,
                            '--weight_buffer_definition', wb_spec,
                            '--emif_definition', emif_spec,
                            '--mapping_vector_definition', proj,
                            '--include_sv_sim_models', "False"
                            ])
    assert result.exit_code == 0
    assert 'Final output files' in result.output

    mod_name = "test"
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
    if not ("Errors: 0" in outtxt):
        print(outtxt)
    assert "Errors: 0" in outtxt
    
    print("\n -- vsim") 
    vsim = os.path.join(VSIM_PATH, "vsim")

    #vsim_script = "run 100ns; quit"
    vsim_script = "run -all; quit"
    process = subprocess.Popen([vsim, "-c", "-do", vsim_script, "test"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    outtxt = str(process.stdout.read())
    
    with open('vsim_out.txt', 'w') as f:
        print(outtxt, file=f)
    if not ("Errors: 0" in outtxt):
        print(outtxt)
    assert "Errors: 0" in outtxt

def test_simulate_pymtl_accelerator():
    """Test generate statemachine"""
    runner = CliRunner()
    
    # Check successful output
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mlb_spec.yaml")
    ab_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "b1_spec.yaml")
    wb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "b0_spec.yaml")
    proj = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "projection_spec.yaml")
    emif_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "emif_spec.yaml")
    result = runner.invoke(cli.simulate_accelerator,
                           ['--module_name', "test",
                            '--eb_definition', mlb_spec,
                            '--act_buffer_definition', ab_spec,
                            '--weight_buffer_definition', wb_spec,
                            '--emif_definition', emif_spec,
                            '--mapping_vector_definition', proj])
    print(result.output)
    assert result.exit_code == 0
    assert 'Statemachine simulation was successful' in result.output

def test_simulate_random_statemachine():
    """Test generate statemachine"""
    runner = CliRunner()
    
    # Check successful output
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mlb_spec.yaml")
    ab_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "input_spec_0.yaml")
    wb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "weight_spec_0.yaml")
    proj = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "projection_spec_0.yaml")
    emif_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "emif_spec_0.yaml")
    result = runner.invoke(cli.simulate_accelerator_with_random_input,
                           ['--module_name', "test",
                            '--eb_definition', mlb_spec,
                            '--act_buffer_definition', ab_spec,
                            '--weight_buffer_definition', wb_spec,
                            '--emif_definition', emif_spec,
                            '--mapping_vector_definition', proj])
    print(result.output)
    assert result.exit_code == 0
    assert 'Statemachine simulation was successful' in result.output

    
def test_generate_simulate_odin_layer():
    """Test generate statemachine"""
    runner = CliRunner()
    
    assert VSIM_PATH, "Set environment variable VSIM_PATH to the location of the modelsim executables"
    # Check successful output
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mlb_spec_intel.yaml")
    ab_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "input_spec_intel_8.yaml")
    wb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "input_spec_intel_8.yaml")
    layer = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "layer_spec.yaml")
    emif_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "emif_spec_intel.yaml")
    result = runner.invoke(cli.generate_accelerator_verilog,
                           ['--module_name', "test",
                            '--eb_definition', mlb_spec,
                            '--act_buffer_definition', ab_spec,
                            '--weight_buffer_definition', wb_spec,
                            '--emif_definition', emif_spec,
                            '--layer_definition', layer,
                            '--eb_count', 50,
                            '--include_sv_sim_models', "False"
                            ])
    assert result.exit_code == 0
    assert 'Final output files' in result.output

    outfile = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           ".."), "test_odin.v")
    archfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "test_arch_intel_mini.xml")
    command = [VTR_FLOW_PATH, outfile, archfile,
               "-ending_stage", "abc"]
    print("ODIN command ::" + str(command))
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    assert "OK" in str(process.stdout.read())
