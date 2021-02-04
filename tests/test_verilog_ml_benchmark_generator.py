#!/usr/bin/env python

"""Tests for `verilog_ml_benchmark_generator` package."""

import pytest
import os

from click.testing import CliRunner
from verilog_ml_benchmark_generator import cli


def test_generate_statemachine():
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
    result = runner.invoke(cli.generate_accelerator_verilog,
                           ['--module_name', "test",
                            '--mlb_definition', mlb_spec,
                            '--act_buffer_definition', ab_spec,
                            '--weight_buffer_definition', wb_spec,
                            '--emif_definition', emif_spec,
                            '--projection_definition', proj])
    assert result.exit_code == 0
    assert 'Final output files' in result.output

def test_simulate_statemachine():
    """Test generate statemachine"""
    runner = CliRunner()
    
    # Check successful output
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mlb_spec_0.yaml")
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
                            '--mlb_definition', mlb_spec,
                            '--act_buffer_definition', ab_spec,
                            '--weight_buffer_definition', wb_spec,
                            '--emif_definition', emif_spec,
                            '--projection_definition', proj])
    print(result.output)
    assert result.exit_code == 0
    assert 'Statemachine simulation was successful' in result.output
