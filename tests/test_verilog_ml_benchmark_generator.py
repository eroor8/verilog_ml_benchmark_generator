#!/usr/bin/env python

"""Tests for `verilog_ml_benchmark_generator` package."""

import pytest
import os

from click.testing import CliRunner
from verilog_ml_benchmark_generator import cli

def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()

    # Run without command name from executable
    result = runner.invoke(cli.cli)
    assert result.exit_code == 0

    # Check help output
    help_result = runner.invoke(cli.generate_full_datapath, ['--help'])
    assert help_result.exit_code == 0
    assert 'Usage:' in help_result.output

def test_generate_full_datapath():
    """Test generate full datapath"""
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
    result = runner.invoke(cli.generate_full_datapath,
                           ['--module_name', "test",
                            '--mlb_definition', mlb_spec,
                            '--act_buffer_definition', ab_spec,
                            '--weight_buffer_definition', wb_spec,
                            '--projection_definition', proj])
    assert result.exit_code == 0
    assert 'Datapath generation was successful' in result.output

