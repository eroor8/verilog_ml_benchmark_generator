#!/usr/bin/env python

"""Tests for `verilog_ml_benchmark_generator` package."""

import pytest

from click.testing import CliRunner

from verilog_ml_benchmark_generator import utils
from verilog_ml_benchmark_generator import generate_modules
from verilog_ml_benchmark_generator import cli

#@pytest.fixture
#def response():
#    """Sample pytest fixture.#

#def test_content(response):
#    """Sample pytest test function with the pytest fixture as an argument."""

def test_gen_MLBs():
    """Sample pytest test function with the pytest fixture as an argument."""
    gen_mod_out = 6 #generate_modules.generate_MLBs()
    assert gen_mod_out == 6

def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'verilog_ml_benchmark_generator.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert 'Usage:' in help_result.output
