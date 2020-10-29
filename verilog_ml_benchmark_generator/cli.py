"""Function wrappers accessible from the command line."""
# External imports
import sys
import click
import os
import yaml
from jsonschema import validate

# Internal imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
import generate_modules

# Global variables
# Schemas to validate input yamls
port_schema = {"type" : "object",
               "properties" : {
                   "name" : {"type" : "string", "minLength":1},
                   "width" : {"type" : "number", "minimum":1},
                   "direction" : {"type" : "string", "enum":["in", "out"]},
                   "type" : {"type":"string", "enum":["I","O","W","C","CLK","RESET"]}
                },
}
hw_spec_schema = {
    "type" : "object",
    "properties" : {
        "block_name" : {"type":"string", "minLength":1},
        "ports" : {"type":"array",
                   "items": port_schema
                  },
        }
}

@click.group()
def cli():
    pass

@click.command()
@click.option('--output_prefix', default='', help='Prefix of files to be generated')
def main(output_prefix):
    """Console script for verilog_ml_benchmark_generator - ..."""
    print("Running verilog_ml_benchmark_generator.cli.main")
    if (len(output_prefix)>0):
        click.echo('Generating file ' + output_prefix + '_dut.v!')
    else:
        click.echo('Generating file dut.v')
    utils.print_hello_util()
    return 0


@click.command()
@click.option('--module_name', default='activation_function',
              help='Name of activation function module')
@click.option('--function', default="RELU", help='Activation function used',
              type=click.Choice(['RELU'],case_sensitive=True))
@click.option('--input_width', default=32, help='Bit-width of input value', type=int)
@click.option('--output_width', default=32, help='Bit-width of output value', type=int)
@click.option('--registered', default=False, help='Register the activation function output',
              type=bool)
def create_activation_function(module_name, function, input_width, output_width, registered):
    """Generate an activation function module - Currently only RELU is supported"""
    generate_modules.generate_activation_function(module_name, function, input_width,
                                                  output_width, registered, True)
    return 0

@click.command()
@click.option('--module_name', default='activation_functions',
              help='Name of activation function module')
@click.option('--mlb_definition', required=True,
              help='YAML definition of ML Blocks',
              type=click.File('rb'))
@click.option('--act_buffer_definition', required=True,
              help='YAML definition of activation buffers',
              type=click.File('rb'))
@click.option('--weight_buffer_definition', required=True,
              help='YAML definition of weight buffers',
              type=click.File('rb'))
@click.option('--projection_definition', required=False,
              help='YAML definition of some projection',
              type=click.File('rb'))
def generate_full_datapath(module_name, mlb_definition, act_buffer_definition,
                          weight_buffer_definition, projection_definition):
    """Generate an activation function module - Currently only RELU is supported"""
    mlb_yaml = yaml.safe_load(mlb_definition)
    try: 
        validate(instance=parsed_yaml, schema=hw_spec_schema)
    except:
        print("Error during validation of MLB definition YAML file:\n", sys.exc_info()[0])
        raise
    wb_yaml = yaml.safe_load(weight_buffer_definition)
    try: 
        validate(instance=wb_yaml, schema=hw_spec_schema)
    except:
        print("Error during validation of weight buffer definition YAML file:\n", sys.exc_info()[0])
        raise
    ab_yaml = yaml.safe_load(activation_buffer_definition)
    try: 
        validate(instance=ab_yaml, schema=hw_spec_schema)
    except:
        print("Error during validation of activation buffer definition YAML file:\n", sys.exc_info()[0])
        raise
    proj_yaml = yaml.safe_load(activation_buffer_definition)
    try: 
        validate(instance=proj_yaml, schema=projection_schema)
    except:
        print("Error during validation of activation buffer definition YAML file:\n", sys.exc_info()[0])
        raise
    generate_modules.generate_full_datapath(module_name, mlb_yaml, wb_yaml, ab_yaml, proj_yaml, True)
    return 0


@click.command()
@click.option('--module_name', default='activation_functions',
              help='Name of activation function module')
@click.option('--function', default="RELU", help='Activation function used',
              type=click.Choice(['RELU'],case_sensitive=True))
@click.option('--activation_count', default=1, help='Number of instantiations', type=int)
@click.option('--input_width', default=32, help='Bit-width of input value', type=int)
@click.option('--output_width', default=32, help='Bit-width of output value', type=int)
@click.option('--registered', default=False, help='Register the activation function output',
              type=bool)
def instantiate_activation_functions(module_name, function, activation_count,
                                     input_width, output_width, registered):
    """Generate an activation function module - Currently only RELU is supported"""
    generate_modules.generate_activation_functions(module_name, function, activation_count,
                                                   input_width, output_width, registered,
                                                   True)
    return 0

@click.command()
@click.option('--module_name', default='HW_block_modules',
              help='Name of module containing instantiations')
@click.option('--block_count', default=1, help='Number of instantiations', type=int)
@click.option('--block_definition', required=True, help='YAML definition of MLB',
              type=click.File('rb'))
def instantiate_block_wrapper(module_name, block_count, block_definition):
    """Generate a module instantiating ML Blocks (a verilog module implementing some subset of the nested loops)"""
    parsed_yaml = yaml.safe_load(block_definition)
    try: 
        validate(instance=parsed_yaml, schema=hw_spec_schema)
    except:
        print("Error during validation of MLB definition YAML file:\n", sys.exc_info()[0])
        raise
    generate_modules.generate_block_wrapper(module_name, block_count, parsed_yaml, True)
    return 0

cli.add_command(main)
cli.add_command(create_activation_function)
cli.add_command(instantiate_block_wrapper)
cli.add_command(instantiate_activation_functions)
cli.add_command(generate_full_datapath)

if __name__ == "__main__":
    cli()
