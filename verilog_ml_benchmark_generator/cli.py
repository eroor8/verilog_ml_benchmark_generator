"""Function wrappers accessible from the command line."""
# External imports
import sys
import click
import os
import yaml

# Internal imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import generate_modules


@click.group()
def cli():  # pragma: no cover
    pass


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
def generate_full_datapath(module_name, mlb_definition,
                           act_buffer_definition, weight_buffer_definition,
                           projection_definition):
    """Generate an activation function module -
       Currently only RELU is supported"""
    mlb_yaml = yaml.safe_load(mlb_definition)
    wb_yaml = yaml.safe_load(weight_buffer_definition)
    ab_yaml = yaml.safe_load(act_buffer_definition)
    proj_yaml = yaml.safe_load(projection_definition)
    generate_modules.generate_full_datapath(module_name, mlb_yaml, wb_yaml,
                                            ab_yaml, proj_yaml, True)
    print("Datapath generation was successful")
    return 0


cli.add_command(generate_full_datapath)

if __name__ == "__main__":
    cli()  # pragma: no cover
