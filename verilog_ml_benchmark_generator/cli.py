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
    if type(proj_yaml) == list:
        generate_modules.generate_full_datapath(module_name, mlb_yaml, wb_yaml,
                                            ab_yaml, proj_yaml, True)
    else:
        generate_modules.generate_full_datapath(module_name, mlb_yaml, wb_yaml,
                                            ab_yaml, [proj_yaml], True)
    print("Datapath generation was successful")
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
@click.option('--emif_definition', required=True,
              help='YAML definition of external memory interface',
              type=click.File('rb'))
@click.option('--projection_definition', required=False,
              help='YAML definition of some projection',
              type=click.File('rb'))
def generate_statemachine(module_name, mlb_definition,
                           act_buffer_definition, weight_buffer_definition,
                           emif_definition,
                           projection_definition):
    """Generate an activation function module -
       Currently only RELU is supported"""
    mlb_yaml = yaml.safe_load(mlb_definition)
    wb_yaml = yaml.safe_load(weight_buffer_definition)
    ab_yaml = yaml.safe_load(act_buffer_definition)
    proj_yaml = yaml.safe_load(projection_definition)
    emif_yaml = yaml.safe_load(emif_definition)
    generate_modules.generate_statemachine(module_name, mlb_yaml, wb_yaml,
                                            ab_yaml, proj_yaml,
                                            write_to_file=True, emif_spec=emif_yaml)
    print("Statemachine generation was successful")
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
@click.option('--emif_definition', required=True,
              help='YAML definition of external memory interface',
              type=click.File('rb'))
@click.option('--projection_definition', required=False,
              help='YAML definition of some projection',
              type=click.File('rb'))
@click.option('--randomize_inputs', required=False, default=True,
              help='Randomize initial EMIF contents')
@click.option('--input_address', required=False, default=0,
              help='Address of inputs in off chip data')
@click.option('--weight_address', required=False, default=0,
              help='Address of weights in off chip data')
@click.option('--output_address', required=False, default=0,
              help='Address of outputs in off chip data')
def simulate_statemachine(module_name, mlb_definition,
                           act_buffer_definition, weight_buffer_definition,
                           emif_definition,
                           projection_definition,
                           randomize_inputs,
                           input_address,
                           weight_address,
                           output_address):
    """Generate an activation function module -
       Currently only RELU is supported"""
    mlb_yaml = yaml.safe_load(mlb_definition)
    wb_yaml = yaml.safe_load(weight_buffer_definition)
    ab_yaml = yaml.safe_load(act_buffer_definition)
    proj_yaml = yaml.safe_load(projection_definition)
    emif_yaml = yaml.safe_load(emif_definition)
    generate_modules.simulate_statemachine(module_name, mlb_yaml, wb_yaml,
                                            ab_yaml, emif_yaml, proj_yaml,
                                            True, randomize_inputs, output_address,
                                            input_address, weight_address)
    print("Statemachine simulation was successful")
    return 0


cli.add_command(generate_full_datapath)
cli.add_command(simulate_statemachine)
cli.add_command(generate_statemachine)

if __name__ == "__main__":
    cli()  # pragma: no cover
