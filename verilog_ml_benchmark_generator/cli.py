"""Function wrappers accessible from the command line."""
# External imports
import click
import yaml

# Internal imports
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
@click.option('--emif_definition', required=True,
              help='YAML definition of external memory interface',
              type=click.File('rb'))
@click.option('--projection_definition', required=False,
              help='YAML definition of some projection',
              type=click.File('rb'))
def generate_accelerator_verilog(module_name, mlb_definition,
                                 act_buffer_definition,
                                 weight_buffer_definition, emif_definition,
                                 projection_definition):
    """Generate an activation function module -
       Currently only RELU is supported"""
    mlb_yaml = yaml.safe_load(mlb_definition)
    wb_yaml = yaml.safe_load(weight_buffer_definition)
    ab_yaml = yaml.safe_load(act_buffer_definition)
    proj_yaml = yaml.safe_load(projection_definition)
    emif_yaml = yaml.safe_load(emif_definition)
    generate_modules.generate_accelerator_given_mapping(module_name, mlb_yaml,
                                                        wb_yaml, ab_yaml,
                                                        proj_yaml,
                                                        write_to_file=True,
                                                        emif_spec=emif_yaml)
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
@click.option('--input_address', required=False, default=0,
              help='Address of inputs in off chip data')
@click.option('--weight_address', required=False, default=0,
              help='Address of weights in off chip data')
@click.option('--output_address', required=False, default=0,
              help='Address of outputs in off chip data')
def simulate_accelerator_with_random_input(
        module_name, mlb_definition, act_buffer_definition,
        weight_buffer_definition, emif_definition, projection_definition,
        input_address, weight_address, output_address):
    """Generate an activation function module -
       Currently only RELU is supported"""
    mlb_yaml = yaml.safe_load(mlb_definition)
    wb_yaml = yaml.safe_load(weight_buffer_definition)
    ab_yaml = yaml.safe_load(act_buffer_definition)
    proj_yaml = yaml.safe_load(projection_definition)
    emif_yaml = yaml.safe_load(emif_definition)
    generate_modules.simulate_accelerator_with_random_input(
        module_name, mlb_yaml, wb_yaml, ab_yaml, emif_yaml, proj_yaml,
        True, output_address, input_address, weight_address)
    print("Statemachine simulation was successful")
    return 0


cli.add_command(simulate_accelerator_with_random_input)
cli.add_command(generate_accelerator_verilog)

if __name__ == "__main__":
    cli()  # pragma: no cover
