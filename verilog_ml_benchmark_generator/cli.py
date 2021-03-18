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
@click.option('--module_name', default='benchmark_accelerator',
              help='Name of activation function module')
@click.option('--eb_definition', required=True,
              help='YAML definition of Embedded Blocks',
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
@click.option('--mapping_vector_definition', required=False, default=None,
              help='YAML definition of some projection',
              type=click.File('rb'))
@click.option('--layer_definition', required=False, default=None,
              help='YAML definition of CNN layer',
              type=click.File('rb'))
@click.option('--eb_count', required=False, default=0,
              help='Number of Embedded Blocks available',
              type=int)
@click.option('--include_sv_sim_models', required=False, default=False,
              type=bool,
              help='Generate .sv simulation models of all embedded blocks')
@click.option('--input_address', required=False, default=0,
              help='Address of inputs in off chip data')
@click.option('--weight_address', required=False, default=0,
              help='Address of weights in off chip data')
@click.option('--output_address', required=False, default=0,
              help='Address of outputs in off chip data')
def generate_accelerator_verilog(module_name, eb_definition,
                                 act_buffer_definition,
                                 weight_buffer_definition, emif_definition,
                                 mapping_vector_definition,
                                 layer_definition,
                                 eb_count, include_sv_sim_models,
                                 input_address, weight_address,
                                 output_address):
    """Generate an accelerator based on either a specified set of mapping
       vectors, or on a layer specification (in which case appropriate
       mapping vectors will be selected automatically)"""
    mlb_yaml = yaml.safe_load(eb_definition)
    wb_yaml = yaml.safe_load(weight_buffer_definition)
    ab_yaml = yaml.safe_load(act_buffer_definition)
    emif_yaml = yaml.safe_load(emif_definition)
    if (mapping_vector_definition):
        proj_yaml = yaml.safe_load(mapping_vector_definition)
        assert (not layer_definition), "Specify either a projection " + \
            "definition, or layer definition - not both"
        generate_modules.generate_accelerator_given_mapping(
            module_name, mlb_yaml, wb_yaml, ab_yaml, proj_yaml,
            write_to_file=True,
            emif_spec=emif_yaml, fast_gen=not include_sv_sim_models)
    else:
        assert (layer_definition), "Specify either a mapping vector " + \
            "definition, or layer definition"
        assert (eb_count), "The number of embedded blocks available " + \
            "must be specified, and greater than zero"
        layer_yaml = yaml.safe_load(layer_definition)
        generate_modules.generate_accelerator_for_layers(
            module_name, mlb_yaml, wb_yaml, ab_yaml, emif_yaml, eb_count,
            layer_yaml, simulate=False, fast_gen=not include_sv_sim_models,
            preload_o=-1, preload_i=1, iaddr=input_address,
            waddr=weight_address, oaddr=output_address)
    return 0


@click.command()
@click.option('--module_name', default='activation_functions',
              help='Name of activation function module')
@click.option('--eb_definition', required=True,
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
@click.option('--mapping_vector_definition', required=False,
              help='YAML definition of some projection',
              type=click.File('rb'))
@click.option('--input_address', required=False, default=0,
              help='Address of inputs in off chip data')
@click.option('--weight_address', required=False, default=0,
              help='Address of weights in off chip data')
@click.option('--output_address', required=False, default=0,
              help='Address of outputs in off chip data')
def simulate_accelerator_with_random_input(
        module_name, eb_definition, act_buffer_definition,
        weight_buffer_definition, emif_definition, mapping_vector_definition,
        input_address, weight_address, output_address):
    """Generate an activation function module -
       Currently only RELU is supported"""
    mlb_yaml = yaml.safe_load(eb_definition)
    wb_yaml = yaml.safe_load(weight_buffer_definition)
    ab_yaml = yaml.safe_load(act_buffer_definition)
    proj_yaml = yaml.safe_load(mapping_vector_definition)
    emif_yaml = yaml.safe_load(emif_definition)
    generate_modules.simulate_accelerator_with_random_input(
        module_name, mlb_yaml, wb_yaml, ab_yaml, emif_yaml, proj_yaml,
        True, output_address, input_address, weight_address)
    print("Statemachine simulation was successful")
    return 0


@click.command()
@click.option('--module_name', default='activation_functions',
              help='Name of activation function module')
@click.option('--eb_definition', required=True,
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
@click.option('--mapping_vector_definition', required=False,
              help='YAML definition of some projection',
              type=click.File('rb'))
@click.option('--input_address', required=False, default=0,
              help='Address of inputs in off chip data')
@click.option('--weight_address', required=False, default=0,
              help='Address of weights in off chip data')
@click.option('--output_address', required=False, default=0,
              help='Address of outputs in off chip data')
def simulate_accelerator(
        module_name, eb_definition, act_buffer_definition,
        weight_buffer_definition, emif_definition, mapping_vector_definition,
        input_address, weight_address, output_address):
    """Generate an activation function module -
       Currently only RELU is supported"""
    mlb_yaml = yaml.safe_load(eb_definition)
    wb_yaml = yaml.safe_load(weight_buffer_definition)
    ab_yaml = yaml.safe_load(act_buffer_definition)
    proj_yaml = yaml.safe_load(mapping_vector_definition)
    emif_yaml = yaml.safe_load(emif_definition)
    generate_modules.simulate_accelerator(
        module_name, mlb_yaml, wb_yaml, ab_yaml, emif_yaml, proj_yaml,
        True, [output_address], [weight_address], [input_address])
    print("Statemachine simulation was successful")
    return 0


cli.add_command(simulate_accelerator_with_random_input)
cli.add_command(generate_accelerator_verilog)
cli.add_command(simulate_accelerator)

if __name__ == "__main__":
    cli()  # pragma: no cover
