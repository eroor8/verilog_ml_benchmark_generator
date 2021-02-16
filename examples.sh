# Generate accelerator including instantiations of embedded blocks, and
# compile using ODIN
echo "\n\n Example 1: Generate accelerator based on mapping vectors and compile using ODIN"
verilog_ml_benchmark_generator generate-accelerator-verilog --act_buffer_definition ./tests/b1_spec.yaml --weight_buffer_definition ./tests/b0_spec.yaml --emif_definition ./tests/emif_spec.yaml --eb_definition ./tests/mlb_spec.yaml --mapping_vector_definition ./tests/projection_spec.yaml --include_sv_sim_models False
$VTR_FLOW_PATH ./benchmark_accelerator_odin.v  ./tests/test_arch.xml -ending_stage 'abc'

# Generate accelerator including simulation models of embedded blocks,
# and compile using modelsim (equivalently, quartus/vivado should also work).
# Note: for a meaningful simulation, use the testbench in tests/sv_project, or write
# your own testbench.
echo "\n\n Example 2: Generate accelerator based on mapping vectors and compile using Modelsim"
verilog_ml_benchmark_generator generate-accelerator-verilog --act_buffer_definition ./tests/b1_spec.yaml --weight_buffer_definition ./tests/b0_spec.yaml --emif_definition ./tests/emif_spec.yaml --eb_definition ./tests/mlb_spec.yaml --mapping_vector_definition ./tests/projection_spec.yaml --include_sv_sim_models True
$VSIM_PATH/vlib work
$VSIM_PATH/vlog ./benchmark_accelerator_quartus_vivado.sv
$VSIM_PATH/vsim -c -do "run 100ns; quit" benchmark_accelerator

# Generate accelerator and simulate using pymtl
echo "\n\n Example 3: Generate accelerator based on mapping vectors, and simulate using pyMTL, with random inputs"
verilog_ml_benchmark_generator simulate-accelerator-with-random-input --act_buffer_definition ./tests/b1_spec.yaml --weight_buffer_definition ./tests/b0_spec.yaml --emif_definition ./tests/emif_spec.yaml --eb_definition ./tests/mlb_spec.yaml --mapping_vector_definition ./tests/projection_spec.yaml

echo "\n\n Example 4: Generate accelerator based on mapping vectors, and simulate using pyMTL, with specific inputs (all zeros in this case since in emif_spec.yaml fill=[])"
verilog_ml_benchmark_generator simulate-accelerator --act_buffer_definition ./tests/b1_spec.yaml --weight_buffer_definition ./tests/b0_spec.yaml --emif_definition ./tests/emif_spec.yaml --eb_definition ./tests/mlb_spec.yaml --mapping_vector_definition ./tests/projection_spec.yaml

# Generate accelerator including instantiations of embedded blocks, and
# compile using ODIN - this time specify layer dimensions instead of mapping vectors
echo "\n\n Example 5: Generate accelerator based on given layer dimensions and compile using ODIN"
verilog_ml_benchmark_generator generate-accelerator-verilog --act_buffer_definition ./tests/input_spec_intel_8.yaml --weight_buffer_definition ./tests/input_spec_intel_8.yaml --emif_definition ./tests/emif_spec_intel.yaml --eb_definition ./tests/mlb_spec_intel.yaml --include_sv_sim_models False --layer_definition ./tests/layer_spec.yaml --eb_count 988
$VTR_FLOW_PATH ./benchmark_accelerator_odin.v  ./tests/test_arch.xml -ending_stage 'abc'

