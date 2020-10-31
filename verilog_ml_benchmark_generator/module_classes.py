from pymtl3 import *
import warnings
from pymtl3.passes.backends.verilog import *
from pymtl3.passes.backends.yosys import *
from utils import *

def ConnectInToTop(s, curr_inst, port, newname):           
    newinport = InPort(port._dsl.Type)
    setattr(s, newname, newinport)
    port //= newinport
def ConnectOutToTop(s, curr_inst, port, newname):           
    newoutport = OutPort(port._dsl.Type)
    setattr(s, newname, newoutport)
    newoutport //= port
def AddNInputs(s, n, width, prefix):  
    for i in range(n):
        newin = InPort(width)
        setattr(s, prefix+str(i), newin)
def AddNOutputs(s, n, width, prefix):  
    for i in range(n):
        newout = OutPort(width)
        setattr(s, prefix+str(i), newout)  
                    
class Datapath(Component):
    def construct(s, mlb_spec={}, wb_spec={}, ib_spec={}, ob_spec={}, proj_spec={}):
        MAC_datatypes = ['W','I','O']
        buffer_specs = {'W':wb_spec, 'I':ib_spec, 'O':ob_spec}
        
        # Calculate required MLB interface widths
        inner_proj = proj_spec['inner_projection']
        MAC_count = get_mlb_count(inner_proj)
        inner_bus_counts = {dtype: get_proj_stream_count(inner_proj,dtype) for dtype in MAC_datatypes}
        inner_data_widths = {dtype: proj_spec['stream_info'][dtype]['width'] for dtype in MAC_datatypes}
        inner_bus_widths= {dtype: inner_bus_counts[dtype]*inner_data_widths[dtype] for dtype in MAC_datatypes}
 
        # Check that this configuration is supported by the hardware model
        assert MAC_count <= mlb_spec['MAC_info']['num_units']
        for dtype in MAC_datatypes:
            assert inner_bus_widths[dtype] <= get_sum_datatype_width(mlb_spec, dtype)
            assert inner_data_widths[dtype] <= mlb_spec['MAC_info']['data_widths'][dtype]

        # Calculate required number of MLBs, IO streams, activations
        outer_proj = proj_spec['outer_projection']
        MLB_count = get_mlb_count(outer_proj)
        outer_bus_counts = {dtype: get_proj_stream_count(outer_proj, dtype) for dtype in MAC_datatypes}
        outer_bus_widths = {dtype: outer_bus_counts[dtype]*inner_bus_widths[dtype] for dtype in MAC_datatypes}
        total_bus_counts = {dtype: outer_bus_counts[dtype]*inner_bus_counts[dtype] for dtype in MAC_datatypes}
        buffer_counts = {dtype: get_num_buffers_reqd(buffer_specs[dtype],
                          outer_bus_counts[dtype], outer_bus_widths[dtype]) for dtype in MAC_datatypes}
        
        # Instantiate MLBs, buffers
        s.mlb_modules = HWB_Wrapper(mlb_spec, MLB_count)
        s.weight_modules = HWB_Wrapper(buffer_specs['W'], buffer_counts['W'])
        s.input_act_modules = HWB_Wrapper(buffer_specs['I'], buffer_counts['I'])
        s.output_act_modules = HWB_Wrapper(buffer_specs['O'], buffer_counts['O'])
        s.activation_function_modules = Activation_Wrapper(count=total_bus_counts['O'],
                                                           function=get_activation_function_name(proj_spec),
                                                           input_width=inner_data_widths['O'],
                                                           output_width=inner_data_widths['I'], registered=False)

        # Instantiate interconnects
        s.weight_interconnect = WeightInterconnect(buffer_width=get_sum_datatype_width(buffer_specs['W'],'DATAOUT'),
                                    mlb_width=get_max_datatype_width(mlb_spec, 'W'),
                                    mlb_width_used=outer_bus_widths['W'],
                                    num_buffers=buffer_counts['W'],
                                    num_mlbs=MLB_count,
                                    projection=outer_proj)
        s.input_interconnect = InputInterconnect(buffer_width=get_sum_datatype_width(buffer_specs['I'],'DATAOUT'),
                                    mlb_width=get_max_datatype_width(mlb_spec, 'I'),
                                    mlb_width_used=outer_bus_widths['I'],
                                    num_buffers=buffer_counts['I'],
                                    num_mlbs=MLB_count,
                                    projection=outer_proj)
        s.output_interconnect = OutputInterconnect(buffer_width=get_sum_datatype_width(buffer_specs['O'],'DATAOUT'),
                                    mlb_width=get_max_datatype_width(mlb_spec, 'O'),
                                    mlb_width_used=outer_bus_widths['O'],
                                    num_buffers=buffer_counts['O'],
                                    num_mlbs=MLB_count,
                                    projection=outer_proj)
                
        for port in (s.activation_function_modules.get_input_value_ports() +
                    s.mlb_modules.get_input_value_ports() +
                    s.output_act_modules.get_input_value_ports() +
                    s.input_act_modules.get_input_value_ports() +
                    s.weight_interconnect.get_input_value_ports() +
                    s.output_interconnect.get_input_value_ports() +
                    s.input_interconnect.get_input_value_ports() +
                    s.weight_modules.get_input_value_ports()):
            if port._dsl.my_name not in s.__dict__.keys():
                port //= 0

            
        # Generate input interconnect .. 
           # Consider cascade chains
           # Consider NxN crossbars for convolving inputs for urw chains
           # Consider broadcast inputs
           # TODO Consider cascaded inputs for ue
        # Generate output interconnect .. 
           # Consider cascade chains
           # Consider connections to activations, activations to buffers
           # TODO Consider partial sums to buffers
           # TODO Consider partial sums from buffers
        # Connect em up
        # Unit tests
        # Method to switch between interconnects
        # Method to map hw blocks to physical locations and to match up different projections
        # Method to decide which buffers to write to when (a table?)

        
        # Generate weight interconnect ... Connect weights
           # TODO Consider ... broadcast
           # TODO Consider ... cascade
        # Generate input interconnect .. 
           # Consider NxN crossbars for convolving inputs for urw chains
        # Generate output interconnect .. 
           # Consider cascade chains
           # Consider connections to activations, activations to buffers
           # TODO Consider partial sums to buffers
           # TODO Consider partial sums from buffers
        # Method to switch between interconnects
        # Method to map hw blocks to physical locations and to match up different projections
        # Method to decide which buffers to write to when (a table?)

        
        # Look into modelling MLB for simulations
        # Run whole simulations here in python...
        #    Make sure whole thing is runnable
        #    Write golden comparison
        
class RELU(Component):
    def construct(s, input_width=1, output_width=1, registered=False):  
        # Shorten the module name to the provided name.
        s.activation_function_in = InPort(input_width)
        s.activation_function_out = OutPort(output_width)
        s.internal0 = Wire(1)
        s.internal1 = Wire(1)
        s.internal0 //= s.reset
        s.internal1 //= s.clk
        signbit = input_width-1
        
        if registered:
            @update_ff
            def upblk():
                 s.activation_function_out <<= (s.activation_function_in[0:output_width]
                                                if (s.activation_function_in[signbit] == 0) else 0)
        else:
            @update
            def upblk():
                 s.activation_function_out @= (s.activation_function_in[0:output_width]
                                               if (s.activation_function_in[signbit] == 0) else 0)

class WeightInterconnect(Component):
    def construct(s, buffer_width=1, mlb_width=1, mlb_width_used=1, num_buffers=1, num_mlbs=1,
                  projection={}):        
        outs_per_in = math.floor(buffer_width/mlb_width_used)
        assert num_mlbs <= outs_per_in*num_buffers
        AddNInputs(s, num_buffers, buffer_width, "inputs_from_buffer_")

        outidx=0
        for ug in range(projection['UG']['value']):
            for ue in range(projection['UE']['value']):
                for ub in range(projection['UB']['value']):
                    for urn in range(projection['URN']['value']):
                        for urw in range(projection['URW']['value']):
                            out_idx=get_overall_idx(projection,{'URW':urw,'URN':urn,'UB':ub,'UE':ue,'UG':ug})
                            
                            # Create an output to send to this MLB
                            newout = OutPort(mlb_width) # Add output to MLB
                            setattr(s, "output_to_mlb_"+str(out_idx), newout)
                            newin = InPort(mlb_width) # Add input from MLB
                            setattr(s, "input_from_mlb_"+str(out_idx), newin)
                            internalw = Wire(mlb_width) # tie it off in case
                            setattr(s, "internal_"+str(out_idx), internalw)
                            internalw //= newin
                            w_idx=get_overall_idx(projection,{'URW':urw,'URN':urn,'UE':ue,'UG':ug})
        
                            # Figure out which input to connect it to
                            input_bus_idx=math.floor(w_idx/outs_per_in)
                            input_bus = getattr(s, "inputs_from_buffer_"+str(input_bus_idx))
                            input_bus_start=(w_idx%outs_per_in)*mlb_width_used
                            input_bus_end=((w_idx%outs_per_in)+1)*mlb_width_used
                            connect(newout[0:mlb_width_used], input_bus[input_bus_start:input_bus_end])

        for i in range(outidx+1, num_mlbs):
            newout = OutPort(mlb_width)
            setattr(s, "output_bus_"+str(out_idx), newout)
            newout //= 0

class OutputInterconnect(Component):
    def construct(s, mlb_width=1, mlb_width_used=1, num_afs=1, num_mlbs=1,
                  projection={}):        
        outs_per_in = math.floor(mlb_width/mlb_width_used)

        # Add inputs from buffers
        AddNOutputs(s, num_afs, mlb_width_used, "outputs_to_afs_")
        
        # Add input and output ports from each MLB
        outidx=0
        for ug in range(projection['UG']['value']):
            for ue in range(projection['UE']['value']):
                for ub in range(projection['UB']['value']):
                    for urn in range(projection['URN']['value']):
                        for urw in range(projection['URW']['value']):
                            out_idx=get_overall_idx(projection,{'URW':urw,'URN':urn,'UB':ub,'UE':ue,'UG':ug})
                            newout = OutPort(mlb_width) # Add output to MLB
                            setattr(s, "output_to_mlb_"+str(out_idx), newout)
                            newin = InPort(mlb_width) # Add input from MLB
                            setattr(s, "input_from_mlb_"+str(out_idx), newin)
                            internalw = Wire(mlb_width) # tie it off in case
                            setattr(s, "internal_"+str(out_idx), internalw)
                            internalw //= newin
                            
                            ### Connect adjacent inputs
                            if (urw > 0):
                                out_idx_prev=get_overall_idx(projection,{'URW':urw-1,'URN':urn,'UB':ub,'UE':ue,'UG':ug})
                                prev_input = getattr(s, "input_from_mlb_"+str(out_idx_prev))
                                connect(newout[0:mlb_width_used], prev_input[0:mlb_width_used])
                            else:
                                # Figure out which input to connect it to
                                i_idx=get_overall_idx(projection,{'URN':urn,'UB':ub,'UG':ug})
                                input_bus_idx=math.floor(i_idx/outs_per_in)
                                input_bus = getattr(s, "inputs_from_buffer_"+str(input_bus_idx))
                                input_bus_start=(i_idx%outs_per_in)*mlb_width_used
                                input_bus_end=((i_idx%outs_per_in)+1)*mlb_width_used
                                connect(newout[0:mlb_width_used], input_bus[input_bus_start:input_bus_end])
                        
        for i in range(out_idx+1, num_mlbs):
            newout = OutPort(mlb_width)
            setattr(s, "output_to_mlb_"+str(i), newout)
            newout //= 0
  
class InputInterconnect(Component):
    def construct(s, buffer_width=1, mlb_width=1, mlb_width_used=1, num_buffers=1, num_mlbs=1,
                  projection={}):        
        outs_per_in = math.floor(buffer_width/mlb_width_used)
        assert num_mlbs <= outs_per_in*num_buffers

        # Add inputs from buffers
        AddNInputs(s, num_buffers, buffer_width, "inputs_from_buffer_")
        
        # Add input and output ports from each MLB
        outidx=0
        for ug in range(projection['UG']['value']):
            for ue in range(projection['UE']['value']):
                for ub in range(projection['UB']['value']):
                    for urn in range(projection['URN']['value']):
                        for urw in range(projection['URW']['value']):
                            out_idx=get_overall_idx(projection,{'URW':urw,'URN':urn,'UB':ub,'UE':ue,'UG':ug})
                            newout = OutPort(mlb_width) # Add output to MLB
                            setattr(s, "output_to_mlb_"+str(out_idx), newout)
                            newin = InPort(mlb_width) # Add input from MLB
                            setattr(s, "input_from_mlb_"+str(out_idx), newin)
                            internalw = Wire(mlb_width) # tie it off in case
                            setattr(s, "internal_"+str(out_idx), internalw)
                            internalw //= newin
                            
                            ### Connect adjacent inputs
                            if (urw > 0):
                                out_idx_prev=get_overall_idx(projection,{'URW':urw-1,'URN':urn,'UB':ub,'UE':ue,'UG':ug})
                                prev_input = getattr(s, "input_from_mlb_"+str(out_idx_prev))
                                connect(newout[0:mlb_width_used], prev_input[0:mlb_width_used])
                            else:
                                # Figure out which input to connect it to
                                i_idx=get_overall_idx(projection,{'URN':urn,'UB':ub,'UG':ug})
                                input_bus_idx=math.floor(i_idx/outs_per_in)
                                input_bus = getattr(s, "inputs_from_buffer_"+str(input_bus_idx))
                                input_bus_start=(i_idx%outs_per_in)*mlb_width_used
                                input_bus_end=((i_idx%outs_per_in)+1)*mlb_width_used
                                connect(newout[0:mlb_width_used], input_bus[input_bus_start:input_bus_end])
                        

        for i in range(out_idx+1, num_mlbs):
            newout = OutPort(mlb_width)
            setattr(s, "output_to_mlb_"+str(i), newout)
            newout //= 0
            
class Activation_Wrapper(Component):
    def construct(s, count=1, function="RELU", input_width=1, output_width=1, registered=False):       
        for i in range(count):
            assert (function == "RELU"), "NON-RELU functions not currently implemented."
            curr_inst = RELU(input_width, output_width, registered)
            setattr(s, function+'_inst_' + str(i), curr_inst)
            
            for port in curr_inst.get_input_value_ports():
                if port._dsl.my_name not in s.__dict__.keys():
                    ConnectInToTop(s, curr_inst, port, port._dsl.my_name+"_"+str(i))
            for port in curr_inst.get_output_value_ports():
                if port._dsl.my_name not in s.__dict__.keys():
                    ConnectOutToTop(s, curr_inst, port, port._dsl.my_name+"_"+str(i))
        
class HWB(Component):
    def construct(s, spec={}):
        for port in spec['ports']:
            if not port["type"] in ("CLK", "RESET"): 
                if (port["direction"] == "in"):
                    setattr(s, port["name"], InPort(int(port["width"])))
                else:
                    newout = OutPort(port["width"])
                    setattr(s, port["name"], newout)
                    newout //= newout._dsl.Type(0)  
        s._dsl.args = [spec['block_name']] # Shorten the module name to the provided name.
       
class HWB_Wrapper(Component):
    def construct(s, spec={}, count=1):
        # Add ports shared between instances to the top level
        for port in spec['ports']:
            if (port['type'] == 'C'):
                if (port['direction'] == "in"):
                    setattr(s, port['name'], InPort(port['width']))
                else:
                    setattr(s, port['name'], OutPort(port['width']))
        
        for i in range(count):
            curr_inst = HWB(spec)
            setattr(s, spec['block_name']+'_inst_' + str(i), curr_inst)
            for port in spec['ports']:
                if (port['type'] == 'C'):
                    instport = getattr(curr_inst,port["name"])
                    instport //=  getattr(s,port["name"])
                elif port['type'] not in ('CLK', 'RESET'):
                    if (port['direction'] == "in"):
                        ConnectInToTop(s, curr_inst, getattr(curr_inst,port["name"]), port["name"]+"_"+str(i))
                    else:
                        ConnectOutToTop(s, curr_inst, getattr(curr_inst,port["name"]), port["name"]+"_"+str(i))
                        
        s._dsl.args = [spec['block_name']] # Shorten the module name to the provided name.
