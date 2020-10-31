from pymtl3 import *
import warnings
from pymtl3.passes.backends.verilog import *
from pymtl3.passes.backends.yosys import *
from utils import *

def ConnectInToTop(s, curr_inst, port, i):           
    newinport = InPort(port._dsl.Type)
    setattr(s, port._dsl.my_name+"_"+str(i), newinport)
    port //= newinport
def ConnectOutToTop(s, curr_inst, port, i):           
    newoutport = OutPort(port._dsl.Type)
    setattr(s, port._dsl.my_name+"_"+str(i), newoutport)
    newoutport //= port
                    
class Datapath(Component):
    def construct(s, mlb_spec={}, wb_spec={}, ib_spec={}, ob_spec={}, proj_spec={}):
        MAC_datatypes = ['W','I','O']
        datatype_buffer_specs = {'W':wb_spec, 'I':ib_spec, 'O':ob_spec}
        
        # Calculate required MLB interface widths
        inner_proj = proj_spec['inner_projection']
        MAC_count = get_mlb_count(inner_proj)
        inner_bus_counts = {dtype: get_proj_stream_count(inner_proj,dtype) for dtype in MAC_datatypes}
        inner_data_widths = {dtype: proj_spec['stream_info'][dtype]['width'] for dtype in MAC_datatypes}
        inner_bus_widths= {dtype: inner_bus_counts[dtype]*inner_data_widths[dtype] for dtype in MAC_datatypes}
 
        # Check that this configuration is supported by the hardware model
        assert MAC_count <= mlb_spec['MAC_info']['num_units']
        for dtype in MAC_datatypes:
            assert inner_bus_widths[dtype] <= get_max_datatype_width(mlb_spec, dtype)
            assert inner_data_widths[dtype] <= mlb_spec['MAC_info']['data_widths'][dtype]

        # Calculate required number of MLBs, IO streams, activations
        outer_proj = proj_spec['outer_projection']
        MLB_count = get_mlb_count(outer_proj)
        outer_bus_counts = {dtype: get_proj_stream_count(outer_proj, dtype) for dtype in MAC_datatypes}
        outer_bus_widths = {dtype: outer_bus_counts[dtype]*inner_bus_widths[dtype] for dtype in MAC_datatypes}
        total_bus_counts = {dtype: outer_bus_counts[dtype]*inner_bus_counts[dtype] for dtype in MAC_datatypes}
        buffer_counts = {dtype: get_num_buffers_reqd(datatype_buffer_specs[dtype],
                          outer_bus_counts[dtype], outer_bus_widths[dtype]) for dtype in MAC_datatypes}
        
        # Instantiate MLBs, buffers
        s.mlb_modules = HWB_Wrapper(mlb_spec, MLB_count)
        s.weight_modules = HWB_Wrapper(datatype_buffer_specs['W'], buffer_counts['W'])
        s.input_act_modules = HWB_Wrapper(datatype_buffer_specs['I'], buffer_counts['I'])
        s.output_act_modules = HWB_Wrapper(datatype_buffer_specs['O'], buffer_counts['O'])
        s.activation_function_modules = Activation_Wrapper(count=total_bus_counts['O'],
                                                           function=get_activation_function_name(proj_spec),
                                                           input_width=inner_data_widths['O'],
                                                           output_width=inner_data_widths['I'], registered=False)

        s.test = SplitBuffer(4, 2)
        
        for port in (s.activation_function_modules.get_input_value_ports() +
                    s.mlb_modules.get_input_value_ports() +
                    s.output_act_modules.get_input_value_ports() +
                    s.input_act_modules.get_input_value_ports() +
                    s.test.get_input_value_ports() +
                    s.weight_modules.get_input_value_ports()):
            if port._dsl.my_name not in s.__dict__.keys():
                port //= 0

                
        # Generate weight interconnect ... Connect weights
           # Consider ... parallel load
               # 1) Separate buffer out to separate busses (wide busses to narrow busses)
               # 2) Connect to weights in order (narrow busses to narrow busses)
           # TODO Consider ... broadcast
           # TODO Consider ... cascade
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
        # Method to switch between interconnects
        # Method to map hw blocks to physical locations and to match up different projections
        # Method to decide which buffers to write to when (a table?)

        
        # Generate weight interconnect ... Connect weights
           # Consider ... parallel load
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
    def construct(s, input_bus_width=1, output_bus_width=1):        
        # Split
                 
class SplitBuffer(Component):
    def construct(s, input_bus_width=1, output_bus_width=1):        
        s.input_bus = InPort(input_bus_width)
        assert output_bus_width > 0
        for i in range(math.floor(input_bus_width/output_bus_width)):
            output_bus = OutPort(output_bus_width)
            setattr(s, "output_bus_"+str(i), output_bus)
            input_start = i*output_bus_width
            input_end = (i+1)*output_bus_width
            connect(output_bus, s.input_bus[input_start:input_end])

              
class Activation_Wrapper(Component):
    def construct(s, count=1, function="RELU", input_width=1, output_width=1, registered=False):       
        for i in range(count):
            assert (function == "RELU"), "NON-RELU functions not currently implemented."
            curr_inst = RELU(input_width, output_width, registered)
            setattr(s, function+'_inst_' + str(i), curr_inst)
            
            for port in curr_inst.get_input_value_ports():
                if port._dsl.my_name not in s.__dict__.keys():
                    ConnectInToTop(s, curr_inst, port, i)
            for port in curr_inst.get_output_value_ports():
                if port._dsl.my_name not in s.__dict__.keys():
                    ConnectOutToTop(s, curr_inst, port, i)
        
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
                        ConnectInToTop(s, curr_inst, getattr(curr_inst,port["name"]), i)
                    else:
                        ConnectOutToTop(s, curr_inst, getattr(curr_inst,port["name"]), i)
                        
        s._dsl.args = [spec['block_name']] # Shorten the module name to the provided name.
