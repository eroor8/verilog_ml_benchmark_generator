from pymtl3 import *
import warnings
from pymtl3.passes.backends.verilog import *
from pymtl3.passes.backends.yosys import *

class RELU(Component):
    def construct(s, input_width=1, output_width=1, registered=False):  
        # Shorten the module name to the provided name.
        s.activation_function_in = InPort(input_width)
        #s.mode = InPort(input_width)
        s.activation_function_out = OutPort(output_width)
        s.internal0 = Wire(1)
        s.internal1 = Wire(1)
        s.internal0 //= s.reset
        s.internal1 //= s.clk
        maxbits = 2**(input_width-1)
        
        if registered:
            @update_ff
            def upblk():
                 s.activation_function_out <<= (s.activation_function_in if (s.activation_function_in < maxbits) else 0)
        else:
            @update
            def upblk():
                 s.activation_function_out @= (s.activation_function_in if (s.activation_function_in < maxbits) else 0)

class Datapath(Component):
    def construct(s, mlb_spec={}, wb_spec={}, ab_spec={}, projection={}):
        # Read projection
        # How many MLBs required? Instantiate them
        # How many Weight buffers required? Instantitate them
        # How many activations required? Instantiate them
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

        # Look into modelling MLB for simulations
        # Run whole simulations here in python...
        #    Make sure whole thing is runnable
        #    Write golden comparison

                 
class Activation_Wrapper(Component):
    def construct(s, count=1, function="RELU", input_width=1, output_width=1, registered=False):
        
        for i in range(0,count):
            if not (function == "RELU"):
                print("NON-RELU functions not currently implemented.")
                return
            curr_inst = RELU(input_width, output_width, registered)
            setattr(s, function+'_inst_' + str(i), curr_inst)
            
            for port in curr_inst.get_input_value_ports():
                print (port._dsl.my_name)
                if port._dsl.my_name not in s.__dict__.keys():
                    newinport = InPort(port._dsl.Type)
                    setattr(s, port._dsl.my_name+"_"+str(i), newinport)
                    instport = getattr(curr_inst,port._dsl.my_name )
                    instport //= newinport
            for port in curr_inst.get_output_value_ports():
                print (port._dsl.my_name)
                if port._dsl.my_name not in s.__dict__.keys():
                    newoutport = OutPort(port._dsl.Type)
                    setattr(s, port._dsl.my_name+"_"+str(i), newoutport)
                    newoutport //= getattr(curr_inst,port._dsl.my_name)

                    
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
        
        for i in range(0,count):
            curr_inst = HWB(spec)
            setattr(s, spec['block_name']+'_inst_' + str(i), curr_inst)
            for port in spec['ports']:
                if (port['type'] == 'C'):
                    instport = getattr(curr_inst,port["name"])
                    instport //=  getattr(s,port["name"])
                elif port['type'] not in ('CLK', 'RESET'):
                    if (port['direction'] == "in"):
                        newinport = InPort(port['width'])
                        instport = getattr(curr_inst,port["name"])
                        setattr(s, port['name']+"_"+str(i), newinport)
                        instport //= newinport
                    else:
                        newoutport = OutPort(port['width'])
                        setattr(s, port['name']+"_"+str(i), newoutport)
                        newoutport //= getattr(curr_inst,port["name"])
                        
        s._dsl.args = [spec['block_name']] # Shorten the module name to the provided name.
        
