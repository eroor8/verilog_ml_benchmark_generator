"""" 
PYMTL Component Classes Implementing different parts of the dataflow
ASSUMPTIONS:
- Weights are always preloaded
- Weight stationary flow
- Inputs don't require muxing

"""


from pymtl3 import *
import warnings
from pymtl3.passes.backends.verilog import *
from pymtl3.passes.backends.yosys import *
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *
import re
             
class RELU(Component):
    """" This class implements a RELU function. It can either be registered
         or unregistered (at compile time) and has a single input and output
         in addition to clk and reset.
         RELU function: fout = (fin > 0)? fin : 0.
         Input and output widths are specified, and either one could be wider. 

         :param activation_function_in: Input port (fin)
         :type activation_function_in: Component class
         :param activation_function_out: Output port (fout)
         :type activation_function_out: Component class
         :param internal_out0/1: Internal port used to connect input to out
         :type internal_out0/1: Component class

    """
    def construct(s, input_width=1, output_width=1, registered=False):
        """ Constructor for RELU 
         :param input_width: Bit-width of ``activation_function_in``
         :type input_width: int
         :param output_width: Bit-width of ``activation_function_out``
         :type output_width: int
         :param registered: Whether to register the RELU output
         :type registered: Boolean
        """
        # Shorten the module name to the provided name.
        s.activation_function_in = InPort(input_width)
        s.activation_function_out = OutPort(output_width)
        tie_off_clk_reset(s)
        min_width = min(input_width,output_width)
        max_width = max(input_width,output_width)
        s.internal_out0 = Wire(min_width)
        if (output_width>min_width):
            s.internal_out1 = Wire(output_width-min_width)
        else:
            s.internal_out1 = Wire(1)
        
        if registered:
            @update_ff
            def upblk0():
                if s.reset:
                    s.internal_out0 <<= 0
                    s.internal_out1 <<= 0
                else:
                    s.internal_out0 <<= (s.activation_function_in[0:min_width]
                                    if (s.activation_function_in[input_width-1] == 0) \
                                    else 0)
        else:
            @update
            def upblk3():
                 s.internal_out0 @= (s.activation_function_in[0:min_width]
                                    if (s.activation_function_in[input_width-1] == 0) \
                                    else 0)
                 if (output_width > min_width):
                     s.internal_out1 @= 0
        s.activation_function_out[0:min_width] //= s.internal_out0
        if (output_width > min_width):
            s.activation_function_out[min_width:max_width] //= s.internal_out1
         
class ActivationWrapper(Component):
    """" This module wraps several instantiations of an activation function.
         It has the same inputs and outputs as the activation function * the 
         number of functions, named <activation_function_port>_<instance>.
         Clock and reset are common. 
    """
    def construct(s, count=1, function="RELU", input_width=1, \
                  output_width=1, registered=False):    
        """ Constructor for ActivationWrapper

         :param count: Number of activation functions to instantiate
         :type count: int
         :param function: Type of activation function (eg "RELU")
         :type function: string
         :param input_width: Bit-width of ``activation_function_in``
         :type input_width: int
         :param output_width: Bit-width of ``activation_function_out``
         :type output_width: int
         :param registered: Whether to register the RELU output
         :type registered: Boolean
        """   
        for i in range(count):
            assert (function == "RELU"), "NON-RELU functions not currently implemented."
            curr_inst = RELU(input_width, output_width, registered)
            setattr(s, function+'_inst_' + str(i), curr_inst)
            
            for port in curr_inst.get_input_value_ports():
                connect_in_to_top(s, port, port._dsl.my_name+"_"+str(i))
            for port in curr_inst.get_output_value_ports():
                connect_out_to_top(s, port, port._dsl.my_name+"_"+str(i))

class MAC_Register(Component):
    """" This module implements a single register.
         :param input_data: Input port
         :type input_data: Component class 
         :param input_sum: Input partial sum
         :type input_sum: Component class 
         :param output_data: Output port
         :type output_data: Component class
         :param output_sum: Ouput partial sum
         :type output_sum: Component class
    """
    def construct(s, reg_width, sum_width=1):
        """ Constructor for register
         :param reg_width: Bit-width of register
         :type reg_width: int
         :param sum_width: Bit-width of register
         :type sum_width: int
        """
        AddOutPort(s,reg_width,"output_data")
        AddInPort(s,reg_width,"input_data")
        AddOutPort(s,sum_width,"output_sum")
        AddInPort(s,sum_width,"input_sum")
        min_width = min(sum_width, reg_width)
        
        s.REG = Wire(reg_width)
        @update_ff
        def upblk0():
            if s.reset:
                s.REG <<= 0
            else:
                s.REG <<= s.input_data
        @update
        def upblk1():
            s.output_sum @= s.input_sum[0:min_width] + s.REG[0:min_width]
        s.output_data //= s.REG
                
class ShiftRegister(Component):
    """" This module implements a shift register of length ``n``
         Also provide sum of stored values.
         :param input_data: Input port
         :type input_data: Component class 
         :param output_data: Output port
         :type output_data: Component class
         :param sum: Output port
         :type sum: int  
    """
    def construct(s, reg_width=1, sum_width=1, length=1):
        """ Constructor for shift register

         :param reg_width: Bit-width of registers 
         :type reg_width: int
         :param sum_width: Bit-width of registers 
         :type sum_width: int
         :param length: Shift register length
         :type length: int
        """
        AddOutPort(s,sum_width,"sum")
        AddOutPort(s,reg_width,"output_data")
        AddInPort(s,reg_width,"input_data")
        
        for shift_reg in range(length):
            newreg = MAC_Register(reg_width, reg_width)
            setattr(s, "SR_"+str(shift_reg), newreg)

        if (length > 0):
            connect(s.SR_0.input_data,s.input_data)
            s.SR_0.input_sum //= 0
            last_reg = getattr(s, "SR_"+str(length-1))
            s.output_data //= last_reg.output_data
            s.sum //= last_reg.output_sum
        else: 
            s.output_data //= s.input_data
            s.sum //= 0
            
        for shift_reg in range(1,length):
            SR = getattr(s, "SR_"+str(shift_reg))
            PREV_SR = getattr(s, "SR_"+str(shift_reg-1))
            connect(SR.input_data,PREV_SR.output_data)  
            connect(SR.input_sum,PREV_SR.output_sum)  

        
class HWB(Component):
    """" This module represents some block, as specified in a json description.
         The point is to provide the interface of the actual hardware block so that
         it can be instantiated in other modules. The contents of this block is 
         empty, and all outputs are tied to zero.
         One port is added for each port listed in the json port list.
         The module will end up being named "HWB__<block_name>" 
    """
    def construct(s, spec={}, inner_proj={}):
        """ Constructor for HWB

         :param spec: Dictionary describing hardware block ports and functionality 
         :type spec: dict
         :param inner_proj: Dictionary describing projection of computations onto ML block
         :type inner_proj: dict
        """   
        # If this is an ML block, add behavioural info
        dtype_latency={}
        if len(inner_proj) > 0:
            latency_w = get_proj_chain_length(inner_proj,'W')
            latency_i = get_proj_chain_length(inner_proj,'I')
            if "PRELOAD" in inner_proj:
                assert "MAC_info" in spec and "num_units" in spec["MAC_info"]
                for preloaded in inner_proj["PRELOAD"]:
                    if (preloaded["dtype"] == 'W'):
                        latency_w = spec["MAC_info"]["num_units"]/preloaded['bus_count']
                    elif (preloaded["dtype"] == 'I'):
                        latency_i = spec["MAC_info"]["num_units"]/preloaded['bus_count']
                    
            
            # Find corresponding input and output
            win = list(filter(lambda port: (port['type'] == 'W') and \
                          (port["direction"] == "in"), spec['ports']))[0]
            wouts = filter(lambda port: (port['type'] == 'W') and \
                          (port["direction"] == "out"), spec['ports'])
            iin = list(filter(lambda port: (port['type'] == 'I') and \
                          (port["direction"] == "in"), spec['ports']))[0]
            iouts = filter(lambda port: (port['type'] == 'I') and \
                          (port["direction"] == "out"), spec['ports'])
            s.SR_W = ShiftRegister(win['width'],win['width'], math.ceil(latency_w))
            s.SR_I = ShiftRegister(iin['width'],iin['width'], math.ceil(latency_i))

            #Connect up
            newin = AddInPort(s,win["width"],win["name"])
            connect(newin, s.SR_W.input_data)
            newin = AddInPort(s,iin["width"],iin["name"])
            connect(newin, s.SR_I.input_data)
            for outp in wouts:
                newout = AddOutPort(s,outp["width"],outp["name"])
                connect(newout,s.SR_W.output_data)
            for outp in iouts:
                newout = AddOutPort(s,outp["width"],outp["name"])
                connect(newout, s.SR_I.output_data)
            
        for port in spec['ports']:
            if not port["type"] in ("CLK", "RESET"): 
                if (port["direction"] == "in"):
                    AddInPort(s, port["width"],port["name"])
                else:
                    if port["name"] not in s.__dict__.keys():
                        newout = AddOutPort(s,port["width"],port["name"])
                        newout //= newout._dsl.Type(0)  
        s._dsl.args = [spec['block_name']] # Shorten the module name to the provided name.

            
            
                
class HWB_Wrapper(Component):
    """" This module wraps several instantiations of some specified block (``spec``).
         Input ports with datatype "C" (config) and "ADDRESS" are shared between 
         all instances. All other ports are duplicated one for each instance on the
         top level, and named <instance_port_name>_<instance>.
         Clock and reset are common. 
    """
    def construct(s, spec={}, count=1, name="_v1", projection={}):
        """ Constructor for HWB_Wrapper

         :param spec: Dictionary describing hardware block ports and functionality 
         :type spec: dict
         :param count: Number of blocks to instantiate 
         :type count: int
         :param name: String appended to resultant module name to avoid name 
                      collision bug.
         :type count: string
        """   
        # Add ports shared between instances to the top level
        for port in spec['ports']:
            if ((port['type'] == 'C' or port['type'] == 'ADDRESS') and
                (port['direction'] == "in")):
                AddInPort(s,port['width'], port["name"])
         
        for i in range(count):
            curr_inst = HWB(spec, projection)
            setattr(s, spec['block_name']+'_inst_' + str(i), curr_inst)
            for port in spec['ports']:
                if ((port['type'] == 'C' or port['type'] == 'ADDRESS')
                    and port["direction"] == "in"):
                    instport = getattr(curr_inst,port["name"])
                    instport //=  getattr(s,port["name"])
                elif port['type'] not in ('CLK', 'RESET'):
                    if (port['direction'] == "in"):
                        connect_in_to_top(s, getattr(curr_inst,port["name"]), \
                                          port["name"]+"_"+str(i))
                    else:
                        connect_out_to_top(s, getattr(curr_inst,port["name"]), \
                                           port["name"]+"_"+str(i))
        tie_off_clk_reset(s)
 
class MergeBusses(Component):
    """" This module connects narrow input busses to wider output busses.
         ``ins_per_out`` busses of width ``in_width`` are merged into output
        busses of width ``out_width``. Unconnected outputs are tied to 0.

         :param input_<i>: Input port
         :type input_<i>: Component class 
         :param output_<i>: Output port
         :type output_<i>: Component class
    """
    def construct(s, in_width=1, num_ins=1, out_width=1, num_outs=1, ins_per_out=0):
        """ Constructor for MergeBusses 

         :param in_width: Bit-width of input ports
         :type in_width: int
         :param num_ins: Number of input ports
         :type num_ins: int
         :param out_width: Bit-width of output ports
         :type out_width: int
         :param num_outs: Number of output ports
         :type num_outs: int
         :param ins_per_out: Number of output ports connecting to each input. 
                             The maximum possible by default.
         :type ins_per_out: int
        """
        if (ins_per_out == 0):
            ins_per_out = math.floor(out_width/in_width)
        assert ins_per_out > 0
        assert ins_per_out*num_outs >= num_ins, "Merge busses: Ins per out: " + \
            str(ins_per_out) + " num_outs:" + str(num_outs) + ", num_ins:" + str(num_ins)
        
        # Add outputs to activation functions
        add_n_inputs(s, num_ins , in_width, "input_")
        add_n_outputs(s, num_outs, out_width, "output_")
        
        # Add input and output ports from each MLB
        for inp in range(num_ins):
            bus_idx = math.floor(inp/ins_per_out)
            bus_start=(inp%ins_per_out)*in_width
            bus_end=((inp%ins_per_out)+1)*in_width
            input_bus = getattr(s, "input_"+str(inp))
            output_bus = getattr(s, "output_"+str(bus_idx))
            connect(input_bus[0:in_width], output_bus[bus_start:bus_end])
            
        for i in range(num_outs):
            output_bus = getattr(s, "output_"+str(i))
            if (i > math.floor(num_ins/ins_per_out)):
                output_bus //= 0
            elif ((ins_per_out*in_width < out_width)):
                output_bus[ins_per_out*in_width:out_width] //= 0
            
        tie_off_clk_reset(s)

class WeightInterconnect(Component):
    """" This module connects the weight ports between the inner instances and
         the buffers.

         1) Connect weight buffers to MLBs
            Assume that entire input bus of each MLB should connect to the same
            buffer to simplify things. One buffer can connect to many MLBs though
            if the output is wide enough.

         TODO: Allow for directly connecting weights between instances
         TODO: Allow for preloading all weights from a single buffer

         :param inputs_from_buffer_<i>: Input port from weight buffer for i from 0
                                        to ``num_buffers``
         :type inputs_from_buffer_<i>: Component class
         :param inputs_from_mlb_<i>: Input port from MLB (currently disconnected)
         :type inputs_from_mlb_<i>: Component class
         :param outputs_to_mlb_<i>: Output port to MLB
         :type outputs_to_mlb_<i>: Component class
    """
    def construct(s, buffer_width=1, mlb_width=1, mlb_width_used=1, \
                  num_buffers=1, num_mlbs=1, projection={}):        
        """ Constructor for WeightInterconnect 

         :param buffer_width: Bit-width of buffer datain/dataout ports
         :type buffer_width: int
         :param mlb_width: Total bit-width of the weight ports of the inner instances.
         :type mlb_width: int
         :param mlb_width_used: Bit-width of the weight stream connecting to each inner
                           instance for this projection.
         :type mlb_width_used: int
         :param num_buffers: Total number of weight buffers
         :type num_buffers: int
         :param num_mlbs: Total number of inner instances 
         :type num_mlbs: int
         :param projection: Projection specification
         :type projection: dict
        """
        # Validate inputs
        streams_per_buffer = math.floor(buffer_width/mlb_width_used)
        assert mlb_width_used <= mlb_width
        assert streams_per_buffer > 0, "Insufficiently wide input buffer"
        assert num_mlbs >= get_var_product(projection, ['UG','UE','UB','URN','URW']), \
               "Insufficient number of MLBs"
        assert num_buffers >= math.ceil(\
                    get_var_product(projection, ['UG','UE','URN','URW'])/ \
                    streams_per_buffer) , \
               "Insufficient number of weight buffers"

        # Add inputs from buffers
        add_n_inputs(s, num_buffers, buffer_width, "inputs_from_buffer_")

        for ug in range(projection['UG']['value']):
            for ue in range(projection['UE']['value']):
                for ub in range(projection['UB']['value']):
                    for urn in range(projection['URN']['value']):
                        for urw in range(projection['URW']['value']):
                            # Get instance number of the MLB
                            out_idx=get_overall_idx(projection, \
                                    {'URW':urw,'URN':urn,'UB':ub,'UE':ue,'UG':ug})
                            
                            # Create ports to and from the MLB
                            newout = AddOutPort(s, mlb_width, \
                                                "outputs_to_mlb_"+str(out_idx))
                            newin = AddInPort(s, mlb_width, \
                                              "inputs_from_mlb_"+str(out_idx))
                            
                            # Connect all MLB weight inputs to buffers
                            stream_idx=get_overall_idx(projection,\
                                    {'URW':urw,'URN':urn,'UE':ue,'UG':ug}) 
                            input_bus_idx=math.floor(stream_idx/streams_per_buffer)
                            input_bus = getattr(s, "inputs_from_buffer_"+ \
                                                str(input_bus_idx))
                            section_idx=stream_idx%streams_per_buffer
                            input_bus_start=section_idx*mlb_width_used
                            input_bus_end=(section_idx+1)*mlb_width_used
                            connect(newout[0:mlb_width_used], \
                                    input_bus[input_bus_start:input_bus_end])

        # Tie disconnected MLBs to 0
        for i in range(num_mlbs):
            if ("outputs_to_mlb_"+str(i) not in s.__dict__.keys()):
                newout = OutPort(mlb_width)
                setattr(s, "outputs_to_mlb_"+str(i), newout)
                newout //= 0
            newin = AddInPort(s, mlb_width, "inputs_from_mlb_"+str(i))
        tie_off_clk_reset(s)
  
class InputInterconnect(Component):
    """" This module connects the input ports between the inner instances and
         the buffers.

         1) Connect input activation buffers to MLBs
            Assume that entire input bus of each MLB chain should connect to the same
            buffer to simplify things. One buffer can connect to many MLBs though
            if the output is wide enough.

         1) Connect MLBs to each other
            Chains of URW MLBs have the same input. Connect the inputs between
            these sets of blocks.

         TODO: Allow for preloading inputs instead of streaming them.
         TODO: deal with crossbars

         :param inputs_from_buffer_<i>: Input port from weight buffer for i from 0
                                        to ``num_buffers``
         :type inputs_from_buffer_<i>: Component class
         :param inputs_from_mlb_<i>: Input port from MLB (currently disconnected)
         :type inputs_from_mlb_<i>: Component class
         :param outputs_to_mlb_<i>: Output port to MLB
         :type outputs_to_mlb_<i>: Component class
    """
    def construct(s, buffer_width=1, mlb_width=1, mlb_width_used=1, num_buffers=1, \
                  num_mlbs=1, projection={}):
        """ Constructor for InputInterconnect 

         :param buffer_width: Bit-width of buffer datain/dataout ports
         :type buffer_width: int
         :param mlb_width: Total bit-width of the weight ports of the inner instances.
         :type mlb_width: int
         :param mlb_width_used: Bit-width of the weight stream connecting to each inner
                           instance for this projection.
         :type mlb_width_used: int
         :param num_buffers: Total number of weight buffers
         :type num_buffers: int
         :param num_mlbs: Total number of inner instances 
         :type num_mlbs: int
         :param projection: Projection specification
         :type projection: dict
        """
        streams_per_buffer = math.floor(buffer_width/mlb_width_used)
        assert mlb_width_used <= mlb_width
        assert streams_per_buffer > 0, "Insufficiently wide input buffer"
        assert num_mlbs >= get_var_product(projection, ['UG','UE','UB','URN','URW']), \
               "Insufficient number of MLBs"
        assert num_buffers >= math.ceil(\
                    get_var_product(projection, ['UG','UB','URN'])/ \
                    streams_per_buffer) , \
               "Insufficient number of input buffers"
        
        # Add inputs from buffers
        add_n_inputs(s, num_buffers, buffer_width, "inputs_from_buffer_")
        
        # Add input and output ports from each MLB
        outidx=0
        for ug in range(projection['UG']['value']):
            for ue in range(projection['UE']['value']):
                for ub in range(projection['UB']['value']):
                    for urn in range(projection['URN']['value']):
                        for urw in range(projection['URW']['value']):
                            mlb_idx=get_overall_idx(projection, \
                                {'URW':urw,'URN':urn,'UB':ub,'UE':ue,'UG':ug})
                            newout = AddOutPort(s, mlb_width, \
                                     "outputs_to_mlb_"+str(mlb_idx))
                            newin = AddInPort(s, mlb_width, \
                                           "inputs_from_mlb_"+str(mlb_idx))
                            
                            ### Connect adjacent inputs
                            if (urw > 0):
                                mlb_idx_prev=get_overall_idx(projection, \
                                    {'URW':urw-1,'URN':urn,'UB':ub,'UE':ue,'UG':ug})
                                prev_input = getattr(s, "inputs_from_mlb_"+ \
                                                     str(mlb_idx_prev))
                                connect(newout[0:mlb_width_used], \
                                        prev_input[0:mlb_width_used])
                            else:
                                # Figure out which input to connect it to
                                stream_idx=get_overall_idx(projection, \
                                                           {'URN':urn,'UB':ub,'UG':ug})
                                input_bus_idx=math.floor(stream_idx/streams_per_buffer)
                                input_bus = getattr(s, "inputs_from_buffer_"+ \
                                                    str(input_bus_idx))
                                section_idx=stream_idx%streams_per_buffer
                                input_bus_start=section_idx*mlb_width_used
                                input_bus_end=(section_idx+1)*mlb_width_used
                                connect(newout[0:mlb_width_used], \
                                        input_bus[input_bus_start:input_bus_end])
                        
        # Tie disconnected MLBs to 0
        for i in range(num_mlbs):
            if ("outputs_to_mlb_"+str(i) not in s.__dict__.keys()):
                newout = OutPort(mlb_width)
                setattr(s, "outputs_to_mlb_"+str(i), newout)
                newout //= 0
            newin = AddInPort(s, mlb_width, "inputs_from_mlb_"+str(i))
        tie_off_clk_reset(s)


class OutputPSInterconnect(Component):
    """" This module connects the output ports between the inner instances and
         the activation functions

         1) Connect MLBs to activation functions
            Split the outputs of chains of MLBs into activations, and connect
            them to the corresponding activation functions.

         1) Connect MLBs to each other
            Chains of URW*URN MLBs have the same output. Connect the outputs between
            these sets of blocks.

         TODO: Allow for weight stationary flow
         TODO: Allow of loading partial sums from a buffer.
         TODO: Send partial sums to a buffer.

         :param outputs_to_afs_<i>: Output ports connecting to activation functions
         :type outputs_to_afs_<i>: Component class
         :param inputs_from_mlb_<i>: Input port from MLB (currently disconnected)
         :type inputs_from_mlb_<i>: Component class
         :param outputs_to_mlb_<i>: Output port to MLB
         :type outputs_to_mlb_<i>: Component class
    """
    def construct(s, af_width=1, mlb_width=1, mlb_width_used=1, num_afs=1, num_mlbs=1,
                  projection={}):       
        """ Constructor for OutputInterconnect 

         :param af_width: Bit-width of activation function input
         :type af_width: int
         :param mlb_width: Total bit-width of the weight ports of the inner instances.
         :type mlb_width: int
         :param mlb_width_used: Bit-width of the weight stream connecting to each inner
                           instance for this projection.
         :type mlb_width_used: int
         :param num_afs: Total number of activation functions available
         :type num_afs: int
         :param num_mlbs: Total number of inner instances 
         :type num_mlbs: int
         :param projection: Projection specification
         :type projection: dict
        """ 
        acts_per_stream = math.floor(mlb_width_used/af_width)
        assert mlb_width_used <= mlb_width
        assert mlb_width_used % af_width == 0, \
            "The activation input width should be a factor of the total output stream width"
        assert acts_per_stream > 0, "Activation function width too wide"
        assert num_mlbs >= get_var_product(projection, ['UG','UE','UB','URN','URW']), \
               "Insufficient number of MLBs"
        assert num_afs >= math.ceil(\
                    get_var_product(projection, ['UG','UB','UE'])* \
                                    acts_per_stream) , \
               "Insufficient number of activation functions"

        # Add outputs to activation functions
        add_n_outputs(s, num_afs, af_width, "outputs_to_afs_")
        
        # Add input and output ports from each MLB
        outidx=0
        for ug in range(projection['UG']['value']):
            for ue in range(projection['UE']['value']):
                for ub in range(projection['UB']['value']):
                    for urn in range(projection['URN']['value']):
                        for urw in range(projection['URW']['value']):
                            mlb_idx=get_overall_idx(projection,\
                                {'URW':urw,'URN':urn,'UB':ub,'UE':ue,'UG':ug})
                            newout = AddOutPort(s,mlb_width, \
                                                "outputs_to_mlb_"+str(mlb_idx))
                            newin = AddInPort(s,mlb_width, \
                                              "inputs_from_mlb_"+str(mlb_idx))
                            
                            ### Connect only the last output of the chain out
                            if ((urw==projection['URW']['value']-1) and \
                                (urn==projection['URN']['value']-1)):
                                # Figure out which output to connect it to
                                stream_idx=get_overall_idx(projection, \
                                                      {'UB':ub,'UE':ue,'UG':ug})
                                output_bus_idx=stream_idx*acts_per_stream
                                for out_part in range(acts_per_stream):
                                    output_bus = getattr(s, "outputs_to_afs_"+ \
                                                         str(output_bus_idx+out_part))
                                    output_bus_start=out_part*af_width
                                    output_bus_end=(out_part+1)*af_width
                                    #output_bus //= 1
                                    connect(output_bus, \
                                            newin[output_bus_start:output_bus_end])
                            if (urw>0) or (urn>0):
                                # Connect the other blocks in the chain together
                                if (urw > 0):
                                    mlb_idx_prev=get_overall_idx(projection, \
                                        {'URW':urw-1,'URN':urn,'UB':ub,'UE':ue,'UG':ug})
                                else:
                                    mlb_idx_prev=get_overall_idx(projection, \
                                        {'URW':projection['URW']['value']-1,
                                         'URN':urn-1,'UB':ub,'UE':ue,'UG':ug})
                                prev_input = getattr(s, "inputs_from_mlb_"+str(mlb_idx_prev))
                                connect(newout[0:mlb_width_used], \
                                        prev_input[0:mlb_width_used])
                            else:
                                newout[0:mlb_width_used] //= 0
                        

        # Tie disconnected MLBs to 0
        for i in range(num_mlbs):
            if ("outputs_to_mlb_"+str(i) not in s.__dict__.keys()):
                newout = OutPort(mlb_width)
                setattr(s, "outputs_to_mlb_"+str(i), newout)
                newout //= 0
            newin = AddInPort(s, mlb_width, "inputs_from_mlb_"+str(i))
        tie_off_clk_reset(s)
        
class Datapath(Component):
    def construct(s, mlb_spec={}, wb_spec={}, ib_spec={}, ob_spec={}, proj_spec={}):
        """ Constructor for RELU """
        MAC_datatypes = ['W','I','O']
        buffer_specs = {'W':wb_spec, 'I':ib_spec, 'O':ob_spec}
        
        # Calculate required MLB interface widths
        inner_proj = proj_spec['inner_projection']
        MAC_count = get_mlb_count(inner_proj)
        inner_bus_counts = {dtype: get_proj_stream_count(inner_proj,dtype) \
                            for dtype in MAC_datatypes}
        inner_data_widths = {dtype: proj_spec['stream_info'][dtype]['width'] \
                             for dtype in MAC_datatypes}
        print(inner_data_widths)
        print(inner_bus_counts)
        inner_bus_widths= {dtype: inner_bus_counts[dtype]*inner_data_widths[dtype] \
                           for dtype in MAC_datatypes}
 
        # Check that this configuration is supported by the hardware model
        assert MAC_count <= mlb_spec['MAC_info']['num_units']
        for dtype in MAC_datatypes:
            assert inner_bus_widths[dtype] <= get_sum_datatype_width(mlb_spec, dtype)
            assert inner_data_widths[dtype] <= mlb_spec['MAC_info']['data_widths'][dtype]

        # Calculate required number of MLBs, IO streams, activations
        outer_proj = proj_spec['outer_projection']
        MLB_count = get_mlb_count(outer_proj)
        outer_bus_counts = {dtype: get_proj_stream_count(outer_proj, dtype) \
                            for dtype in MAC_datatypes}
        outer_bus_widths = {dtype: outer_bus_counts[dtype]*inner_bus_widths[dtype] \
                            for dtype in MAC_datatypes}
        total_bus_counts = {dtype: outer_bus_counts[dtype]*inner_bus_counts[dtype] \
                            for dtype in MAC_datatypes}
        print(outer_bus_widths)
        print(outer_bus_counts)
        buffer_counts = {dtype: get_num_buffers_reqd(buffer_specs[dtype],
                          outer_bus_counts[dtype], inner_bus_widths[dtype]) \
                         for dtype in ['I','W']}
        
        buffer_counts['O'] = get_num_buffers_reqd(buffer_specs['O'],
                          outer_bus_counts['O'], inner_bus_counts['O']*inner_bus_widths['I'] ) 
        
        print(buffer_counts)
        # Instantiate MLBs, buffers
        s.mlb_modules = HWB_Wrapper(mlb_spec, MLB_count, projection=inner_proj)
        s.weight_modules = HWB_Wrapper(buffer_specs['W'], buffer_counts['W'])
        s.input_act_modules = HWB_Wrapper(buffer_specs['I'], buffer_counts['I'])
        s.output_act_modules = HWB_Wrapper(buffer_specs['O'], buffer_counts['O'], name='_v2')
        s.activation_function_modules = ActivationWrapper(count=total_bus_counts['O'],
                                                           function=get_activation_function_name(proj_spec),
                                                           input_width=inner_data_widths['O'],
                                                           output_width=inner_data_widths['I'],
                                                           registered=False)

        print(s.output_act_modules.get_input_value_ports() )
        # Instantiate interconnects
        s.weight_interconnect = WeightInterconnect(buffer_width=get_sum_datatype_width(buffer_specs['W'],'DATAOUT'),
                                    mlb_width=get_sum_datatype_width(mlb_spec, 'W', ["in"]),
                                    mlb_width_used=inner_bus_widths['W'],
                                    num_buffers=buffer_counts['W'],
                                    num_mlbs=MLB_count,
                                    projection=outer_proj)
        s.input_interconnect = InputInterconnect(buffer_width=get_sum_datatype_width(buffer_specs['I'],'DATAOUT'),
                                    mlb_width=get_sum_datatype_width(mlb_spec, 'I', ["in"]),
                                    mlb_width_used=inner_bus_widths['I'],
                                    num_buffers=buffer_counts['I'],
                                    num_mlbs=MLB_count,
                                    projection=outer_proj)
        s.output_ps_interconnect = OutputPSInterconnect(af_width=inner_data_widths['O'],
                                    mlb_width=get_sum_datatype_width(mlb_spec, 'O', ["in"]),
                                    mlb_width_used=inner_bus_widths['O'],
                                    num_afs=total_bus_counts['O'],
                                    num_mlbs=MLB_count,
                                    projection=outer_proj)
        s.output_interconnect = MergeBusses(in_width=inner_data_widths['I'],
                                    num_ins=total_bus_counts['O'],
                                    out_width=get_sum_datatype_width(buffer_specs['O'], 'DATAIN'),
                                    num_outs=buffer_counts['O'])

        # Connect up all module
        connected_ins = []
        for portname in get_ports_of_type(mlb_spec, 'W', ["out"]):
            connected_ins += connect_ports_by_name(s.mlb_modules, portname["name"], s.weight_interconnect, "inputs_from_mlb")
        for portname in get_ports_of_type(mlb_spec, 'W', ["in"]):
            connected_ins +=connect_ports_by_name(s.weight_interconnect, "outputs_to_mlb", s.mlb_modules, portname["name"])
        for portname in get_ports_of_type(buffer_specs['W'], 'DATAOUT', ["out"]):
            connected_ins +=connect_ports_by_name(s.weight_modules, portname["name"], s.weight_interconnect, "inputs_from_buffer")
            
        for portname in get_ports_of_type(mlb_spec, 'I', ["out"]):
            connected_ins +=connect_ports_by_name(s.mlb_modules, portname["name"], s.input_interconnect, "inputs_from_mlb")
        for portname in get_ports_of_type(mlb_spec, 'I', ["in"]):
            connected_ins +=connect_ports_by_name(s.input_interconnect, "outputs_to_mlb", s.mlb_modules, portname["name"])
        for portname in get_ports_of_type(buffer_specs['I'], 'DATAOUT', ["out"]):
            connected_ins +=connect_ports_by_name(s.input_act_modules, portname["name"], s.input_interconnect, "inputs_from_buffer")
   
        for portname in get_ports_of_type(mlb_spec, 'O', ["out"]):
            connected_ins +=connect_ports_by_name(s.mlb_modules, portname["name"], s.output_ps_interconnect, "inputs_from_mlb")
        for portname in get_ports_of_type(mlb_spec, 'O', ["in"]):
            connected_ins +=connect_ports_by_name(s.output_ps_interconnect, "outputs_to_mlb", s.mlb_modules, portname["name"])
        connected_ins +=connect_ports_by_name(s.output_ps_interconnect, "outputs_to_afs", s.activation_function_modules, "activation_function_in")

        connected_ins +=connect_ports_by_name(s.activation_function_modules, "activation_function_out", s.output_interconnect, "input")
        for portname in get_ports_of_type(buffer_specs['O'], 'DATAIN', ["in"]):
            connected_ins +=connect_ports_by_name(s.output_interconnect, "output", s.output_act_modules, portname["name"])

        for ii in get_ports_of_type(buffer_specs['O'], 'DATAOUT', ["out"]):
            print("  ----  " + str(ii))
        for port in s.output_act_modules.get_output_value_ports():
            print(port)
            for dout in get_ports_of_type(buffer_specs['O'], 'DATAOUT', ["out"]):
                if dout["name"] in port._dsl.my_name:
                    connect_out_to_top(s, port, port._dsl.my_name)
        print(s.output_act_modules.get_input_value_ports() )
        for port in (s.activation_function_modules.get_input_value_ports() +
                    s.mlb_modules.get_input_value_ports() +
                    s.output_act_modules.get_input_value_ports() +
                    s.input_act_modules.get_input_value_ports() +
                    s.weight_interconnect.get_input_value_ports() +
                    s.output_ps_interconnect.get_input_value_ports() +
                    s.input_interconnect.get_input_value_ports() +
                    s.weight_modules.get_input_value_ports()):
            if (port._dsl.my_name not in s.__dict__.keys()) and (port not in connected_ins):
                port //= 0

        # Finish cleaning / unit tests
        # Make sure it still gets through odin
        # better memory interfaces
        # Go through todos
        # Method to switch between interconnects
        # Different types of blocks
        # Method to map hw blocks to physical locations and to match up different projections
        # Method to decide which buffers to write to when (a table?)

        # Look into modelling MLB for simulations
        # Run whole simulations here in python...
        #    Make sure whole thing is runnable
        #    Write golden comparison


