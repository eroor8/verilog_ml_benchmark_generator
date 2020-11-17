""""
PYM-TL Component Classes Implementing different parts of the dataflow
ASSUMPTIONS:
- Weights are always preloaded
- Weight stationary flow
- Inputs don't require muxing

"""
from pymtl3 import *
from pymtl3.passes.backends.verilog import *
from pymtl3.passes.backends.yosys import *
import sys
import math
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
import module_helper_classes
import module_classes
        
class SM_BufferWen(Component):
    def construct(s, buf_count_width, curr_buffer_count):
        
        # Load data into weight buffers
        utils.AddInPort(s, 1, "we_in")
        if (buf_count_width > 0):
            utils.AddInPort(s, buf_count_width, "buffer_count")
            @update
            def upblk_set_wen0():
                if (s.buffer_count == curr_buffer_count):
                    s.we @= s.we_in
                else:
                    s.we @= 0
        else:
            @update
            def upblk_set_wen1():
                s.we @= s.we_in
        utils.AddOutPort(s, 1, "we")
        utils.tie_off_clk_reset(s)
        
class SM_InputSel(Component):
    def construct(s, value_width, buf_count_width, curr_buffer_count):
        
        # Load data into weight buffers
        utils.AddInPort(s, value_width, "vin")
        utils.AddInPort(s, value_width, "cv")
        utils.AddOutPort(s, value_width, "vout")
        if (buf_count_width > 0):
            utils.AddInPort(s, buf_count_width, "buffer_count")
            @update
            def upblk_set_wen0():
                if (s.buffer_count == curr_buffer_count):
                    s.vout @= s.cv
                else:
                    s.vout @= s.vin
        else:
            @update
            def upblk_set_wen1():
                s.vout @= s.cv
        utils.AddOutPort(s, 1, "we") 
        utils.tie_off_clk_reset(s)
              
class SM_LoadBufs(Component):
    def construct(s, buffer_count, write_count):
        start = utils.AddInPort(s,1,"start")
        s.buf_count = Wire(max(int(math.log(buffer_count,2)),1))
        s.buf_address = OutPort(max(int(math.log(write_count,2)),1))
        s.rdy = OutPort(1)
        s.buf_state = Wire(2)
        s.buf_wen = Wire(1)
        utils.add_n_outputs(s, buffer_count, 1, "wen_")
        
        for wb in range(buffer_count):
            new_wen = SM_BufferWen(int(math.log(buffer_count,2)), wb)
            setattr(s, "buf_wen{}".format(wb), new_wen)
            if (buffer_count > 1):
                new_wen.buffer_count //= s.buf_count
            new_wen.we_in //= s.buf_wen
            out_wen = getattr(s, "wen_{}".format(wb))
            out_wen //= new_wen.we        
        INIT, LOAD = 0, 1
        
        @update_ff
        def upblk_set_wen():
            if s.reset:
                s.buf_count <<= 0
                s.buf_wen <<= 0
                s.buf_state <<= INIT
                s.buf_address <<= 0
                s.rdy <<= 1
            else:
                if (s.buf_state == INIT):
                    if (s.start):
                        s.buf_state <<= LOAD
                        s.buf_wen <<= 1
                        s.rdy <<= 0
                elif (s.buf_state == LOAD):
                    if (s.buf_address == (write_count-1)):
                        s.buf_address <<= 0
                        if (s.buf_count == (buffer_count-1)):
                            s.buf_state <<= INIT
                            s.rdy <<= 1
                            s.buf_count <<= 0
                            s.buf_wen <<= 0
                        else:
                            s.buf_state <<= LOAD
                            s.buf_count <<= s.buf_count + 1
                    else:
                        s.buf_address <<= s.buf_address + 1
        utils.tie_off_clk_reset(s)
                        
class SM_WriteOffChip(Component):
    def construct(s, buffer_count, write_count, addr_width, datawidth):
        s.start = InPort(1)
        s.buf_count = Wire(max(int(math.log(buffer_count,2)),1))
        s.address = OutPort(addr_width)
        s.dataout = OutPort(datawidth)
        s.rdy = OutPort(1)
        s.state = Wire(2)
        s.wen = OutPort(1)
        datain_inputs = utils.add_n_inputs(s, buffer_count, datawidth, "datain_")
        datain_inputs = utils.add_n_wires(s, buffer_count, datawidth, "sel_cin")
        
        for wb in range(buffer_count):
            new_sel = SM_InputSel(datawidth, int(math.log(buffer_count,2)), wb)
            setattr(s, "insel{}".format(wb), new_sel)
            if (buffer_count > 1):
                new_sel.buffer_count //= s.buf_count
            new_sel.cv //= getattr(s, "datain_{}".format(wb))
            if (wb == 0):
                new_sel.vin //= 0
            else:
                last_sel = getattr(s, "insel{}".format(wb-1))
                new_sel.vin //= last_sel.vout
            if (wb == buffer_count - 1):
                s.dataout //= new_sel.vout
                
        INIT, LOAD = 0, 1
        @update_ff
        def upblk_set_wen_ff():
            if s.reset:
                s.buf_count <<= 0
                s.wen <<= 0
                s.state <<= INIT
                s.address <<= 0
                s.rdy <<= 1
            else:
                if (s.state == INIT):
                    if (s.start):
                        s.state <<= LOAD
                        s.wen <<= 1
                        s.rdy <<= 0
                elif (s.state == LOAD):
                    if (s.address == (write_count-1)):
                        s.address <<= 0
                        if (s.buf_count == (buffer_count-1)):
                            s.state <<= INIT
                            s.rdy <<= 1
                            s.buf_count <<= 0
                            s.wen <<= 0
                        else:
                            s.state <<= LOAD
                            s.buf_count <<= s.buf_count + 1
                    else:
                        s.address <<= s.address + 1
        utils.tie_off_clk_reset(s)

class SM_PreloadMLBWeights(Component):
    def construct(s, addr_width,
                  num_outer_tiles, outer_tile_size,
                  inner_tile_size,
                  outer_tile_repeat_x, inner_tile_repeat_x):
        w_addr_width = max(int(math.log(outer_tile_size*num_outer_tiles,2)),addr_width)
        s.start = InPort(1)
        s.outer_tile_repeat_count = Wire(max(int(math.log(outer_tile_repeat_x,2)),1))
        s.inner_tile_repeat_count = Wire(max(int(math.log(inner_tile_repeat_x,2)),1))
        s.index_within_inner_tile = Wire(w_addr_width)
        s.inner_tile_index = Wire(w_addr_width)
        max_inner_tile_idx = int(outer_tile_size/inner_tile_size)
        s.outer_tile_index = Wire(w_addr_width)
        s.address = OutPort(addr_width)
        s.addr_w = Wire(w_addr_width)
        s.rdy = OutPort(1)
        s.state = Wire(2)
        s.wen = OutPort(1)
            
        INIT, LOAD = 0, 1
        @update
        def upblk_preload_comb():
            s.addr_w @= s.index_within_inner_tile + s.inner_tile_index*inner_tile_size + \
                         s.outer_tile_index*outer_tile_size
            s.address @= s.addr_w[0:addr_width]
            
        @update_ff
        def upblk_preload_ff():
            if s.reset:
                s.wen <<= 0
                s.state <<= INIT
                s.rdy <<= 1
                s.outer_tile_repeat_count <<= 0
                s.inner_tile_repeat_count <<= 0
                s.index_within_inner_tile <<= 0
                s.inner_tile_index <<= 0
                s.outer_tile_index <<= 0
            else:
                if (s.state == INIT):
                    if (s.start):
                        s.state <<= LOAD
                        s.wen <<= 1
                        s.rdy <<= 0
                elif (s.state == LOAD):
                    if (s.index_within_inner_tile < (inner_tile_size - 1)):
                        s.index_within_inner_tile <<= s.index_within_inner_tile + 1
                    else:
                        s.index_within_inner_tile <<= 0
                        if (s.inner_tile_repeat_count < (inner_tile_repeat_x - 1)):
                            s.inner_tile_repeat_count <<= s.inner_tile_repeat_count + 1
                        else:
                            s.inner_tile_repeat_count <<= 0
                            if (s.inner_tile_index < (max_inner_tile_idx - 1)):
                                s.inner_tile_index <<= s.inner_tile_index + 1
                            else:
                                s.inner_tile_index <<= 0
                                if (s.outer_tile_repeat_count < (outer_tile_repeat_x - 1)):
                                    s.outer_tile_repeat_count <<= \
                                        s.outer_tile_repeat_count + 1
                                else:
                                    s.outer_tile_repeat_count <<= 0
                                    if (s.outer_tile_index < (num_outer_tiles - 1)):
                                        s.outer_tile_index <<= s.outer_tile_index + 1
                                    else:
                                        s.outer_tile_index <<= 0
                                        s.state <<= INIT
                                        s.wen <<= 0
                                        s.rdy <<= 1
        utils.tie_off_clk_reset(s)
                        
class SM_IterateThruAddresses(Component):
    def construct(s, write_count, addr_width, initial_address=0):
        if initial_address < 0:
            abs_init_val = 2**addr_width + initial_address
        else:
            abs_init_val = initial_address
        w_addr_width = max(int(math.ceil(math.log(write_count,2))),addr_width)
        
        s.start = InPort(1)
        s.incr = OutPort(w_addr_width)
        s.address = OutPort(addr_width)
        s.w_address = Wire(w_addr_width)
        s.address_offset = Wire(addr_width)
        s.rdy = OutPort(1)
        s.wen = OutPort(1)
        s.state = Wire(4)

        @update
        def upblk_set_wenc():
            if (s.wen):
                s.w_address @= s.incr + abs_init_val
            else:
                s.w_address @= 0
            s.address @= s.w_address[0:addr_width]
            
        INIT, LOAD = 0, 1
        @update_ff
        def upblk_set_wen():
            if s.reset:
                s.state <<= INIT
                s.incr <<= 0
                s.rdy <<= 1
                s.wen <<= 0
            else:
                if (s.state == INIT):
                    if (s.start):
                        s.state <<= LOAD
                        s.rdy <<= 0
                        s.wen <<= 1
                    s.incr <<= 0
                elif (s.state == LOAD):
                    if (s.incr == (write_count-1)):
                        s.state <<= INIT
                        s.rdy <<= 1
                        s.wen <<= 0
                        s.incr <<= 0
                    else:
                        s.incr <<= s.incr + 1
        utils.tie_off_clk_reset(s)
            
class StateMachine(Component):
    """" This module includes the whole datapath and the state machine controlling it:

         :param datapath: Contains all MLB modules
         :type datapath: Component
         :param weight_buffer_load: Sets control signals for the weight buffers
         :type  weight_buffer_load: Component
         :param input_buffer_load: Sets control signals for the weight buffers
         :type  input_buffer_load: Component
         :param weight_buffer_control: Sets control signals for the weight buffers
         :type weight_buffer_control: Component
    """
    def construct(s, mlb_spec={}, wb_spec={}, ib_spec={}, ob_spec={},
                  proj_spec={}):
        """ Constructor for Datapath

         :param mlb_spec: Contains information about ML block used
         :type mlb_spec: dict
         :param wb_spec: Contains information about weight buffers used
         :type wb_spec: dict
         :param ib_spec: Contains information about input buffers used
         :type ib_spec: dict
         :param ob_spec: Contains information about output buffers used
         :type ob_spec: dict
        """
        print("{:=^60}".format(" Constructing Statemachine with MLB block " +
                               str(mlb_spec.get('block_name', "unnamed") +
                                   " ")))
        MAC_datatypes = ['W', 'I', 'O']
        buffer_specs = {'W': wb_spec, 'I': ib_spec, 'O': ob_spec}

        # Calculate required MLB interface widths and print information
        inner_proj = proj_spec['inner_projection']
        MAC_count = utils.get_mlb_count(inner_proj)
        inner_bus_counts = {dtype: utils.get_proj_stream_count(inner_proj,
                                                               dtype)
                            for dtype in MAC_datatypes}
        inner_data_widths = {dtype: proj_spec['stream_info'][dtype]
                             for dtype in MAC_datatypes}
        inner_bus_widths = {dtype: inner_bus_counts[dtype] *
                            inner_data_widths[dtype]
                            for dtype in MAC_datatypes}
        
        # Calculate required number of MLBs, IO streams, activations
        outer_proj = proj_spec['outer_projection']
        MLB_count = utils.get_mlb_count(outer_proj)
        outer_bus_counts = {dtype: utils.get_proj_stream_count(outer_proj,
                                                               dtype)
                            for dtype in MAC_datatypes}
        outer_bus_widths = {dtype: outer_bus_counts[dtype] *
                            inner_bus_widths[dtype]
                            for dtype in MAC_datatypes}
        total_bus_counts = {dtype: outer_bus_counts[dtype] *
                            inner_bus_counts[dtype]
                            for dtype in MAC_datatypes}
        buffer_counts = {dtype: utils.get_num_buffers_reqd(buffer_specs[dtype],
                         outer_bus_counts[dtype], inner_bus_widths[dtype])
                         for dtype in ['I', 'W']}
        buffer_counts['O'] = utils.get_num_buffers_reqd(buffer_specs['O'],
                                                        outer_bus_counts['O'] *
                                                        inner_bus_counts['O'],
                                                        inner_data_widths['I'])
        connected_ins = []
        # Instantiate MLBs, buffers
        s.datapath = module_classes.Datapath(mlb_spec, wb_spec, ib_spec,
                                             ob_spec, proj_spec)
        
        # Load data into weight buffers
        s.sm_start = InPort(1)
        addrw_ports = list(utils.get_ports_of_type(buffer_specs['W'], 'ADDRESS', ["in"]))
        s.load_wbufs = SM_LoadBufs(buffer_counts["W"], 2**addrw_ports[0]["width"])  
        connected_ins = [s.datapath.weight_modules_portaaddr_top,
                         s.datapath.input_act_modules_portaaddr_top,
                         s.datapath.mlb_modules_a_en_top,
                         s.datapath.mlb_modules_b_en_top,
                         s.datapath.output_act_modules_portaaddr_top
                         ]
        wen_ports = list(utils.get_ports_of_type(buffer_specs['W'], 'WEN', ["in"]))
        connected_ins += utils.connect_ports_by_name(s.load_wbufs,
                                                     r"wen_(\d+)",
                                                     s.datapath,
                                                     r"weight_modules_portawe_(\d+)_top")
        s.weight_address = InPort(addrw_ports[0]["width"])
        
        # Load data into input buffers
        addri_ports = list(utils.get_ports_of_type(buffer_specs['I'], 'ADDRESS', ["in"]))
        s.load_ibufs = SM_LoadBufs(buffer_counts["I"], 2**addri_ports[0]["width"])
        connected_ins += utils.connect_ports_by_name(s.load_ibufs,
                                                     r"wen_(\d+)",
                                                     s.datapath,
                                                     r"input_act_modules_portawe_(\d+)_top")
        
        # Preload weights into MLBs
        s.preload_weights = SM_PreloadMLBWeights(
            addr_width=addrw_ports[0]["width"],
            num_outer_tiles=outer_proj["UG"]["value"],
            outer_tile_size=outer_proj["URN"]["value"]*\
                            outer_proj["URW"]["value"]*\
                            outer_proj["UE"]["value"]*\
                            inner_proj["UG"]["value"]*\
                            inner_proj["URN"]["value"]*\
                            inner_proj["URW"]["value"]*\
                            inner_proj["UE"]["value"],
            inner_tile_size=inner_proj["URN"]["value"]*\
                            inner_proj["URW"]["value"]*\
                            inner_proj["UE"]["value"],
            outer_tile_repeat_x=outer_proj["UB"]["value"],
            inner_tile_repeat_x=inner_proj["UB"]["value"])
        s.external_a_en = InPort(1)
        s.datapath.mlb_modules_a_en_top //= s.preload_weights.wen
        
        ## Stream Inputs into MLB and read outputs into buffer
        addro_ports = list(utils.get_ports_of_type(buffer_specs['O'], 'ADDRESS', ["in"]))
        obuf_len = 2**addro_ports[0]["width"]
        s.stream_inputs = SM_IterateThruAddresses(obuf_len, addri_ports[0]["width"])
        s.stream_outputs = SM_IterateThruAddresses(obuf_len, addri_ports[0]["width"],
                                                   initial_address=-1)
        s.datapath.mlb_modules_b_en_top //= s.stream_inputs.wen
        connected_ins += utils.connect_ports_by_name(s.stream_outputs,
                                                     r"wen",
                                                     s.datapath,
                                                     r"output_act_modules_portawe_(\d+)_top")
        
        # Now read the outputs out to off-chip memory       
        datao_ports = list(utils.get_ports_of_type(buffer_specs['O'], 'DATA', ["out"]))
        s.external_out = OutPort(datao_ports[0]["width"])
        
        s.write_off = SM_WriteOffChip(buffer_counts['O'], 2**addro_ports[0]["width"],
                                      addro_ports[0]["width"], datao_ports[0]["width"])
        connected_ins += utils.connect_ports_by_name(s.datapath,
                                                     r"portadataout_(\d+)",
                                                     s.write_off,
                                                     r"datain_(\d+)")    
        s.external_out //= s.write_off.dataout
        s.state = Wire(5)
        s.done = OutPort(1)
        INIT,LOADING_W_BUFFERS,LOADING_I_BUFFERS,LOADING_MLBS,STREAMING_MLBS, \
            WRITE_OFFCHIP,DONE = Bits5(1),Bits5(2),Bits5(3),Bits5(4),Bits5(5), \
                Bits5(6),Bits5(7)
   
        @update_ff
        def connect_weight_address_ff():
            if s.reset:
                s.state <<= INIT
                s.done <<= 0
            else:
                if (s.state == INIT):
                    if s.sm_start:
                        s.state <<= LOADING_W_BUFFERS
                elif (s.state == LOADING_W_BUFFERS):
                    if s.load_wbufs.rdy:
                        s.state <<= LOADING_I_BUFFERS
                elif (s.state == LOADING_I_BUFFERS):
                    if s.load_ibufs.rdy:
                        s.state <<= LOADING_MLBS
                elif (s.state == LOADING_MLBS):
                    if s.preload_weights.rdy:
                        s.state <<= STREAMING_MLBS
                elif (s.state == STREAMING_MLBS):
                    if s.stream_outputs.rdy:
                        s.state <<= WRITE_OFFCHIP
                elif (s.state == WRITE_OFFCHIP):
                    if s.write_off.rdy:
                        s.state <<= DONE
                        s.done <<= 1
                elif s.state == DONE:
                    s.done <<= 1
                    
        @update
        def connect_weight_address():
            if (s.state == LOADING_MLBS):
                s.datapath.weight_modules_portaaddr_top @= s.preload_weights.address
            else:
                s.datapath.weight_modules_portaaddr_top @= s.load_wbufs.buf_address
            if (s.state == STREAMING_MLBS):
                s.datapath.input_act_modules_portaaddr_top @= s.stream_inputs.address
            else:
                s.datapath.input_act_modules_portaaddr_top @= s.load_ibufs.buf_address
            if (s.state == STREAMING_MLBS):
                s.datapath.output_act_modules_portaaddr_top @= s.stream_outputs.address
            else:
                s.datapath.output_act_modules_portaaddr_top @= s.write_off.address
            s.load_wbufs.start @= (s.state == INIT) & s.sm_start 
            s.load_ibufs.start @= (s.state == LOADING_W_BUFFERS) & s.load_wbufs.rdy 
            s.preload_weights.start @= (s.state == LOADING_I_BUFFERS) & s.load_ibufs.rdy
            s.stream_inputs.start @= (s.state == LOADING_MLBS) & s.preload_weights.rdy
            s.stream_outputs.start @= (s.state == LOADING_MLBS) & s.preload_weights.rdy 
            s.write_off.start @= (s.state == STREAMING_MLBS) & s.stream_outputs.rdy
        print(connected_ins)
        
        # Connect all inputs not otherwise connected to top
        for inst in [s.datapath]:
            for port in (inst.get_input_value_ports()):
                if (port._dsl.my_name not in s.__dict__.keys()) and \
                   (port not in connected_ins):
                    utils.connect_in_to_top(s, port, inst._dsl.my_name + "_" +
                                            port._dsl.my_name + "_sm")
                    print(inst._dsl.my_name + "_" +
                                            port._dsl.my_name + "_sm")
            for port in (inst.get_output_value_ports()):
                if (port._dsl.my_name not in s.__dict__.keys()) and \
                   (port not in connected_ins):
                    utils.connect_out_to_top(s, port, inst._dsl.my_name + "_" +
                                            port._dsl.my_name + "_sm")
                    print(inst._dsl.my_name + "_" +
                                            port._dsl.my_name + "_sm")

        print(s.__dict__)

                      
class SM_LoadBufsEMIF(Component):
    def construct(s, buffer_count, write_count, addr_width, datawidth,
                  emif_addr_width, emif_data_width, startaddr=0):
        assert(emif_addr_width>0)
        assert(emif_data_width>0)
        s.start = InPort(1)
        s.buf_count = Wire(max(int(math.log(buffer_count,2)),1))
        s.buf_address = OutPort(addr_width)
        s.buf_writedata = OutPort(datawidth)
        s.buf_wen = Wire(1)
        utils.add_n_outputs(s, buffer_count, 1, "wen_")
        
        s.emif_address = OutPort(emif_addr_width)
        s.emif_write = OutPort(1)
        s.emif_write //= 0
        s.emif_writedata = OutPort(emif_data_width)
        assert(emif_data_width >= datawidth)
        s.emif_writedata[0:datawidth] //= 0
        s.emif_read = OutPort(1)
        s.emif_readdatavalid = InPort(1)
        s.emif_waitrequest = InPort(1)
        s.emif_readdata = InPort(emif_data_width)
        
        s.rdy = OutPort(1)
        s.state = Wire(2)
        
        for wb in range(buffer_count):
            new_wen = SM_BufferWen(int(math.log(buffer_count,2)), wb)
            setattr(s, "buf_wen{}".format(wb), new_wen)
            if (buffer_count > 1):
                new_wen.buffer_count //= s.buf_count
            new_wen.we_in //= s.buf_wen
            out_wen = getattr(s, "wen_{}".format(wb))
            out_wen //= new_wen.we
                
        INIT, LOAD = 0, 1
        @update_ff
        def upblk_set_wen_ff():
            if s.reset:
                s.buf_count <<= 0
                s.state <<= INIT
                s.buf_address <<= 0
                s.emif_address <<= startaddr
                s.emif_read <<= 0
                s.rdy <<= 1
                s.buf_wen <<= 0
            else:
                #print("***WEN: " + str(s.buf_wen))
                if (s.state == INIT):
                    #print("***INIT: " + str(s.buf_address))
                    s.buf_wen <<= 0
                    if (s.start):
                        s.state <<= LOAD
                        s.rdy <<= 0
                        s.emif_address <<= startaddr
                        s.emif_read <<= 1
                        s.buf_address <<= 0
                        s.buf_count <<= 0
                    else:
                        s.rdy <<= 1
                elif (s.state == LOAD):
                    if (s.emif_waitrequest == 0):
                        if (s.emif_address < (startaddr+write_count*buffer_count-1)):
                            s.emif_address <<= s.emif_address + 1
                        else:
                            s.emif_address <<= 0
                            s.emif_read <<= 0
                    if (s.emif_readdatavalid == 1):
                        s.buf_writedata <<= s.emif_readdata[0:datawidth]
                        s.buf_wen <<= 1
                    else:
                        s.buf_wen <<= 0
                        
                    if (s.buf_wen):
                        if (s.buf_address == (write_count-1)) & \
                           (s.buf_count == (buffer_count-1)):
                            s.state <<= INIT
                        elif (s.buf_address == (write_count-1)) & \
                             (s.buf_count < (buffer_count-1)):
                            s.buf_count <<= s.buf_count + 1
                            s.buf_address <<= 0
                        elif (s.buf_address < (write_count-1)):
                            s.buf_address <<= s.buf_address + 1
                            
        utils.tie_off_clk_reset(s)
                        
class SM_WriteOffChipEMIF(Component):
    def construct(s, buffer_count, write_count, addr_width, datawidth,
                  emif_addr_width, emif_data_width, startaddr=0):
        assert (emif_addr_width>0)
        assert (emif_data_width>0)
        s.start = InPort(1)
        s.buf_count = Wire(max(int(math.log(buffer_count,2)),1))
        s.address = OutPort(addr_width)
        s.bufdata = Wire(datawidth)
        
        s.emif_address = OutPort(emif_addr_width)
        s.emif_write = OutPort(1)
        s.emif_writedata = OutPort(emif_data_width)
        s.emif_writedata[0:datawidth] //= s.bufdata
        s.emif_read = OutPort(1)
        s.emif_read //= 0
        s.emif_waitrequest = InPort(1)
        
        s.rdy = OutPort(1)
        s.state = Wire(2)
        datain_inputs = utils.add_n_inputs(s, buffer_count, datawidth, "datain_")
        datain_inputs = utils.add_n_wires(s, buffer_count, datawidth, "sel_cin")
        
        for wb in range(buffer_count):
            new_sel = SM_InputSel(datawidth, int(math.log(buffer_count,2)), wb)
            setattr(s, "insel{}".format(wb), new_sel)
            if (buffer_count > 1):
                new_sel.buffer_count //= s.buf_count
            new_sel.cv //= getattr(s, "datain_{}".format(wb))
            if (wb == 0):
                new_sel.vin //= 0
            else:
                last_sel = getattr(s, "insel{}".format(wb-1))
                new_sel.vin //= last_sel.vout
            if (wb == buffer_count - 1):
                s.bufdata //= new_sel.vout
                
        INIT, LOAD = 0, 1
        @update_ff
        def upblk_set_wen_ff():
            if s.reset:
                s.buf_count <<= 0
                s.state <<= INIT
                s.address <<= 0
                s.emif_address <<= startaddr
                s.emif_write <<= 0
                s.rdy <<= 1
            else:
                if (s.state == INIT):
                    if (s.start):
                        s.state <<= LOAD
                        s.rdy <<= 0
                        s.emif_write <<= 1
                    else:
                        s.rdy <<= 1
                elif (s.state == LOAD):
                    if (s.emif_waitrequest == 0):
                        if (s.address == (write_count-1)):
                            s.address <<= 0
                            if (s.buf_count == (buffer_count-1)):
                                s.state <<= INIT
                                s.buf_count <<= 0
                                s.emif_write <<= 0
                                s.emif_address <<= 0
                            else:
                                s.buf_count <<= s.buf_count + 1
                                s.emif_address <<= s.emif_address + 1
                        else:
                            s.address <<= s.address + 1
                            s.emif_address <<= s.emif_address + 1
        utils.tie_off_clk_reset(s)
            
class StateMachineEMIF(Component):
    """" This module includes the whole datapath and the state machine controlling it:

         :param datapath: Contains all MLB modules
         :type datapath: Component
         :param weight_buffer_load: Sets control signals for the weight buffers
         :type  weight_buffer_load: Component
         :param input_buffer_load: Sets control signals for the weight buffers
         :type  input_buffer_load: Component
         :param weight_buffer_control: Sets control signals for the weight buffers
         :type weight_buffer_control: Component
    """
    def construct(s, mlb_spec={}, wb_spec={}, ib_spec={}, ob_spec={}, emif_spec={},
                  proj_spec={}, w_address=0, i_address=0, o_address=0):
        """ Constructor for Datapath

         :param mlb_spec: Contains information about ML block used
         :type mlb_spec: dict
         :param wb_spec: Contains information about weight buffers used
         :type wb_spec: dict
         :param ib_spec: Contains information about input buffers used
         :type ib_spec: dict
         :param ob_spec: Contains information about output buffers used
         :type ob_spec: dict
        """
        print("{:=^60}".format(" Constructing Statemachine with MLB block " +
                               str(mlb_spec.get('block_name', "unnamed") +
                                   " ")))
        MAC_datatypes = ['W', 'I', 'O']
        buffer_specs = {'W': wb_spec, 'I': ib_spec, 'O': ob_spec}

        # Calculate required MLB interface widths and print information
        inner_proj = proj_spec['inner_projection']
        MAC_count = utils.get_mlb_count(inner_proj)
        inner_bus_counts = {dtype: utils.get_proj_stream_count(inner_proj,
                                                               dtype)
                            for dtype in MAC_datatypes}
        inner_data_widths = {dtype: proj_spec['stream_info'][dtype]
                             for dtype in MAC_datatypes}
        inner_bus_widths = {dtype: inner_bus_counts[dtype] *
                            inner_data_widths[dtype]
                            for dtype in MAC_datatypes}
        
        # Calculate required number of MLBs, IO streams, activations
        outer_proj = proj_spec['outer_projection']
        MLB_count = utils.get_mlb_count(outer_proj)
        outer_bus_counts = {dtype: utils.get_proj_stream_count(outer_proj,
                                                               dtype)
                            for dtype in MAC_datatypes}
        outer_bus_widths = {dtype: outer_bus_counts[dtype] *
                            inner_bus_widths[dtype]
                            for dtype in MAC_datatypes}
        total_bus_counts = {dtype: outer_bus_counts[dtype] *
                            inner_bus_counts[dtype]
                            for dtype in MAC_datatypes}
        buffer_counts = {dtype: utils.get_num_buffers_reqd(buffer_specs[dtype],
                         outer_bus_counts[dtype], inner_bus_widths[dtype])
                         for dtype in ['I', 'W']}
        buffer_counts['O'] = utils.get_num_buffers_reqd(buffer_specs['O'],
                                                        outer_bus_counts['O'] *
                                                        inner_bus_counts['O'],
                                                        inner_data_widths['I'])
        connected_ins = []
        
        # Instantiate MLBs, buffers, EMIF
        s.datapath = module_classes.Datapath(mlb_spec, wb_spec, ib_spec,
                                             ob_spec, proj_spec)
        s.emif_inst = module_classes.HWB_Sim(emif_spec, proj_spec, sim=True)
        connected_ins = [s.emif_inst.read, s.emif_inst.write, s.emif_inst.writedata,
                         s.emif_inst.address]
        
        # Load data into weight buffers
        s.sm_start = InPort(1)
        addrw_ports = list(utils.get_ports_of_type(buffer_specs['W'], 'ADDRESS', ["in"]))
        dataw_ports = list(utils.get_ports_of_type(buffer_specs['W'], 'DATA', ["out"]))
        s.load_wbufs_emif = SM_LoadBufsEMIF(buffer_counts['W'], 2**addrw_ports[0]["width"],
                                   addrw_ports[0]["width"], dataw_ports[0]["width"],
                                   utils.get_sum_datatype_width(emif_spec, "AVALON_ADDRESS",
                                                                "in"),
                                   utils.get_sum_datatype_width(emif_spec,
                                                                "AVALON_WRITEDATA", "in"),
                                   w_address)
        s.datapath.weight_datain //= s.load_wbufs_emif.buf_writedata
        connected_ins += [s.datapath.weight_modules_portaaddr_top,
                         s.datapath.input_act_modules_portaaddr_top,
                         s.datapath.mlb_modules_a_en_top,
                         s.datapath.mlb_modules_b_en_top,
                          s.datapath.output_act_modules_portaaddr_top,
                          s.datapath.weight_datain,
                          s.datapath.input_datain
                         ]
        wen_ports = list(utils.get_ports_of_type(buffer_specs['W'], 'WEN', ["in"]))
        connected_ins += utils.connect_ports_by_name(s.load_wbufs_emif,
                                                     r"wen_(\d+)",
                                                     s.datapath,
                                                     r"weight_modules_portawe_(\d+)_top")
        s.weight_address = InPort(addrw_ports[0]["width"])
        
        # Load data into input buffers
        addri_ports = list(utils.get_ports_of_type(buffer_specs['I'], 'ADDRESS', ["in"]))
        datai_ports = list(utils.get_ports_of_type(buffer_specs['I'], 'DATA', ["out"]))
        s.load_ibufs_emif = SM_LoadBufsEMIF(
            buffer_counts['I'], 2**addri_ports[0]["width"],
            addri_ports[0]["width"], datai_ports[0]["width"],
            utils.get_sum_datatype_width(emif_spec, "AVALON_ADDRESS", "in"),
            utils.get_sum_datatype_width(emif_spec, "AVALON_WRITEDATA", "in"),
            i_address)
        s.datapath.input_datain //= s.load_ibufs_emif.buf_writedata
        connected_ins += utils.connect_ports_by_name(s.load_ibufs_emif,
                                                     r"wen_(\d+)",
                                                     s.datapath,
                                                     r"input_act_modules_portawe_(\d+)_top")
        
        # Preload weights into MLBs
        s.preload_weights = SM_PreloadMLBWeights(
            addr_width=addrw_ports[0]["width"],
            num_outer_tiles=outer_proj["UG"]["value"],
            outer_tile_size=outer_proj["URN"]["value"]*\
                            outer_proj["URW"]["value"]*\
                            outer_proj["UE"]["value"]*\
                            inner_proj["UG"]["value"]*\
                            inner_proj["URN"]["value"]*\
                            inner_proj["URW"]["value"]*\
                            inner_proj["UE"]["value"],
            inner_tile_size=inner_proj["URN"]["value"]*\
                            inner_proj["URW"]["value"]*\
                            inner_proj["UE"]["value"],
            outer_tile_repeat_x=outer_proj["UB"]["value"],
            inner_tile_repeat_x=inner_proj["UB"]["value"])
        s.external_a_en = InPort(1)
        s.datapath.mlb_modules_a_en_top //= s.preload_weights.wen
        
        ## Stream Inputs into MLB and read outputs into buffer
        addro_ports = list(utils.get_ports_of_type(buffer_specs['O'], 'ADDRESS', ["in"]))
        obuf_len = 2**addro_ports[0]["width"]
        s.stream_inputs = SM_IterateThruAddresses(obuf_len, addri_ports[0]["width"])
        s.stream_outputs = SM_IterateThruAddresses(obuf_len, addri_ports[0]["width"],
                                                   initial_address=-1)
        s.datapath.mlb_modules_b_en_top //= s.stream_inputs.wen
        connected_ins += utils.connect_ports_by_name(s.stream_outputs,
                                                     r"wen",
                                                     s.datapath,
                                                     r"output_act_modules_portawe_(\d+)_top")
        
        # Now read the outputs out to off-chip memory       
        datao_ports = list(utils.get_ports_of_type(buffer_specs['O'], 'DATA', ["out"]))
        s.external_out = OutPort(datao_ports[0]["width"])
        
        s.write_off_emif = SM_WriteOffChipEMIF(buffer_counts['O'],
                                               2**addro_ports[0]["width"],
                                               addro_ports[0]["width"],
                                               datao_ports[0]["width"],
                                               utils.get_sum_datatype_width(emif_spec,
                                                   "AVALON_ADDRESS", "in"),
                                               utils.get_sum_datatype_width(emif_spec,
                                                   "AVALON_WRITEDATA", "in"),
                                               o_address)


        connected_ins += utils.connect_ports_by_name(s.datapath,
                                                     r"portadataout_(\d+)",
                                                     s.write_off_emif,
                                                     r"datain_(\d+)")   
        #s.external_out //= s.write_off.dataout
        s.state = Wire(5)
        s.done = OutPort(1)
        INIT, LOADING_W_BUFFERS, LOADING_I_BUFFERS, LOADING_MLBS, STREAMING_MLBS, \
            WRITE_OFFCHIP, DONE = Bits5(1),Bits5(2),Bits5(3),Bits5(4),Bits5(5), \
                Bits5(6),Bits5(7)
   
        @update_ff
        def connect_weight_address_ff():
            #print("---------------- STATE " + str(s.state))
            if s.reset:
                s.state <<= INIT
                s.done <<= 0
            else:
                if (s.state == INIT):
                    if s.sm_start:
                        s.state <<= LOADING_W_BUFFERS
                elif (s.state == LOADING_W_BUFFERS):
                    if s.load_wbufs_emif.rdy:
                        s.state <<= LOADING_I_BUFFERS
                elif (s.state == LOADING_I_BUFFERS):
                    if s.load_ibufs_emif.rdy:
                        s.state <<= LOADING_MLBS
                elif (s.state == LOADING_MLBS):
                    if s.preload_weights.rdy:
                        s.state <<= STREAMING_MLBS
                elif (s.state == STREAMING_MLBS):
                    if s.stream_outputs.rdy:
                        s.state <<= WRITE_OFFCHIP
                elif (s.state == WRITE_OFFCHIP):
                    if s.write_off_emif.rdy:
                        s.state <<= DONE
                        s.done <<= 1
                elif s.state == DONE:
                    s.done <<= 1
                    
        @update
        def connect_weight_address():
            if (s.state == LOADING_MLBS):
                s.datapath.weight_modules_portaaddr_top @= s.preload_weights.address
            else:
                s.datapath.weight_modules_portaaddr_top @= s.load_wbufs_emif.buf_address
            if (s.state == STREAMING_MLBS):
                s.datapath.input_act_modules_portaaddr_top @= s.stream_inputs.address
            else:
                s.datapath.input_act_modules_portaaddr_top @= s.load_ibufs_emif.buf_address
            if (s.state == STREAMING_MLBS):
                s.datapath.output_act_modules_portaaddr_top @= s.stream_outputs.address
            else:
                s.datapath.output_act_modules_portaaddr_top @= s.write_off_emif.address
            if (s.state == WRITE_OFFCHIP):
                s.emif_inst.address @= s.write_off_emif.emif_address
                s.emif_inst.read @= s.write_off_emif.emif_read
                s.emif_inst.write @= s.write_off_emif.emif_write
                s.emif_inst.writedata @= s.write_off_emif.emif_writedata
            elif (s.state == LOADING_W_BUFFERS):
                s.emif_inst.address @= s.load_wbufs_emif.emif_address
                s.emif_inst.read @= s.load_wbufs_emif.emif_read
                s.emif_inst.write @= s.load_wbufs_emif.emif_write
                s.emif_inst.writedata @= s.load_wbufs_emif.emif_writedata
            else:
                s.emif_inst.address @= s.load_ibufs_emif.emif_address
                s.emif_inst.read @= s.load_ibufs_emif.emif_read
                s.emif_inst.write @= s.load_ibufs_emif.emif_write
                s.emif_inst.writedata @= s.load_ibufs_emif.emif_writedata
            s.load_wbufs_emif.start @= (s.state == INIT) & s.sm_start 
            s.load_wbufs_emif.emif_readdatavalid @= s.emif_inst.readdatavalid
            s.load_wbufs_emif.emif_waitrequest @= s.emif_inst.waitrequest
            s.load_wbufs_emif.emif_readdata @= s.emif_inst.readdata
            s.load_ibufs_emif.start @= (s.state == LOADING_W_BUFFERS) & \
                s.load_wbufs_emif.rdy 
            s.load_ibufs_emif.emif_readdatavalid @= s.emif_inst.readdatavalid
            s.load_ibufs_emif.emif_waitrequest @= s.emif_inst.waitrequest
            s.load_ibufs_emif.emif_readdata @= s.emif_inst.readdata
            s.preload_weights.start @= (s.state == LOADING_I_BUFFERS) & s.load_ibufs_emif.rdy
            s.stream_inputs.start @= (s.state == LOADING_MLBS) & s.preload_weights.rdy
            s.stream_outputs.start @= (s.state == LOADING_MLBS) & s.preload_weights.rdy 
            s.write_off_emif.start @= (s.state == STREAMING_MLBS) & s.stream_outputs.rdy
            s.write_off_emif.emif_waitrequest @= s.emif_inst.waitrequest
        
        # Connect all inputs not otherwise connected to top
        for inst in [s.datapath, s.emif_inst]:
            for port in (inst.get_input_value_ports()):
                if (port._dsl.my_name not in s.__dict__.keys()) and \
                   (port not in connected_ins):
                    utils.connect_in_to_top(s, port, inst._dsl.my_name + "_" +
                                            port._dsl.my_name + "_sm")
                    print(inst._dsl.my_name + "_" +
                                            port._dsl.my_name + "_sm")
            for port in (inst.get_output_value_ports()):
                if (port._dsl.my_name not in s.__dict__.keys()) and \
                   (port not in connected_ins):
                    utils.connect_out_to_top(s, port, inst._dsl.my_name + "_" +
                                            port._dsl.my_name + "_sm")
                    print(inst._dsl.my_name + "_" +
                                            port._dsl.my_name + "_sm")

        print(s.__dict__)
