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
                if s.we_in & (s.buffer_count == curr_buffer_count):
                    s.we @= 1
                else:
                    s.we @= 0
        else:
            @update
            def upblk_set_wen1():
                if s.we_in:
                    s.we @= 1
                else:
                    s.we @= 0
        utils.AddOutPort(s, 1, "we")  
              
class SM_LoadBufs(Component):
    def construct(s, buffer_count, write_count):
        s.start = InPort(1)
        s.buf_count = Wire(max(int(math.log(buffer_count,2)),1))
        s.buf_address = OutPort(max(int(math.log(write_count,2)),1))
        s.done = OutPort(1)
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
                      
        INIT, LOAD, DONE = 0, 1, 2
        @update_ff
        def upblk_set_wen():
            if s.reset:
                s.buf_count <<= 0
                s.buf_wen <<= 0
                s.buf_state <<= INIT
                s.buf_address <<= 0
                s.done <<= 0
            else:
                if (s.buf_state == INIT):
                    if (s.start):
                        s.buf_state <<= LOAD
                        s.buf_wen <<= 1
                elif (s.buf_state == LOAD):
                    if (s.buf_address == (write_count-1)):
                        s.buf_address <<= 0
                        if (s.buf_count == (buffer_count-1)):
                            s.buf_state <<= DONE
                            s.done <<= 1
                            s.buf_count <<= 0
                            s.buf_wen <<= 0
                        else:
                            s.buf_state <<= LOAD
                            s.buf_count <<= s.buf_count + 1
                    else:
                        s.buf_address <<= s.buf_address + 1
                elif (s.buf_state == DONE):
                    if not (s.start):
                        s.buf_state == INIT

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
        s.outer_tile_index = Wire(w_addr_width)
        s.address = OutPort(addr_width)
        s.addr_w = Wire(w_addr_width)
        s.done = OutPort(1)
        s.state = Wire(2)
        s.wen = OutPort(1)
            
        INIT, LOAD, DONE = 0, 1, 2
        @update
        def upblk_preload_comb():
            s.addr_w @= s.index_within_inner_tile + s.inner_tile_index*inner_tile_size + \
                         s.outer_tile_index*outer_tile_size
            s.address @= s.addr_w[0:addr_width]
            
        @update_ff
        def upblk_preload_ff():
            if s.reset:
                print("SM INIT")
                s.wen <<= 0
                s.state <<= INIT
                s.done <<= 0
                s.outer_tile_repeat_count <<= 0
                s.inner_tile_repeat_count <<= 0
                s.index_within_inner_tile <<= 0
                s.inner_tile_index <<= 0
                s.outer_tile_index <<= 0
            else:
                if (s.state == INIT):
                    print("SM INIT, start {}".format(s.start))
                    if (s.start):
                        s.state <<= LOAD
                        s.wen <<= 1
                elif (s.state == LOAD):
                    print("SM LOAD: {}/{}, {}/{}, {}/{}".format(s.index_within_inner_tile, inner_tile_size,
                                                                s.inner_tile_repeat_count, inner_tile_repeat_x,
                                                         s.inner_tile_index, (outer_tile_size/inner_tile_size)))
                    if (s.index_within_inner_tile < (inner_tile_size - 1)):
                        s.index_within_inner_tile <<= s.index_within_inner_tile + 1
                    else:
                        s.index_within_inner_tile <<= 0
                        if (s.inner_tile_repeat_count < (inner_tile_repeat_x - 1)):
                            s.inner_tile_repeat_count <<= s.inner_tile_repeat_count + 1
                        else:
                            s.inner_tile_repeat_count <<= 0
                            if (s.inner_tile_index < int(outer_tile_size/inner_tile_size)-1):
                                s.inner_tile_index <<= s.inner_tile_index + 1
                            else:
                                s.inner_tile_index <<= 0
                                if (s.outer_tile_repeat_count < (outer_tile_repeat_x - 1)):
                                    s.outer_tile_repeat_count <<= s.outer_tile_repeat_count + 1
                                else:
                                    s.outer_tile_repeat_count <<= 0
                                    if (s.outer_tile_index < (num_outer_tiles - 1)):
                                        s.outer_tile_index <<= s.outer_tile_index + 1
                                    else:
                                        s.outer_tile_index <<= 0
                                        s.state <<= DONE
                                        s.wen <<= 0
                                        s.done <<= 1
                elif (s.state == DONE):
                    if not (s.start):
                        s.state == INIT
        
class SM_IterateThruAddresses(Component):
    def construct(s, write_count, addr_width):
        s.start = InPort(1)
        s.incr = OutPort(addr_width)
        s.buf_address = OutPort(addr_width)
        s.address_offset = Wire(addr_width)
        s.done = OutPort(1)
        s.buf_state = Wire(2)

        @update
        def upblk_set_wenc():
            s.buf_address @= s.address_offset + s.incr
            
        INIT, LOAD, DONE = 0, 1, 2
        @update_ff
        def upblk_set_wen():
            if s.reset:
                s.buf_state <<= INIT
                s.incr <<= 0
                s.done <<= 0
            else:
                if (s.buf_state == INIT):
                    if (s.start):
                        s.buf_state <<= LOAD
                        s.done <<= 0
                        s.incr <<= s.incr + 1
                    s.incr <<= 0
                elif (s.buf_state == LOAD):
                    if (s.buf_address == (write_count-2)):
                        s.buf_state <<= INIT
                        s.done <<= 1
                    s.incr <<= s.incr + 1
            print("Info Inner")
            print(s.buf_state)
            print(s.buf_address)
            print(s.start)
            
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

        # Instantiate MLBs, buffers
        s.datapath = module_classes.Datapath(mlb_spec, wb_spec, ib_spec,
                                             ob_spec, proj_spec)

        # Load data into weight buffers
        s.sm_start = InPort(1)
        addrw_ports = list(utils.get_ports_of_type(buffer_specs['W'], 'ADDRESS', ["in"]))
        s.load_wbufs = SM_LoadBufs(buffer_counts["W"], 2**addrw_ports[0]["width"])
        s.load_wbufs.start //= s.sm_start   
        connected_ins = [s.datapath.weight_modules_portaaddr_top]
        connected_ins += utils.connect_ports_by_name_v2(s.load_wbufs,
                                                     r"wen_(\d+)",
                                                     s.datapath,
                                                     r"weight_modules_portawe_(\d+)_top")
        #s.datapath.weight_modules_portaaddr_top //= s.load_wbufs.buf_address
        s.weight_address = InPort(addrw_ports[0]["width"])

        # Load data into input buffers
        addri_ports = list(utils.get_ports_of_type(buffer_specs['I'], 'ADDRESS', ["in"]))
        s.load_ibufs = SM_LoadBufs(buffer_counts["I"], 2**addri_ports[0]["width"])
        s.load_ibufs.start //= s.load_wbufs.done  
        connected_ins += [s.datapath.input_act_modules_portaaddr_top]
        connected_ins += utils.connect_ports_by_name_v2(s.load_ibufs,
                                                     r"wen_(\d+)",
                                                     s.datapath,
                                                     r"input_act_modules_portawe_(\d+)_top")
        s.datapath.input_act_modules_portaaddr_top //= s.load_ibufs.buf_address
        
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
        s.preload_weights.start //= s.load_ibufs.done
        s.external_a_en = InPort(1)
        s.datapath.mlb_modules_a_en_top //= s.preload_weights.wen
        connected_ins += [s.datapath.mlb_modules_a_en_top]
        
        # Stream Inputs into MLB
        #addro_ports = list(utils.get_ports_of_type(buffer_specs['O'], 'ADDRESS', ["in"]))
        #s.preload_weights = SM_PreloadMLBWeights(
        #    addr_width=addri_ports[0]["width"],
        #    num_outer_tiles=outer_proj["UG"]["value"],
        #    outer_tile_size=outer_proj["URN"]["value"]*\
        #                    outer_proj["URW"]["value"]*\
        #                    outer_proj["UE"]["value"]*\
        #                    inner_proj["UG"]["value"]*\
        #                    inner_proj["URN"]["value"]*\
        #                    inner_proj["URW"]["value"]*\
        #                    inner_proj["UE"]["value"],
        #    inner_tile_size=inner_proj["URN"]["value"]*\
        #                    inner_proj["URW"]["value"]*\
        #                    inner_proj["UE"]["value"],
        #    outer_tile_repeat_x=1,
        #    inner_tile_repeat_x=1)
        #s.preload_weights.start //= s.load_ibufs.done
        #s.external_a_en = InPort(1)
        #s.datapath.mlb_modules_a_en_top //= s.preload_weights.wen
        #connected_ins += [s.datapath.mlb_modules_a_en_top]

        @update
        def connect_weight_address():
            if s.load_wbufs.done:
                if s.preload_weights.done:
                    s.datapath.weight_modules_portaaddr_top @= s.weight_address
                elif s.load_wbufs.done:
                    s.datapath.weight_modules_portaaddr_top @= s.preload_weights.address
            else:
                s.datapath.weight_modules_portaaddr_top @= s.load_wbufs.buf_address
        
        @update_ff
        def connect_weight_addressf():
            print("PYTHON- " + str(s.weight_address))
            print("SM- " + str(s.preload_weights.address))
            print("PYTHON EN- " + str(s.external_a_en))
            #if s.load_wbufs.done and not s.preload_weights.done:
            #    assert(s.weight_address == s.preload_weights.address)

            
        connected_outs = []
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
                   (port not in connected_outs):
                    utils.connect_out_to_top(s, port, inst._dsl.my_name + "_" +
                                            port._dsl.my_name + "_sm")
                    print(inst._dsl.my_name + "_" +
                                            port._dsl.my_name + "_sm")

        print(s.__dict__)
