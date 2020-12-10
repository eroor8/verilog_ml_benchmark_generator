""""
PYMTL Helper modules for implementing statemachine, simmodels
- MAC
- Register
- Shift Register
- MLB
- SingleValueBuffer
- Buffer
"""
from pymtl3 import *
from pymtl3.passes.backends.verilog import *
from pymtl3.passes.backends.yosys import *
import sys
import os
import math
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
import module_classes
import random

class MUX_NXN(Component):
    def construct(s, input_width, input_count, sim=False):
        assert(input_width>0)
        assert(input_count>0)
        utils.add_n_inputs(s, input_count, input_width, "in")
        s.sel = InPort(math.ceil(math.log(max(input_count,2),2)))
        for i in range(input_count):
            newout = utils.AddOutPort(s, input_width, "out"+str(i))
            newmux = MUXN(input_width, input_count)
            setattr(s, "mux" + str(i), newmux)
            newmux.sel //= s.sel
            newout //= newmux.out
            for j in range(input_count):
                inport = getattr(s, "in" + str((j+i)%input_count))
                inportm = getattr(newmux, "in" + str(j))
                inportm //= inport
        utils.tie_off_clk_reset(s)

class MUXN(Component):
    def construct(s, input_width, input_count, sim=False):
        assert(input_width>0)
        assert(input_count>0)
        s.long_input = Wire(input_width*input_count)
        for i in range(input_count):
            newin = utils.AddInPort(s, input_width, "in"+str(i))
            s.long_input[i*input_width:(i+1)*input_width] //= newin 
        utils.AddOutPort(s, input_width, "out")
        utils.AddInPort(s, math.ceil(math.log(max(input_count,2),2)), "sel")
        s.w_sel = Wire(math.ceil(math.log(max(input_width,2),2))+2)
        s.w_sel[0:math.ceil(math.log(max(input_count,2),2))] //= s.sel
        if (input_count > 1):
            @update
            def upblk_set_wen0():
                s.out @= s.long_input[s.w_sel*input_width:(s.w_sel+1)*input_width]
        else:
            @update
            def upblk_set_wen1():
                s.out @= s.long_input
        utils.tie_off_clk_reset(s)
        
class Register(Component):
    """" This module implements a single register.
         :param input_data: Input port
         :type input_data: Component class
         :param output_data: Output port
         :type output_data: Component class
    """
    def construct(s, reg_width, preload_value=0, sim=False):
        """ Constructor for register
         :param reg_width: Bit-width of register
         :type reg_width: int
        """
        utils.AddOutPort(s, reg_width,"output_data")
        utils.AddInPort(s, reg_width,"input_data")
        utils.AddInPort(s, 1,"ena")
        s.REG = Wire(reg_width)
        
        @update_ff
        def upblk0():
            if s.reset:
                s.REG <<= preload_value
            else:
                if s.ena:
                    s.REG <<= s.input_data
        s.output_data //= s.REG


class ShiftRegister(Component):
    """" This module implements a shift register of length ``n``
         :param input_data: Input port
         :type input_data: Component class
         :param output_data_<i>: Output port
         :type output_data_<i>: Component class
    """
    def construct(s, reg_width=1, length=1, sim=False):
        """ Constructor for shift register

         :param reg_width: Bit-width of registers
         :type reg_width: int
         :param length: Shift register length
         :type length: int
        """
        utils.AddOutPort(s, reg_width, "output_data")
        utils.AddInPort(s, reg_width, "input_data")
        utils.AddInPort(s, 1, "ena")

        for shift_reg in range(length):
            newreg = Register(reg_width, sim=sim)
            setattr(s, "SR_" + str(shift_reg), newreg)
            connect(newreg.ena, s.ena)

        if (length > 0):
            connect(s.SR_0.input_data, s.input_data)
            last_reg = getattr(s, "SR_" + str(length - 1))
            s.output_data //= last_reg.output_data
        else:
            s.output_data //= s.input_data

        for shift_reg in range(1, length):
            SR = getattr(s, "SR_" + str(shift_reg))
            PREV_SR = getattr(s, "SR_" + str(shift_reg - 1))
            connect(SR.input_data, PREV_SR.output_data)

class EMIF(Component):
    """" This module implements a  buffer
         :param datain: Input port
         :type datain: Component class
         :param address: Output data
         :type address: Component class
         :param dataout: Output data
         :type dataout: Component class
         :param wen: Write enable
         :type wen: Component class
    """
    def construct(s, datawidth=8, length=1, startaddr=0,
                  preload_vector=[], pipelined=False,
                  max_pipeline_transfers=4, sim=False):
        """ Constructor for Buffer
         :param datawidth: Bit-width of input, output data
         :type datawidth: int
         :param addrwidth: Bit-width of address
         :type addrwidth: int
         :param length: Number of values in the buffer
         :type length: int
         :param startaddr: Useful if there are many buffers
         :type startaddr: int
        """
        addrwidth = int(math.ceil(math.log(length+startaddr,2)))
        utils.AddInPort(s, addrwidth, "avalon_address")
        #utils.AddInPort(s, int(math.ceil(addrwidth/8)), "avalon_byteenable")
        utils.AddInPort(s, 1, "avalon_read")
        utils.AddInPort(s, 1, "avalon_write")
        utils.AddInPort(s, datawidth, "avalon_writedata")
        #utils.AddInPort(s, 1, "avalon_lock")
        utils.AddOutPort(s, datawidth, "avalon_readdata")
        utils.AddOutPort(s, 1, "avalon_waitrequest")
        utils.AddOutPort(s, 1, "avalon_readdatavalid")
        utils.AddOutPort(s, 1, "avalon_writeresponsevalid")
        #utils.AddOutPort(s, 1, "burstcount")
        #utils.AddOutPort(s, 1, "beginbursttransfer")
        
        #s.data = Wire(length*datawidth)
        wide_addr_width = math.ceil(math.log(datawidth*(length+startaddr), 2))
        s.waddress = Wire(wide_addr_width)
        connect(s.waddress[0:addrwidth], s.avalon_address)
        s.waddress[addrwidth:wide_addr_width] //= 0
        
        s.buf = Buffer(datawidth, length, startaddr, preload_vector, keepdata=False,
                       sim=True)
        INIT, WAIT_READING, DONE_READ, WAIT_WRITING, DONE_WRITE, DONE = Bits5(1), \
            Bits5(2),Bits5(3),Bits5(4),Bits5(5),Bits5(6)
        s.state = Wire(5)
        curr_pipeline_transfer_count = 0
        s.latency_countdown = Wire(10)
        pending_transfers = [[] for i in range(max_pipeline_transfers+1)]
        s.curr_pending_start = Wire(10)
        s.curr_pending_end = Wire(10)
        
        @update_ff
        def upff():
            curr_rand = random.randint(0,3)
            if s.reset:
                s.state <<= INIT
                s.curr_pending_start <<= 0
                s.curr_pending_end <<= 0
                if (pipelined):
                    s.avalon_readdatavalid <<= 0
                    s.avalon_writeresponsevalid <<= 0          
            else:  
                num_pending_transfers = s.curr_pending_end - s.curr_pending_start
                #print("Pending " + str(pending_transfers))
                #print("Num pending" + str(num_pending_transfers))
                #print("Read " + str(s.avalon_read))
                #print("Address" + str(s.avalon_address))
                #print("Readdata " + str(s.avalon_readdata))
                #print("Writedata " + str(s.avalon_writedata))
                #print("Readdatavalid " + str(s.avalon_readdatavalid))
                #print("Waitrequest " + str(s.avalon_waitrequest))
                #print("Write " + str(s.avalon_write))
                #print("Countdown " + str(s.latency_countdown))
                if pipelined:
                    if (s.avalon_read or s.avalon_write) and \
                       (num_pending_transfers < max_pipeline_transfers): 
                            pending_transfers[s.curr_pending_end%max_pipeline_transfers] = \
                                [int(s.avalon_read), int(s.avalon_write),
                                 int(s.avalon_address), int(s.avalon_writedata)]
                            s.curr_pending_end <<= s.curr_pending_end + 1

                    if (s.latency_countdown == 0) and (num_pending_transfers > 0):
                        s.avalon_readdatavalid <<= pending_transfers[s.curr_pending_start % \
                                                      max_pipeline_transfers][0]
                        s.curr_pending_start <<= s.curr_pending_start + 1
                        s.buf.address <<= pending_transfers[s.curr_pending_start % \
                                                            max_pipeline_transfers][2]
                        s.buf.wen <<= pending_transfers[s.curr_pending_start % \
                                                        max_pipeline_transfers][1]
                        s.buf.datain <<= pending_transfers[s.curr_pending_start % \
                                                           max_pipeline_transfers][3]
                        s.latency_countdown <<= curr_rand
                    else:
                        if (s.latency_countdown > 0):
                            s.latency_countdown <<= s.latency_countdown - 1
                        else:
                            s.latency_countdown <<= curr_rand
                        s.avalon_readdatavalid <<= 0
                        
                else:
                    if (s.state == INIT):
                        if s.avalon_read:
                            s.state <<= WAIT_READING
                            s.buf.address <<= s.avalon_address
                            s.latency_countdown <<= curr_rand
                        elif s.avalon_write:
                            s.state <<= WAIT_WRITING
                            s.buf.address <<= s.avalon_address
                            s.buf.datain <<= s.avalon_writedata
                            s.buf.wen <<= 1
                            s.latency_countdown <<= curr_rand
                    elif (s.state == WAIT_READING):
                        s.latency_countdown <<= s.latency_countdown - 1
                        if (s.latency_countdown == 0):
                            s.state <<= DONE_READ
                    elif (s.state == DONE_READ):
                        s.state <<= INIT
                        s.latency_countdown <<= 0
                    elif (s.state ==  WAIT_WRITING):
                        s.latency_countdown <<= s.latency_countdown - 1
                        if (s.latency_countdown == 0):
                            s.state <<= DONE_WRITE
                    elif (s.state ==  DONE_WRITE):
                        s.state <<= INIT
                
        @update
        def upblk0():
            s.avalon_readdata @= s.buf.dataout
            if (pipelined):
                num_pending_transfers = s.curr_pending_end - s.curr_pending_start
                if (s.avalon_read or s.avalon_write) and \
                   (num_pending_transfers == max_pipeline_transfers):
                    s.avalon_waitrequest @= 1
                else:
                    s.avalon_waitrequest @= 0
            else:
                if ((s.state == INIT) and (s.avalon_read == 0) and (s.avalon_write == 0)) \
                   or (s.state == DONE_READ) or (s.state == DONE_WRITE):
                    s.avalon_waitrequest @= 0
                else:
                    s.avalon_waitrequest @= 1

class Buffer(Component):
    """" This module implements a  buffer
         :param datain: Input port
         :type datain: Component class
         :param address: Output data
         :type address: Component class
         :param dataout: Output data
         :type dataout: Component class
         :param wen: Write enable
         :type wen: Component class
    """
    def construct(s, datawidth=8, length=1, startaddr=0,
                  preload_vector=[], keepdata=True, sim=False):
        """ Constructor for Buffer
         :param datawidth: Bit-width of input, output data
         :type datawidth: int
         :param addrwidth: Bit-width of address
         :type addrwidth: int
         :param length: Number of values in the buffer
         :type length: int
         :param startaddr: Useful if there are many buffers
         :type startaddr: int
        """
        addrwidth = math.ceil(math.log(length+startaddr,2))
        utils.AddInPort(s, datawidth, "datain")
        utils.AddInPort(s, addrwidth, "address")
        utils.AddInPort(s, 1, "wen")
        utils.AddOutPort(s, datawidth, "dataout")
        if (keepdata):
            s.data = Wire(length*datawidth)
        s.waddress = Wire(math.ceil(math.log(datawidth*(length+startaddr), 2)))
        connect(s.waddress[0:addrwidth], s.address)
        s.waddress[addrwidth:math.ceil(math.log(datawidth*length, 2))] //= 0
        for i in range(length):
            if i < len(preload_vector):
                preload_value = preload_vector[i]
            else:
                preload_value = 0
            val = BufferValue(datawidth, addrwidth, i + startaddr, preload_value,
                              sim)
            setattr(s, "V" + str(i), val)
            val.datain //= s.datain
            val.wen //= s.wen
            val.address //= s.address
            if (keepdata):
                connect(s.data[(i * datawidth):((i + 1) * datawidth)], val.dataout)

        if keepdata:
            @update
            def upblk0():
                if ((s.waddress - startaddr)) <= (length-1):
                    s.dataout @= s.data[(s.waddress - startaddr) * datawidth:
                        (s.waddress - startaddr) * datawidth + datawidth]
                else:
                    s.dataout @= 0
        else:     
            @update
            def upblk1():
                if ((s.waddress - startaddr)) <= (length-1):
                    currval = getattr(s, "V" + str(int(s.waddress - startaddr)))
                    s.dataout @= currval.dataout
                else:
                    s.dataout @= 0

class BufferValue(Component):
    """" This module implements a single value in a buffer
         :param datain: Input port
         :type datain: Component class
         :param address: Output data
         :type address: Component class
         :param dataout: Output data
         :type dataout: Component class
         :param wen: Write enable
         :type wen: Component class
    """
    def construct(s, datawidth=8, addrwidth=8, addr=0, preload_value=0, sim=False):
        """ Constructor for BufferValue
         :param datawidth: Bit-width of input, output data
         :type datawidth: int
         :param addrwidth: Bit-width of address
         :type addrwidth: int
         :param addr: Address of this value
         :type addr: int
        """
        assert (2 ** addrwidth > addr)
        utils.AddInPort(s, datawidth, "datain")
        utils.AddInPort(s, addrwidth, "address")
        utils.AddInPort(s, 1, "wen")
        utils.AddOutPort(s, datawidth, "dataout")
        s.inner_reg = Register(datawidth, preload_value, sim=sim)
        s.inner_reg.input_data //= s.datain
        s.dataout //= s.inner_reg.output_data
                
        @update
        def upblk0():
            s.inner_reg.ena @= s.wen & (addr == s.address)

class MAC(Component):
    """" This module implements a single MAC.
         Weight and input registers are loaded based on
         w_en and i_en. Output is combinational.
         :param input_in: Input port
         :type input_in: Component class
         :param input_en: Input register enable
         :type input_en: Component class
         :param weight_in: Weight port
         :type weight_in: Component class
         :param weight_en: Weight register enable
         :type weight_en: Component class
         :param sum_in: Input partial sum
         :type sum_in: Component class
         :param input_out: Output port
         :type input_out: Component class
         :param weight_out: Weight output
         :type weight_out: Component class
         :param sum_out: Ouput partial sum
         :type sum_out: Component class
         :param acc_en: Accumulate output
         :type acc_en: Component class
    """
    def construct(s, input_width=8, weight_width=8, sum_width=32, sim=False):
        """ Constructor for register
         :param input_width: Bit-width of register
         :type input_width: int
         :param weight_width: Bit-width of register
         :type weight_width: int
         :param sum_width: Bit-width of register
         :type sum_width: int
        """
        assert(sum_width >= input_width)
        assert(sum_width >= weight_width)
        utils.AddInPort(s,input_width, "input_in")
        utils.AddInPort(s,1, "input_en")
        utils.AddOutPort(s,input_width, "input_out")
        s.input_reg = Register(input_width, sim=sim)
        s.input_reg.input_data //= s.input_in
        s.input_reg.ena //= s.input_en
        s.input_out //= s.input_reg.output_data
        
        utils.AddInPort(s,weight_width, "weight_in")
        utils.AddInPort(s,1, "weight_en")
        utils.AddOutPort(s,weight_width, "weight_out")
        s.weight_reg = Register(weight_width, sim=sim)
        s.weight_reg.input_data //= s.weight_in
        s.weight_reg.ena //= s.weight_en
        s.weight_out //= s.weight_reg.output_data
        
        utils.AddInPort(s,sum_width, "sum_in")
        utils.AddInPort(s,1, "acc_en")
        utils.AddOutPort(s,sum_width, "sum_out")
        s.sum_reg = Register(sum_width, sim=sim)
        s.sum_reg.ena //= 1 #s.acc_en

        s.input_out_w = Wire(sum_width)
        s.weight_out_w = Wire(sum_width)
        s.input_in_w = Wire(sum_width)
        s.weight_in_w = Wire(sum_width)
        if (sum_width > input_width):
            connect(s.input_in_w[0:input_width], s.input_in)
            connect(s.input_out_w[0:input_width], s.input_out)
            s.input_in_w[input_width:sum_width] //= 0
            s.input_out_w[input_width:sum_width] //= 0
        else:
            s.input_out_w //= s.input_out
            s.input_in_w //= s.input_in
            
        if (sum_width > weight_width):
            connect(s.weight_in_w[0:weight_width], s.weight_in)
            connect(s.weight_out_w[0:weight_width], s.weight_out)
            s.weight_in_w[weight_width:sum_width] //= 0
            s.weight_out_w[weight_width:sum_width] //= 0
        else:
            s.weight_out_w //= s.weight_out
            s.weight_in_w //= s.weight_in
                
        @update
        def upblk0():
            if s.acc_en:
                #print("ACC SUM_REG <= " + str(int(s.sum_reg.output_data)) + " + " + str(int(s.input_in_w)) + " * " + str(int(s.weight_in_w)) + "=" + str((s.weight_in_w*s.input_in_w)) )
                #print("ACC    OUT <= " + str(int(s.sum_reg.output_data)))
                s.sum_out @= s.sum_reg.output_data + \
                                    s.input_in_w * s.weight_in_w
                s.sum_reg.input_data @= s.sum_reg.output_data + \
                                    s.input_in_w * s.weight_in_w
            else:
                #print("SUM_REG <= " + str(int(s.input_in_w)) + " * " +  str(int(s.weight_in_w)) + "=" + str(int((s.weight_in_w*s.input_in_w))%(2^4))  )
                #print("    OUT <= " + str(int(s.sum_in + s.input_out_w * s.weight_out_w)))
                s.sum_out @= s.sum_in + s.input_out_w * s.weight_out_w
                s.sum_reg.input_data @= s.input_in_w * s.weight_in_w

                
class MACWrapper(Component):
    """" This module wraps several instantiations of MAC.
         It has the same inputs and outputs as the MAC component * the
         number of instantiations, named <MAC_port>_<instance>.
    """
    def construct(s, count=1, input_width=1,
                  weight_width=1, sum_width=32, sim=False):
        """ Constructor for MACWrapper

         :param count: Number of activation functions to instantiate
         :type count: int
         :param input_width: Bit-width of register
         :type input_width: int
         :param weight_width: Bit-width of register
         :type weight_width: int
         :param sum_width: Bit-width of register
         :type sum_width: int
        """
        for i in range(count):
            curr_inst = MAC(input_width, weight_width, sum_width, sim)
            setattr(s, 'MAC_inst_' + str(i), curr_inst)

            for port in curr_inst.get_input_value_ports():
                utils.connect_in_to_top(s, port, port._dsl.my_name + "_" +
                                        str(i))
            for port in curr_inst.get_output_value_ports():
                utils.connect_out_to_top(s, port, port._dsl.my_name + "_" +
                                         str(i))


class MLB(Component):
    """" This is the sim model for an MLB block with given projection.
    """
    def construct(s, proj_specs, sim=False):
        """ Constructor for MLB

         :param proj_spec: Dictionary describing projection of computations
                            onto ML block
         :type proj_spec: dict
        """
        if ["inner_projection"] in proj_specs:
            proj_specs = [proj_specs]
        #proj_spec = proj_specs[0]
        MAC_datatypes = ['W', 'I', 'O']

        # Calculate required MLB interface widths and print information
        #proj_specs = [proj_specs[1],proj_specs[1]]
        projs = [proj_spec["inner_projection"] for proj_spec in proj_specs]
        MAC_counts = [utils.get_mlb_count(proj) for proj in projs]
        bus_counts = {dtype: [utils.get_proj_stream_count(proj, dtype)
                              for proj in projs]
                                   for dtype in MAC_datatypes}
        data_widths = {dtype: [proj_spec['stream_info'][dtype]
                               for proj_spec in proj_specs]
                                    for dtype in MAC_datatypes}
        bus_widths = {dtype: [bus_count * data_width
                              for (bus_count, data_width) in zip(bus_counts[dtype], data_widths[dtype])]
                                   for dtype in MAC_datatypes}

        # Instantiate MACs, IOs
        s.mac_modules = MACWrapper(max(MAC_counts),
                                   max(data_widths['I']),
                                   max(data_widths['W']),
                                   max(data_widths['O']), sim)
        utils.AddInPort(s, max(bus_widths['W']), "W_IN")
        utils.AddOutPort(s, max(bus_widths['W']), "W_OUT")
        utils.AddInPort(s, 1, "W_EN")
        utils.AddInPort(s, max(bus_widths['I']), "I_IN")
        utils.AddOutPort(s, max(bus_widths['I']), "I_OUT")
        utils.AddInPort(s, 1, "I_EN")
        utils.AddInPort(s, 1, "ACC_EN")
        utils.AddInPort(s, max(bus_widths['O']), "O_IN")
        utils.AddOutPort(s, max(bus_widths['O']), "O_OUT")

        # Instantiate interconnects
        weight_interconnects = []
        input_interconnects = []
        output_ps_interconnects = []
        output_interconnects = []
        s.sel = InPort(math.ceil(math.log(max(len(proj_specs),2),2)))
        utils.tie_off_port(s, s.sel)
        for i in range(len(proj_specs)):
            if (i > 0):
                newname = proj_specs[i].get("name",i)
            else:
                newname = ""
            weight_interconnect = module_classes.WeightInterconnect(
                buffer_width=bus_widths['W'][i],
                mlb_width_used=data_widths['W'][i],
                num_mlbs=max(MAC_counts),
                projection=projs[i],
                sim=sim,
                num_mlbs_used=MAC_counts[i]
            )
            setattr(s,"weight_interconnect"+newname, weight_interconnect)
            weight_interconnects += [weight_interconnect]
            input_interconnect = module_classes.InputInterconnect(
                buffer_width=bus_widths['I'][i],
                mlb_width_used=data_widths['I'][i],
                num_mlbs=max(MAC_counts),
                projection=projs[i], sim=sim)
            setattr(s,"input_interconnect"+newname, input_interconnect)
            input_interconnects += [input_interconnect]
            output_ps_interconnect = module_classes.OutputPSInterconnect(
                af_width=data_widths['O'][i],
                mlb_width_used=data_widths['O'][i],
                num_afs=max(bus_counts['O']),
                num_mlbs=max(MAC_counts),
                projection=projs[i], sim=sim,
                input_buf_width=bus_widths['O'][i],
                num_input_bufs=1
            )
            output_ps_interconnects += [output_ps_interconnect]
            setattr(s,"output_ps_interconnect"+newname, output_ps_interconnect)
            output_interconnect = module_classes.MergeBusses(
                in_width=data_widths['O'][i],
                num_ins=max(bus_counts['O']),
                out_width=bus_widths['O'][i],
                sim=sim)
            output_interconnects += [output_interconnect]
            setattr(s,"output_interconnect"+newname, output_interconnect)
        
        # Connect between interconnects, MACs and top level
        for i in range(len(proj_specs)):
            weight_interconnect = weight_interconnects[i]
            utils.connect_ports_by_name(s.mac_modules,
                "weight_out_(\d+)", weight_interconnect, "inputs_from_mlb_(\d+)")
            utils.connect_inst_ports_by_name(s,
                "W_IN", weight_interconnect, "inputs_from_buffer")
        utils.mux_ports_by_name(s, weight_interconnects,
                                "outputs_to_mlb_(\d+)", s.mac_modules, "weight_in_(\d+)", insel=s.sel, sim=True)
        utils.mux_inst_ports_by_name(s, "W_OUT", weight_interconnects,
            "outputs_to_buffer_(\d+)", insel=s.sel, sim=True)
        
        for i in range(len(proj_specs)):
            input_interconnect = input_interconnects[i]
            utils.connect_ports_by_name(s.mac_modules,
                "input_out_(\d+)", input_interconnect, "inputs_from_mlb_(\d+)")
            utils.connect_inst_ports_by_name(s,
                "I_IN", input_interconnect, "inputs_from_buffer")
        utils.mux_ports_by_name(s, input_interconnects,
            "outputs_to_mlb_(\d+)", s.mac_modules, "input_in_(\d+)", insel=s.sel, sim=True)
        utils.mux_inst_ports_by_name(s, "I_OUT", input_interconnects,
            "outputs_to_buffer_(\d+)", insel=s.sel, sim=True)
        
        for i in range(len(proj_specs)):
            output_ps_interconnect = output_ps_interconnects[i]
            output_interconnect = output_interconnects[i]
            utils.connect_ports_by_name(s.mac_modules,
                "sum_out_(\d+)", output_ps_interconnect, "inputs_from_mlb_(\d+)")
            utils.connect_ports_by_name(output_ps_interconnect,
                "outputs_to_afs_(\d+)", output_interconnect, "input_(\d+)")
            utils.connect_inst_ports_by_name(s,
                "O_IN", output_ps_interconnect, "ps_inputs_from_buffer")
        utils.mux_ports_by_name(s, output_ps_interconnects,
            "outputs_to_mlb_(\d+)", s.mac_modules, "sum_in_(\d+)", insel=s.sel, sim=True)
        utils.mux_inst_ports_by_name(s, "O_OUT", output_interconnects,
            "output_(\d+)", insel=s.sel, sim=True)
        utils.connect_inst_ports_by_name(s, "W_EN", s.mac_modules, "weight_en")
        utils.connect_inst_ports_by_name(s, "I_EN", s.mac_modules, "input_en")
        utils.connect_inst_ports_by_name(s, "ACC_EN", s.mac_modules, "acc_en")

