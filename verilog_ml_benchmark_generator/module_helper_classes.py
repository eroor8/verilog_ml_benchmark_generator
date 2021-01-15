""""
PYMTL Helper modules for implementing statemachine, simmodels
- MAC
- Register
- Shift Register
- MLB
- SingleValueBuffer
- Buffer
"""
from pymtl3 import InPort, Component, Wire, update, update_ff, connect, Bits5
import sys
import os
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
import module_classes
import random
fast_generate = False


class MUX_NXN(Component):
    """" Implements a crossbar of N N-input muxes.
    """
    def construct(s, input_width, input_count, sim=False):
        """ Constructor for N-input MUX

         :param input_width: Bit-width of input
         :param input_count: Number of inputs
         :param sim: Whether to skip synthesis
        """
        assert(input_width > 0)
        assert(input_count > 0)
        utils.add_n_inputs(s, input_count, input_width, "in")
        s.sel = InPort(math.ceil(math.log(max(input_count, 2), 2)))
        for i in range(input_count):
            newout = utils.AddOutPort(s, input_width, "out" + str(i))
            if (input_count == 3):
                newmux = MUX3(input_width)
            elif (input_count == 6):
                newmux = MUX6(input_width)
            elif (input_count == 9):
                newmux = MUX9(input_width)
            elif (input_count == 12):
                newmux = MUX12(input_width)
            else:
                if not (input_count == 1):
                    print(input_count)
                newmux = MUXN(input_width, input_count)
            setattr(s, "mux" + str(i), newmux)
            newmux.sel //= s.sel
            newout //= newmux.out
            for j in range(input_count):
                inport = getattr(s, "in" + str((j + i) % input_count))
                inportm = getattr(newmux, "in" + str(j))
                inportm //= inport
        utils.tie_off_clk_reset(s)


class MUX2(Component):
    """" Implements a single 2-input mux
    """
    def construct(s, input_width, sel_width, threshhold, sim=False):
        """ Constructor for 2-input MUX

         :param input_width: Bit-width of input
         :param sel_width: Width of sel signal
         :param threshold: If sel > threshold, choose second input
         :param sim: Whether to skip synthesis
        """
        assert(input_width > 0)
        s.in0 = InPort(input_width)
        s.in1 = InPort(input_width)
        s.sel = InPort(sel_width)
        utils.AddOutPort(s, input_width, "out")

        @update
        def upblk_set_wen1():
            if (s.sel > threshhold):
                s.out @= s.in1
            else:
                s.out @= s.in0
        utils.tie_off_clk_reset(s)


class MUX3(Component):
    """" Implements a single 3-input mux.
    """
    def construct(s, input_width, sim=False):
        """ Constructor for MUX

         :param input_width: Bit-width of input
         :param input_count: Number of inputs to mux between
         :param sim: Whether to skip synthesis
        """
        assert(input_width > 0)
        for i in range(3):
            utils.AddInPort(s, input_width, "in" + str(i))
        utils.AddOutPort(s, input_width, "out")
        utils.AddInPort(s, 2, "sel")

        @update
        def upblk_set_wen0():
            if (s.sel == 0):
                s.out @= s.in0
            elif (s.sel == 1):
                s.out @= s.in1
            else:
                s.out @= s.in2
        utils.tie_off_clk_reset(s)


class MUX6(Component):
    """" Implements a single 6-input mux.
    """
    def construct(s, input_width, sim=False):
        """ Constructor for MUX

         :param input_width: Bit-width of input
         :param input_count: Number of inputs to mux between
         :param sim: Whether to skip synthesis
        """
        assert(input_width > 0)
        for i in range(6):
            utils.AddInPort(s, input_width, "in" + str(i))
        utils.AddOutPort(s, input_width, "out")
        utils.AddInPort(s, 3, "sel")

        @update
        def upblk_set_wen0():
            if (s.sel == 0):
                s.out @= s.in0
            elif (s.sel == 1):
                s.out @= s.in1
            elif (s.sel == 2):
                s.out @= s.in2
            elif (s.sel == 3):
                s.out @= s.in3
            elif (s.sel == 4):
                s.out @= s.in4
            else:
                s.out @= s.in5
        utils.tie_off_clk_reset(s)


class MUX9(Component):
    """" Implements a single 6-input mux.
    """
    def construct(s, input_width, sim=False):
        """ Constructor for MUX

         :param input_width: Bit-width of input
         :param input_count: Number of inputs to mux between
         :param sim: Whether to skip synthesis
        """
        assert(input_width > 0)
        for i in range(9):
            utils.AddInPort(s, input_width, "in" + str(i))
        utils.AddOutPort(s, input_width, "out")
        utils.AddInPort(s, 4, "sel")

        @update
        def upblk_set_wen0():
            if (s.sel == 0):
                s.out @= s.in0
            elif (s.sel == 1):
                s.out @= s.in1
            elif (s.sel == 2):
                s.out @= s.in2
            elif (s.sel == 3):
                s.out @= s.in3
            elif (s.sel == 4):
                s.out @= s.in4
            elif (s.sel == 5):
                s.out @= s.in5
            elif (s.sel == 6):
                s.out @= s.in6
            elif (s.sel == 7):
                s.out @= s.in7
            else:
                s.out @= s.in8
        utils.tie_off_clk_reset(s)


class MUX12(Component):
    """" Implements a single 6-input mux.
    """
    def construct(s, input_width, sim=False):
        """ Constructor for MUX

         :param input_width: Bit-width of input
         :param input_count: Number of inputs to mux between
         :param sim: Whether to skip synthesis
        """
        assert(input_width > 0)
        for i in range(12):
            utils.AddInPort(s, input_width, "in" + str(i))
        utils.AddOutPort(s, input_width, "out")
        utils.AddInPort(s, 4, "sel")

        @update
        def upblk_set_wen0():
            if (s.sel == 0):
                s.out @= s.in0
            elif (s.sel == 1):
                s.out @= s.in1
            elif (s.sel == 2):
                s.out @= s.in2
            elif (s.sel == 3):
                s.out @= s.in3
            elif (s.sel == 4):
                s.out @= s.in4
            elif (s.sel == 5):
                s.out @= s.in5
            elif (s.sel == 6):
                s.out @= s.in6
            elif (s.sel == 7):
                s.out @= s.in7
            elif (s.sel == 8):
                s.out @= s.in8
            elif (s.sel == 9):
                s.out @= s.in9
            elif (s.sel == 10):
                s.out @= s.in10
            else:
                s.out @= s.in11
        utils.tie_off_clk_reset(s)


class MUXN(Component):
    """" Implements a single N-input mux.
    """
    def construct(s, input_width, input_count, sim=False):
        """ Constructor for MUX

         :param input_width: Bit-width of input
         :param input_count: Number of inputs to mux between
         :param sim: Whether to skip synthesis
        """
        assert(input_width > 0)
        assert(input_count > 0)
        s.long_input = Wire(input_width * input_count)
        for i in range(input_count):
            newin = utils.AddInPort(s, input_width, "in" + str(i))
            s.long_input[i * input_width:(i + 1) * input_width] //= newin
        utils.AddOutPort(s, input_width, "out")
        utils.AddInPort(s, math.ceil(math.log(max(input_count, 2), 2)), "sel")
        s.w_sel = Wire(math.ceil(math.log(max(input_width*input_count, 2), 2)))
        s.w_sel[0:math.ceil(math.log(max(input_count, 2), 2))] //= s.sel
        if (input_count > 1):
            @update
            def upblk_set_wen0():
                s.out @= s.long_input[s.w_sel * input_width:(s.w_sel + 1) *
                                      input_width]
        else:
            @update
            def upblk_set_wen1():
                s.out @= s.long_input
        utils.tie_off_clk_reset(s)


class Register(Component):
    """" Implements a single register.
    """
    def construct(s, reg_width, preload_value=0, sim=False):
        """ Constructor for register

         :param reg_width: Bit-width of register
         :param preload_value: Original value of register
         :param sim: Whether to skip synthesis
        """
        utils.AddOutPort(s, reg_width, "output_data")
        utils.AddInPort(s, reg_width, "input_data")
        utils.AddInPort(s, 1, "ena")
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
    """" Implements a shift register of length ``n``
    """
    def construct(s, reg_width=1, length=1, sim=False):
        """ Constructor for shift register

         :param reg_width: Bit-width of register
         :param length: Length of shift register
         :param sim: Whether to skip synthesis
        """
        utils.AddOutPort(s, reg_width, "output_data")
        utils.AddInPort(s, reg_width, "input_data")
        utils.AddInPort(s, 1, "ena")

        for shift_reg in range(length):
            newreg = Register(reg_width, sim=sim)
            setattr(s, "SR_" + str(shift_reg), newreg)
            connect(newreg.ena, s.ena)
            newout = utils.AddOutPort(s, reg_width, "out" + str(shift_reg))
            newout //= newreg.output_data

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
    """" Implements and initializes an EMIF, including the off-chip data.
    """
    def construct(s, datawidth=8, length=1, startaddr=0,
                  preload_vector=[], pipelined=False,
                  max_pipeline_transfers=4, sim=False, fast_gen=False):
        """ Constructor for Buffer

         :param datawidth: Bit-width of input, output data
         :param length: Number of values in the buffer
         :param startaddr: Useful if there are many buffers
         :param pipelined: Enable pipeline transfers
         :param max_pipeline_transfers: Max pipeline transfers
         :param sim: Whether to skip synthesis
        """
        addrwidth = int(math.ceil(math.log(length + startaddr, 2)))
        utils.AddInPort(s, addrwidth, "avalon_address")
        utils.AddInPort(s, 1, "avalon_read")
        utils.AddInPort(s, 1, "avalon_write")
        utils.AddInPort(s, datawidth, "avalon_writedata")
        utils.AddOutPort(s, datawidth, "avalon_readdata")
        utils.AddOutPort(s, 1, "avalon_waitrequest")
        utils.AddOutPort(s, 1, "avalon_readdatavalid")
        utils.AddOutPort(s, 1, "avalon_writeresponsevalid")

        # s.data = Wire(length*datawidth)
        wide_addr_width = math.ceil(math.log(datawidth * (length + startaddr),
                                             2))
        s.waddress = Wire(wide_addr_width)
        connect(s.waddress[0:addrwidth], s.avalon_address)
        s.waddress[addrwidth:wide_addr_width] //= 0
        if (fast_gen):
            s.avalon_readdata //= 0
            s.avalon_readdatavalid //= 0
            s.avalon_waitrequest //= 0
            s.avalon_writeresponsevalid //= 0
        else:
            s.buf = Buffer(datawidth, length, startaddr, preload_vector,
                           keepdata=False, sim=True)
        INIT, WAIT_READING, DONE_READ, WAIT_WRITING, DONE_WRITE = \
            Bits5(1), Bits5(2), Bits5(3), Bits5(4), Bits5(5)
        s.state = Wire(5)
        s.latency_countdown = Wire(10)
        pending_transfers = [[] for i in range(max_pipeline_transfers + 1)]
        s.curr_pending_start = Wire(10)
        s.curr_pending_end = Wire(10)

        if (not fast_gen):
            @update_ff
            def upff():
                curr_rand = random.randint(0, 3)
                if s.reset:
                    s.state <<= INIT
                    s.curr_pending_start <<= 0
                    s.curr_pending_end <<= 0
                    if (pipelined):
                        s.avalon_readdatavalid <<= 0
                        s.avalon_writeresponsevalid <<= 0
                else:
                    num_pending_transfers = s.curr_pending_end - \
                        s.curr_pending_start
                    if pipelined:
                        if (s.avalon_read or s.avalon_write) and \
                           (num_pending_transfers < max_pipeline_transfers):
                            pending_transfers[s.curr_pending_end %
                                              max_pipeline_transfers] = \
                                                  [int(s.avalon_read),
                                                   int(s.avalon_write),
                                                   int(s.avalon_address),
                                                   int(s.avalon_writedata)]
                            s.curr_pending_end <<= s.curr_pending_end + 1

                        if (s.latency_countdown == 0) and \
                           (num_pending_transfers > 0):
                            s.avalon_readdatavalid <<= pending_transfers[
                                s.curr_pending_start %
                                max_pipeline_transfers][0]
                            s.curr_pending_start <<= s.curr_pending_start + 1
                            s.buf.address <<= pending_transfers[
                                s.curr_pending_start %
                                max_pipeline_transfers][2]
                            s.buf.wen <<= pending_transfers[
                                s.curr_pending_start %
                                max_pipeline_transfers][1]
                            s.buf.datain <<= pending_transfers[
                                s.curr_pending_start %
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
                        elif (s.state == WAIT_WRITING):
                            s.latency_countdown <<= s.latency_countdown - 1
                            if (s.latency_countdown == 0):
                                s.state <<= DONE_WRITE
                        elif (s.state == DONE_WRITE):
                            s.state <<= INIT

            @update
            def upblk0():
                s.avalon_readdata @= s.buf.dataout
                if (pipelined):
                    num_pending_transfers = s.curr_pending_end - \
                        s.curr_pending_start
                    if (s.avalon_read or s.avalon_write) and \
                       (num_pending_transfers == max_pipeline_transfers):
                        s.avalon_waitrequest @= 1
                    else:
                        s.avalon_waitrequest @= 0
                else:
                    if ((s.state == INIT) and (s.avalon_read == 0) and
                            (s.avalon_write == 0)) \
                            or (s.state == DONE_READ) \
                            or (s.state == DONE_WRITE):
                        s.avalon_waitrequest @= 0
                    else:
                        s.avalon_waitrequest @= 1


class Buffer(Component):
    """" Implements and initializes a buffer.
    """
    def construct(s, datawidth=8, length=1, startaddr=0,
                  preload_vector=[], keepdata=True, sim=False, fast_gen=False):
        """ Constructor for Buffer

         :param datawidth: Bit-width of input, output data
         :param startaddr: Useful if there are many buffers
         :param length: Number of values in the buffer
         :param preload_vector: Initial buffer data
         :param keepdata: Store data in a way that makes debugging easier
         :param sim: Whether to skip synthesis
        """
        addrwidth = math.ceil(math.log(length + startaddr, 2))
        utils.AddInPort(s, datawidth, "datain")
        utils.AddInPort(s, addrwidth, "address")
        utils.AddInPort(s, 1, "wen")
        utils.AddOutPort(s, datawidth, "dataout")

        if (fast_gen):
            s.dataout //= 0
            return

        if (keepdata):
            s.data = Wire(length * datawidth)
        s.waddress = Wire(math.ceil(math.log(datawidth * (length + startaddr),
                                             2)))
        connect(s.waddress[0:addrwidth], s.address)
        s.waddress[addrwidth:math.ceil(math.log(datawidth * length, 2))] //= 0
        for i in range(length):
            if i < len(preload_vector):
                preload_value = preload_vector[i]
            else:
                preload_value = 0
            val = BufferValue(datawidth, addrwidth, i + startaddr,
                              preload_value,
                              sim)
            setattr(s, "V" + str(i), val)
            val.datain //= s.datain
            val.wen //= s.wen
            val.address //= s.address
            if (keepdata):
                connect(s.data[(i * datawidth):((i + 1) * datawidth)],
                        val.dataout)

        if keepdata:
            @update
            def upblk0():
                if ((s.waddress - startaddr)) <= (length-1):
                    s.dataout @= s.data[(s.waddress - startaddr) * datawidth:
                                        (s.waddress - startaddr) *
                                        datawidth + datawidth]
                else:
                    s.dataout @= 0
        else:
            @update
            def upblk1():
                if ((s.waddress - startaddr)) <= (length - 1):
                    currval = getattr(s, "V" +
                                      str(int(s.waddress - startaddr)))
                    s.dataout @= currval.dataout
                else:
                    s.dataout @= 0


class BufferValue(Component):
    """" This module implements a single value in a buffer
    """
    def construct(s, datawidth=8, addrwidth=8, addr=0, preload_value=0,
                  sim=False):
        """ Constructor for BufferValue

         :param datawidth: Bit-width of input, output data
         :param addrwidth: Bit-width of address
         :param preload_value: Initial value of buffer entry.
         :param addr: Address of this value
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
    """
    def construct(s, input_width=8, weight_width=8, sum_width=32, sim=False,
                  register_input=True):
        """ Constructor for Register

         :param input_width: Bit-width of register
         :param weight_width: Bit-width of register
         :param sum_width: Bit-width of register
         :param sim: Whether to skip synthesis
        """
        assert(sum_width >= input_width)
        assert(sum_width >= weight_width)
        utils.AddInPort(s, input_width, "input_in")
        utils.AddInPort(s, 1, "input_en")
        utils.AddOutPort(s, input_width, "input_out")
        s.input_reg = Register(input_width, sim=sim)
        s.input_reg.input_data //= s.input_in
        s.input_reg.ena //= s.input_en
        s.input_out //= s.input_reg.output_data

        utils.AddInPort(s, weight_width, "weight_in")
        utils.AddInPort(s, 1, "weight_en")
        utils.AddOutPort(s, weight_width, "weight_out")
        s.weight_reg = Register(weight_width, sim=sim)
        s.weight_reg.input_data //= s.weight_in
        s.weight_reg.ena //= s.weight_en
        s.weight_out //= s.weight_reg.output_data

        utils.AddInPort(s, sum_width, "sum_in")
        utils.AddInPort(s, 1, "acc_en")
        utils.AddOutPort(s, sum_width, "sum_out")
        s.sum_reg = Register(sum_width, sim=sim)
        s.sum_reg.ena //= 1

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
                s.sum_out @= s.sum_reg.output_data + \
                    s.input_in_w * s.weight_in_w
                s.sum_reg.input_data @= s.sum_reg.output_data + \
                    s.input_in_w * s.weight_in_w
            else:
                s.sum_out @= s.sum_in + s.input_out_w * s.weight_out_w
                s.sum_reg.input_data @= s.input_in_w * s.weight_in_w


class MACWrapper(Component):
    """" This module wraps several instantiations of MAC.
    """
    def construct(s, count=1, input_width=1, weight_width=1, sum_width=32,
                  sim=False, register_input=True):
        """ Constructor for MACWrapper

         :param count: Number of activation functions to instantiate
         :param input_width: Bit-width of register
         :param weight_width: Bit-width of register
         :param sum_width: Bit-width of register
         :param sim: Whether to skip synthesis
        """
        for i in range(count):
            curr_inst = MAC(input_width, weight_width, sum_width, sim,
                            register_input)
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
    def construct(s, proj_specs, sim=False, register_input=True,
                  fast_gen=False):
        """ Constructor for MLB

         :param proj_spec: Dictionary describing projection of computations
                            onto ML block
         :param sim: Whether to skip synthesis
         :param register_input: Whether to register the input value
         :param fast_gen: If true, create empty module (to speed up pymtl gen)
        """
        if ["inner_projection"] in proj_specs:
            proj_specs = [proj_specs]
        MAC_datatypes = ['W', 'I', 'O']

        # Calculate required MLB interface widths and print information
        # proj_specs = [proj_specs[1],proj_specs[1]]
        projs = [proj_spec["inner_projection"] for proj_spec in proj_specs]
        MAC_counts = [utils.get_mlb_count(proj) for proj in projs]
        bus_counts = {dtype: [utils.get_proj_stream_count(proj, dtype)
                              for proj in projs]
                      for dtype in MAC_datatypes}
        data_widths = {dtype: [proj_spec['stream_info'][dtype]
                               for proj_spec in proj_specs]
                       for dtype in MAC_datatypes}
        bus_widths = {
            dtype: [bus_count * data_width
                    for (bus_count, data_width) in zip(bus_counts[dtype],
                                                       data_widths[dtype])]
            for dtype in MAC_datatypes}

        # Add input and output ports
        utils.AddInPort(s, max(bus_widths['W']), "W_IN")
        utils.AddOutPort(s, max(bus_widths['W']), "W_OUT")
        utils.AddInPort(s, 1, "W_EN")
        utils.AddInPort(s, max(bus_widths['I']), "I_IN")
        utils.AddOutPort(s, max(bus_widths['I']), "I_OUT")
        utils.AddInPort(s, 1, "I_EN")
        utils.AddInPort(s, 1, "ACC_EN")
        utils.AddInPort(s, max(bus_widths['O']), "O_IN")
        utils.AddOutPort(s, max(bus_widths['O']), "O_OUT")
        s.sel = InPort(math.ceil(math.log(max(len(proj_specs), 2), 2)))
        utils.tie_off_port(s, s.sel)
        if (fast_gen):
            s.W_OUT //= 0
            s.I_OUT //= 0
            s.O_OUT //= 0
            return

        # Instantiate MACs, IOs
        s.mac_modules = MACWrapper(max(MAC_counts),
                                   max(data_widths['I']),
                                   max(data_widths['W']),
                                   max(data_widths['O']), sim, register_input)

        # Instantiate interconnects
        weight_interconnects = []
        input_interconnects = []
        output_ps_interconnects = []
        output_interconnects = []
        for i in range(len(proj_specs)):
            if (i > 0):
                newname = proj_specs[i].get("name", i)
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
            setattr(s, "weight_interconnect" + newname, weight_interconnect)
            weight_interconnects += [weight_interconnect]
            input_interconnect = module_classes.InputInterconnect(
                buffer_width=bus_widths['I'][i],
                mlb_width_used=data_widths['I'][i],
                num_mlbs=max(MAC_counts),
                projection=projs[i], sim=sim)
            setattr(s, "input_interconnect" + newname, input_interconnect)
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
            setattr(s, "output_ps_interconnect" + newname,
                    output_ps_interconnect)
            output_interconnect = module_classes.MergeBusses(
                in_width=data_widths['O'][i],
                num_ins=max(bus_counts['O']),
                out_width=bus_widths['O'][i],
                sim=sim)
            output_interconnects += [output_interconnect]
            setattr(s, "output_interconnect" + newname, output_interconnect)

        # Connect between interconnects, MACs and top level
        for i in range(len(proj_specs)):
            weight_interconnect = weight_interconnects[i]
            utils.connect_ports_by_name(s.mac_modules, r"weight_out_(\d+)",
                                        weight_interconnect,
                                        r"inputs_from_mlb_(\d+)")
            utils.connect_inst_ports_by_name(s, "W_IN", weight_interconnect,
                                             "inputs_from_buffer")
        utils.mux_ports_by_name(s, weight_interconnects,
                                r"outputs_to_mlb_(\d+)", s.mac_modules,
                                r"weight_in_(\d+)", insel=s.sel, sim=True)
        utils.mux_inst_ports_by_name(s, "W_OUT", weight_interconnects,
                                     r"outputs_to_buffer_(\d+)", insel=s.sel,
                                     sim=True)
        for i in range(len(proj_specs)):
            input_interconnect = input_interconnects[i]
            utils.connect_ports_by_name(s.mac_modules, r"input_out_(\d+)",
                                        input_interconnect,
                                        r"inputs_from_mlb_(\d+)")
            utils.connect_inst_ports_by_name(s, "I_IN", input_interconnect,
                                             "inputs_from_buffer")
        utils.mux_ports_by_name(s, input_interconnects,
                                r"outputs_to_mlb_(\d+)", s.mac_modules,
                                r"input_in_(\d+)", insel=s.sel, sim=True)
        utils.mux_inst_ports_by_name(s, "I_OUT", input_interconnects,
                                     r"outputs_to_buffer_(\d+)", insel=s.sel,
                                     sim=True)
        for i in range(len(proj_specs)):
            output_ps_interconnect = output_ps_interconnects[i]
            output_interconnect = output_interconnects[i]
            utils.connect_ports_by_name(s.mac_modules, r"sum_out_(\d+)",
                                        output_ps_interconnect,
                                        r"inputs_from_mlb_(\d+)")
            utils.connect_ports_by_name(output_ps_interconnect,
                                        r"outputs_to_afs_(\d+)",
                                        output_interconnect, r"input_(\d+)")
            utils.connect_inst_ports_by_name(s, "O_IN",
                                             output_ps_interconnect,
                                             "ps_inputs_from_buffer")
        utils.mux_ports_by_name(s, output_ps_interconnects,
                                r"outputs_to_mlb_(\d+)", s.mac_modules,
                                r"sum_in_(\d+)", insel=s.sel, sim=True)
        utils.mux_inst_ports_by_name(s, "O_OUT", output_interconnects,
                                     r"output_(\d+)", insel=s.sel, sim=True)
        utils.connect_inst_ports_by_name(s, "W_EN", s.mac_modules,
                                         "weight_en")
        utils.connect_inst_ports_by_name(s, "I_EN", s.mac_modules, "input_en")
        utils.connect_inst_ports_by_name(s, "ACC_EN", s.mac_modules, "acc_en")
