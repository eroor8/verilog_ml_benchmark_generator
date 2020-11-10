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
                  preload_vector=[], sim=False):
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
            connect(s.data[(i * datawidth):((i + 1) * datawidth)], val.dataout)

            
        @update
        def upblk0():
            if ((s.waddress - startaddr)) <= (length-1):
                s.dataout @= s.data[(s.waddress - startaddr) * datawidth:
                    (s.waddress - startaddr) * datawidth + datawidth]
            else:
                s.dataout @= 0
            #    if (s.wen):
            #        print("Address ok... " + str(s.waddress) + " for buf " + str(s._dsl.full_name) + " datawidth " + str(datawidth))
            #else:
            #    print("Address " + str(s.waddress) + " exceeds maximum length! " + str(length) + " for buf " + str(s._dsl.full_name) + " datawidth " + str(datawidth))

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
        s.sum_reg.ena //= s.acc_en

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
            s.sum_reg.input_data @= s.sum_reg.output_data + \
                                    s.input_in_w * s.weight_in_w
            if s.acc_en:
                s.sum_out @= s.sum_reg.output_data
            else:
                s.sum_out @= s.sum_in + s.input_out_w * s.weight_out_w

                
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
    def construct(s, proj_spec, sim=False):
        """ Constructor for MLB

         :param proj_spec: Dictionary describing projection of computations
                            onto ML block
         :type proj_spec: dict
        """
        MAC_datatypes = ['W', 'I', 'O']

        # Calculate required MLB interface widths and print information
        proj = proj_spec["inner_projection"]
        MAC_count = utils.get_mlb_count(proj)
        bus_counts = {dtype: utils.get_proj_stream_count(proj, dtype)
                                   for dtype in MAC_datatypes}
        data_widths = {dtype: proj_spec['stream_info'][dtype]
                                    for dtype in MAC_datatypes}
        bus_widths = {dtype: bus_counts[dtype] *
                                   data_widths[dtype]
                                   for dtype in MAC_datatypes}

        # Instantiate MACs, IOs
        s.mac_modules = MACWrapper(MAC_count,
                                   data_widths['I'],
                                   data_widths['W'],
                                   data_widths['O'], sim)
        utils.AddInPort(s, bus_widths['W'], "W_IN")
        utils.AddOutPort(s, bus_widths['W'], "W_OUT")
        utils.AddInPort(s, 1, "W_EN")
        utils.AddInPort(s, bus_widths['I'], "I_IN")
        utils.AddOutPort(s, bus_widths['I'], "I_OUT")
        utils.AddInPort(s, 1, "I_EN")
        utils.AddInPort(s, 1, "ACC_EN")
        utils.AddInPort(s, bus_widths['O'], "O_IN")
        utils.AddOutPort(s, bus_widths['O'], "O_OUT")

        # Instantiate interconnects
        s.weight_interconnect = module_classes.WeightInterconnect(
            buffer_width=bus_widths['W'],
            mlb_width_used=data_widths['W'],
            num_mlbs=MAC_count,
            projection=proj,
            sim=sim
        )
        s.input_interconnect = module_classes.InputInterconnect(
            buffer_width=bus_widths['I'],
            mlb_width_used=data_widths['I'],
            num_mlbs=MAC_count,
            projection=proj, sim=sim)
        s.output_ps_interconnect = module_classes.OutputPSInterconnect(
            af_width=data_widths['O'],
            mlb_width_used=data_widths['O'],
            num_afs=bus_counts['O'],
            num_mlbs=MAC_count,
            projection=proj, sim=sim,
            input_buf_width=bus_widths['O'],
            num_input_bufs=1
        )
        s.output_interconnect = module_classes.MergeBusses(
            in_width=data_widths['O'],
            num_ins=bus_counts['O'],
            out_width=bus_widths['O'],
            sim=sim)
        
        # Connect between interconnects, MACs and top level
        utils.connect_ports_by_name(s.mac_modules,
            "weight_out", s.weight_interconnect, "inputs_from_mlb")
        utils.connect_ports_by_name(s.weight_interconnect,
            "outputs_to_mlb", s.mac_modules, "weight_in")
        utils.connect_inst_ports_by_name(s,
            "W_IN", s.weight_interconnect, "inputs_from_buffer")
        utils.connect_inst_ports_by_name(s, "W_OUT", s.weight_interconnect,
            "outputs_to_buffer")
        
        utils.connect_ports_by_name(s.mac_modules,
            "input_out", s.input_interconnect, "inputs_from_mlb")
        utils.connect_ports_by_name(s.input_interconnect,
            "outputs_to_mlb", s.mac_modules, "input_in")
        utils.connect_inst_ports_by_name(s,
            "I_IN", s.input_interconnect, "inputs_from_buffer")
        utils.connect_inst_ports_by_name(s, "I_OUT", s.input_interconnect,
            "outputs_to_buffer")
        
        utils.connect_ports_by_name(s.mac_modules,
            "sum_out", s.output_ps_interconnect, "inputs_from_mlb")
        utils.connect_ports_by_name(s.output_ps_interconnect,
            "outputs_to_mlb", s.mac_modules, "sum_in")
        utils.connect_ports_by_name(s.output_ps_interconnect,
            "outputs_to_afs", s.output_interconnect, "input")
        utils.connect_inst_ports_by_name(s,
            "O_IN", s.output_ps_interconnect, "ps_inputs_from_buffer")
        utils.connect_inst_ports_by_name(s, "O_OUT", s.output_interconnect,
            "output")
        utils.connect_inst_ports_by_name(s, "W_EN", s.mac_modules, "weight_en")
        utils.connect_inst_ports_by_name(s, "I_EN", s.mac_modules, "input_en")
        utils.connect_inst_ports_by_name(s, "ACC_EN", s.mac_modules, "acc_en")

