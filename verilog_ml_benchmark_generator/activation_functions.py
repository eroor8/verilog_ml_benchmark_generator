from pymtl3 import InPort, Component, OutPort, Wire, update, update_ff
import math
import utils
import module_helper_classes
il = 1


class RELU(Component):
    """" This class implements a RELU function
         clipped RELU: fout = (fin < 0)? 0 : fin
         Inputs and outputs are fixed point, with any widths
    """
    def construct(s, input_width=1, output_width=1, registered=False,
                  qin=0, qout=0, params={}):
        s.activation_function_in = InPort(input_width)
        s.activation_function_out = OutPort(output_width)
        assert(qin < input_width)
        assert(qout < output_width)
        i_in = input_width - qin
        i_out = output_width - qout
        pad_frac = qout - qin
        rm_frac = max(0, qin - qout)
        pad_int = i_out - i_in

        copy_width = min(qin, qout) + min(i_in, i_out)
        s.copied_bits = Wire(copy_width)
        if (pad_frac > 0):
            s.frac_padding = Wire(pad_frac)
        if (pad_int > 0):
            s.int_padding = Wire(pad_int)
        maxval = 2**(min(copy_width + rm_frac, input_width-1)) - 1
        if registered:
            @update_ff
            def upblk_ff():
                if s.reset:
                    s.copied_bits <<= 0
                else:
                    if (s.activation_function_in > maxval):
                        s.copied_bits <<= 0
                    else:
                        s.copied_bits <<= \
                            s.activation_function_in[rm_frac:
                                                     copy_width+rm_frac]
        else:
            @update
            def upblk0():
                if (s.activation_function_in > maxval):
                    s.copied_bits @= 0
                else:
                    s.copied_bits @= \
                        s.activation_function_in[rm_frac:copy_width+rm_frac]

        if (pad_frac > 0):
            s.activation_function_out[0:pad_frac] //= 0
            s.activation_function_out[pad_frac:
                                      copy_width+pad_frac] //= s.copied_bits
            if (pad_int > 0):
                s.activation_function_out[copy_width+pad_frac:
                                          output_width] //= 0
        else:
            s.activation_function_out[0:copy_width] //= s.copied_bits
            if (pad_int > 0):
                s.activation_function_out[copy_width:output_width] //= 0
        utils.tie_off_clk_reset(s)


class CLIPPED_RELU(Component):
    """" This class implements a clipped RELU function
         clipped RELU: fout = (fin < 0)? 0 : (fin > ceiling)? ceiling : fin
         Inputs and outputs are fixed point, with any widths
    """
    def construct(s, input_width=1, output_width=1, registered=False,
                  qin=0, qout=0, params={'ceil': 2}):
        s.activation_function_in = InPort(input_width)
        s.activation_function_out = OutPort(output_width)
        assert(qin < input_width)
        assert(qout < output_width)
        i_in = input_width - qin
        i_out = output_width - qout

        pad_frac = qout - qin
        rm_frac = max(0, qin - qout)
        pad_int = i_out - i_in

        copy_width = min(qin, qout) + min(i_in, i_out)
        s.copied_bits = Wire(copy_width)
        if (pad_frac > 0):
            s.frac_padding = Wire(pad_frac)
        if (pad_int > 0):
            s.int_padding = Wire(pad_int)
        maxval = 2**(min(copy_width + rm_frac, input_width-1)) - 1
        ceil_qin = params["ceil"] * (2**qin)
        ceil_qout = params["ceil"] * (2**(qout-max(pad_frac, 0)))
        if registered:
            @update_ff
            def upblk_ff():
                if s.reset:
                    s.copied_bits <<= 0
                else:
                    if (s.activation_function_in > maxval):
                        s.copied_bits <<= 0
                    elif (s.activation_function_in > ceil_qin):
                        s.copied_bits <<= ceil_qout
                    else:
                        s.copied_bits <<= \
                            s.activation_function_in[rm_frac:
                                                     copy_width+rm_frac]
        else:
            @update
            def upblk0():
                if (s.activation_function_in > maxval):
                    s.copied_bits @= 0
                elif (s.activation_function_in > ceil_qin):
                    s.copied_bits @= ceil_qout
                else:
                    s.copied_bits @= \
                        s.activation_function_in[rm_frac:copy_width+rm_frac]

        if (pad_frac > 0):
            s.activation_function_out[0:pad_frac] //= 0
            s.activation_function_out[pad_frac:copy_width+pad_frac] //= \
                s.copied_bits
            if (pad_int > 0):
                s.activation_function_out[copy_width+pad_frac:
                                          output_width] //= 0
        else:
            s.activation_function_out[0:copy_width] //= s.copied_bits
            if (pad_int > 0):
                s.activation_function_out[copy_width:output_width] //= 0
        utils.tie_off_clk_reset(s)


class LEAKY_RELU(Component):
    """" This class implements a leaky relu function
             fout = (fin < 0)? x*fin : fin
         Inputs and outputs are fixed point, with any widths
    """
    def construct(s, input_width=1, output_width=1, registered=False,
                  qin=0, qout=0, params={'x': 3}):
        s.activation_function_in = InPort(input_width)
        s.activation_function_out_wire = Wire(output_width)
        s.activation_function_out = OutPort(output_width)
        assert(qin < input_width)
        assert(qout < output_width)
        i_in = input_width - qin
        i_out = output_width - qout

        rm_frac = max(0, qin - qout)
        pad_int = i_out - i_in

        copy_width = min(qin, qout) + min(i_in, i_out)
        s.copied_bits = Wire(copy_width)
        if (pad_int > 0):
            s.int_padding = Wire(pad_int)
            neg_padding = 2**max(pad_int, 1) - 1
        maxval = 2**(min(copy_width + rm_frac, input_width-1)) - 1
        x = params['x']

        @update
        def upblk0():
            if (s.activation_function_in > maxval):
                copy_width1 = max(min(copy_width, input_width-rm_frac-x),
                                  0)
                if (copy_width1 > 0):
                    s.copied_bits[0:copy_width1] @= \
                        s.activation_function_in[rm_frac+x:
                                                 copy_width1+rm_frac+x]
                if (copy_width > copy_width1):
                    s.copied_bits[copy_width1:copy_width] @= \
                        2**(copy_width-copy_width1)-1
            else:
                s.copied_bits @= \
                    s.activation_function_in[rm_frac:copy_width + rm_frac]
        if (pad_int > 0):
            @update
            def upblk_int():
                if (s.activation_function_in[input_width-1] == 1):
                    s.int_padding @= neg_padding
                else:
                    s.int_padding @= 0

        pad_frac = qout - qin
        start_copy_idx = max(0, pad_frac)
        s.activation_function_out_wire[start_copy_idx:
                                       copy_width+start_copy_idx] //= \
            s.copied_bits
        if (pad_frac > 0):
            s.activation_function_out_wire[0:pad_frac] //= 0
        if (pad_int > 0):
            s.activation_function_out_wire[copy_width+start_copy_idx:
                                           output_width] //= s.int_padding

        if (registered):
            @update_ff
            def upblk_out_ff():
                s.activation_function_out <<= s.activation_function_out_wire
        else:
            s.activation_function_out //= s.activation_function_out_wire

        utils.tie_off_clk_reset(s)


class NONE(Component):
    """" This class implements no function (just a wire).
             fout = fin
         Inputs and outputs are fixed point, with any widths
    """
    def construct(s, input_width=1, output_width=1, registered=False,
                  qin=0, qout=0, params={}):
        s.activation_function_in = InPort(input_width)
        s.activation_function_out = OutPort(output_width)
        assert(qin < input_width)
        assert(qout < output_width)
        i_in = input_width - qin
        i_out = output_width - qout

        rm_frac = max(0, qin - qout)
        pad_int = i_out - i_in

        copy_width = min(qin, qout) + min(i_in, i_out)
        s.copied_bits = Wire(copy_width)
        if (pad_int > 0):
            s.int_padding = Wire(pad_int)
            neg_padding = 2**max(pad_int, 1) - 1

        if registered:
            @update_ff
            def upblk0():
                if s.reset:
                    s.copied_bits <<= 0
                else:
                    s.copied_bits <<= \
                        s.activation_function_in[rm_frac:copy_width + rm_frac]
            if (pad_int > 0):
                @update_ff
                def upblk_int_ff():
                    if s.reset:
                        s.int_padding <<= 0
                    else:
                        if (s.activation_function_in[input_width-1] == 1):
                            s.int_padding <<= neg_padding
                        else:
                            s.int_padding <<= 0
        else:
            @update
            def upblk_comb():
                s.copied_bits @= s.activation_function_in[rm_frac:
                                                          copy_width + rm_frac]
            if (pad_int > 0):
                @update
                def upblk_int():
                    if (s.activation_function_in[input_width-1] == 1):
                        s.int_padding @= neg_padding
                    else:
                        s.int_padding @= 0

        pad_frac = qout - qin
        if (pad_frac > 0):
            s.activation_function_out[0:pad_frac] //= 0
            s.activation_function_out[pad_frac:
                                      copy_width + pad_frac] //= s.copied_bits
            if (pad_int > 0):
                s.activation_function_out[copy_width + pad_frac:
                                          output_width] //= s.int_padding
        else:
            s.activation_function_out[0:copy_width] //= s.copied_bits
            if (pad_int > 0):
                s.activation_function_out[copy_width:output_width] //= \
                    s.int_padding
        utils.tie_off_clk_reset(s)


class SIGMOID_LUT(Component):
    """" This class implements a sigmoid function using a lookup table.
         Sigmoid function: fout = 1 / (1 + e^-fin)
         Inputs and outputs are fixed point, with any widths
    """
    def construct(s, input_width=1, output_width=1, registered=False,
                  qin=0, qout=0, params={}):
        s.activation_function_in = InPort(input_width)
        s.activation_function_out = OutPort(output_width)
        signed_vals = [utils.Qx_to_int(i, qin, input_width)
                       for i in range(2**(input_width))]
        sigmoid_vals = [(2**qout) / (1 + math.exp(-signed_vals[i]))
                        for i in range(2**(input_width))]
        s.bufi = module_helper_classes.Buffer(output_width,
                                              2**input_width,
                                              startaddr=0,
                                              preload_vector=sigmoid_vals,
                                              sim=False, fast_gen=False)
        s.bufi.address //= s.activation_function_in
        s.bufi.wen //= 0
        s.bufi.datain //= 0
        s.activation_function_out //= s.bufi.dataout
        utils.tie_off_clk_reset(s)


class ELU_LUT(Component):
    """" This class implements a sigmoid function using a lookup table.
         Sigmoid function: fout = alpha *  (e^fin - 1) if (fin < 0) else fin
         Inputs and outputs are fixed point, with any widths
    """
    def construct(s, input_width=1, output_width=1, registered=False,
                  qin=0, qout=0, params={}):
        s.activation_function_in = InPort(input_width)
        s.activation_function_out = OutPort(output_width)
        signed_vals = [utils.Qx_to_int(i, qin, input_width)
                       for i in range(2**(input_width))]
        sigmoid_vals = [params["alpha"] * (math.exp(signed_vals[i])-1)
                        if (signed_vals[i] < 0) else signed_vals[i]
                        for i in range(2**(input_width))]
        out_vals = [utils.int_to_Qx(sigmoid_vals[i], qout, output_width)
                    for i in range(2**(input_width))]
        s.bufi = module_helper_classes.Buffer(output_width,
                                              2**input_width,
                                              startaddr=0,
                                              preload_vector=out_vals,
                                              sim=False, fast_gen=False)
        s.bufi.address //= s.activation_function_in
        s.bufi.wen //= 0
        s.bufi.datain //= 0
        s.activation_function_out //= s.bufi.dataout
        utils.tie_off_clk_reset(s)


class SELU_LUT(Component):
    """" This class implements a sigmoid function using a lookup table.
         Sigmoid function: fout = alpha *  (e^fin - 1) if (fin < 0) else fin
         Inputs and outputs are fixed point, with any widths
    """
    def construct(s, input_width=1, output_width=1, registered=False,
                  qin=0, qout=0, params={}):
        s.activation_function_in = InPort(input_width)
        s.activation_function_out = OutPort(output_width)
        signed_vals = [utils.Qx_to_int(i, qin, input_width)
                       for i in range(2**(input_width))]
        sigmoid_vals = [params["alpha"] * params["scale"] *
                        (math.exp(signed_vals[i])-1) if (signed_vals[i] < 0)
                        else signed_vals[i] * params["scale"]
                        for i in range(2**(input_width))]
        out_vals = [utils.int_to_Qx(sigmoid_vals[i], qout, output_width)
                    for i in range(2**(input_width))]
        s.bufi = module_helper_classes.Buffer(output_width,
                                              2**input_width,
                                              startaddr=0,
                                              preload_vector=out_vals,
                                              sim=False, fast_gen=False)
        s.bufi.address //= s.activation_function_in
        s.bufi.wen //= 0
        s.bufi.datain //= 0
        s.activation_function_out //= s.bufi.dataout
        utils.tie_off_clk_reset(s)


class TANH_LUT(Component):
    """" This class implements a tanh function using a lookup table.
         Tanh function: fout = 1 / (1 + e^-fin)
         Inputs and outputs are fixed point, with any widths
    """
    def construct(s, input_width=1, output_width=1, registered=False,
                  qin=0, qout=0, params={}):
        s.activation_function_in = InPort(input_width)
        s.activation_function_out = OutPort(output_width)
        signed_vals = [utils.Qx_to_int(i, qin, input_width)
                       for i in range(2**(input_width))]
        sigmoid_vals = [math.tanh(signed_vals[i])
                        for i in range(2**(input_width))]
        out_vals = [utils.int_to_Qx(sigmoid_vals[i], qout, output_width)
                    for i in range(2**(input_width))]
        s.bufi = module_helper_classes.Buffer(output_width, 2**input_width,
                                              startaddr=0,
                                              preload_vector=out_vals,
                                              sim=False, fast_gen=False)
        s.bufi.address //= s.activation_function_in

        s.bufi.wen //= 0
        s.bufi.datain //= 0
        s.activation_function_out //= s.bufi.dataout
        utils.tie_off_clk_reset(s)


class GENERIC_LUT(Component):
    """" This class implements any function using a LUT. LUT values are passed
         in as a parameter.
             fout = lut[fin]
         Inputs and outputs are fixed point, with any widths
    """
    def construct(s, input_width=1, output_width=1, registered=False,
                  qin=0, qout=0, params={'lut': []}):
        s.activation_function_in = InPort(input_width)
        s.activation_function_out = OutPort(output_width)
        stored_vals = [(2**qout)*i for i in params['lut']]
        s.bufi = module_helper_classes.Buffer(output_width, 2**input_width,
                                              startaddr=0,
                                              preload_vector=stored_vals,
                                              sim=False, fast_gen=False)

        s.bufi.address //= s.activation_function_in
        s.bufi.wen //= 0
        s.bufi.datain //= 0
        s.activation_function_out //= s.bufi.dataout


def RELU_SW(input_act, input_width, output_width=0, qin=0, qout=0, params={}):
    """" Python version of relu activation function
         (for pyMTL simulation validation)
    """
    assert(qin < input_width)
    signed_val = utils.Qx_to_int(input_act, qin, input_width)
    if ((signed_val < 0) or (signed_val >= 2**output_width) or
            (-signed_val >= 2**output_width)):
        out_val = 0
    else:
        out_val = signed_val * (2**qout)
    return math.floor(out_val)


def CLIPPED_RELU_SW(input_act, input_width, output_width=0, qin=0, qout=0,
                    params={'ceil': 2}):
    """" Python version of clipped relu activation function
         (for pyMTL simulation validation)
    """
    assert(qin < input_width)
    signed_val = utils.Qx_to_int(input_act, qin, input_width)
    if ((signed_val < 0) or (signed_val >= 2**output_width) or
            (-signed_val >= 2**output_width)):
        out_val = 0
    elif (signed_val > params['ceil']):
        out_val = params['ceil'] * (2**qout)
    else:
        out_val = signed_val * (2**qout)
    return math.floor(out_val)


def LEAKY_RELU_SW(input_act, input_width, output_width=0, qin=0, qout=0,
                  params={'x': 3}):
    """" Python version of leaky relu activation function
         (for pyMTL simulation validation)
    """
    assert(qin < input_width)
    signed_val = utils.Qx_to_int(input_act, qin, input_width)
    if ((signed_val < 0) or (signed_val >= 2**output_width) or
            (-signed_val >= 2**output_width)):
        out_val = signed_val / (2**params['x'])
        out_val = utils.int_to_Qx(out_val, qout, output_width)
    else:
        out_val = utils.int_to_Qx(signed_val, qout, output_width)
    return math.floor(out_val)


def NONE_SW(input_act, input_width, output_width=0, qin=0, qout=0, params={}):
    """" Python version of no activation function (just a wire)
         (for pyMTL simulation validation)
    """
    assert(qin < input_width)
    signed_val = utils.Qx_to_int(input_act, qin, input_width)
    retval = utils.int_to_Qx(signed_val, qout, output_width)
    return retval


def GENERIC_LUT_SW(input_act, input_width, output_width=0, qin=0, qout=0,
                   params={'lut': []}):
    """" Python version of LUT activation function
         (for pyMTL simulation validation)
    """
    assert(qin < input_width)
    retval = utils.int_to_Qx(params['lut'][input_act], qout, output_width)
    return retval


def TANH_LUT_SW(input_act, input_width, output_width=0, qin=0, qout=0,
                params={}):
    """" Python version of tanh activation function
         (for pyMTL simulation validation)
    """
    assert(qin < input_width)
    signed_val = utils.Qx_to_int(input_act, qin, input_width)
    out_int = math.tanh(signed_val)
    out_q = utils.int_to_Qx(out_int, qout, output_width)
    return math.floor(out_q)


def SIGMOID_LUT_SW(input_act, input_width, output_width=0, qin=0,
                   qout=0, params={}):
    """" Python version of sigmoid activation function
         (for pyMTL simulation validation)
    """
    assert(qin < input_width)
    signed_val = utils.Qx_to_int(input_act, qin, input_width)
    out_int = 1 / (1 + math.exp(-signed_val))
    out_q = out_int * (2**qout)
    return out_q


def ELU_LUT_SW(input_act, input_width, output_width=0, qin=0,
               qout=0, params={}):
    """" Python version of sigmoid activation function
         (for pyMTL simulation validation)
    """
    assert(qin < input_width)
    signed_val = utils.Qx_to_int(input_act, qin, input_width)
    if (signed_val < 0):
        out_int = params["alpha"] * (math.exp(signed_val)-1)
    else:
        out_int = signed_val
    out_q = utils.int_to_Qx(out_int, qout, output_width)
    return math.floor(out_q)


def SELU_LUT_SW(input_act, input_width, output_width=0, qin=0,
                qout=0, params={}):
    """" Python version of sigmoid activation function
         (for pyMTL simulation validation)
    """
    assert(qin < input_width)
    signed_val = utils.Qx_to_int(input_act, qin, input_width)
    if (signed_val < 0):
        out_int = params["alpha"] * params["scale"] * (math.exp(signed_val)-1)
    else:
        out_int = signed_val * params["scale"]
    out_q = utils.int_to_Qx(out_int, qout, output_width)
    return math.floor(out_q)
