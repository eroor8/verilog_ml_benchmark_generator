""""
PYMTL Component Classes Implementing different parts of the dataflow
ASSUMPTIONS:
- Weights are always preloaded
- Weight stationary flow
- Inputs don't require muxing

"""
from pymtl3 import InPort, Component, OutPort, connect, Wire, update, update_ff
import math
import copy
import yaml
import utils
import module_helper_classes
il = 1


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
        utils.tie_off_clk_reset(s)
        min_width = min(input_width, output_width)
        max_width = max(input_width, output_width)
        s.internal_out0 = Wire(min_width)
        s.internal_out1 = Wire(max(output_width - min_width, 1))

        if registered:
            @update_ff
            def upblk0():
                if s.reset:
                    s.internal_out0 <<= 0
                    s.internal_out1 <<= 0
                else:
                    s.internal_out0 <<= (s.activation_function_in[0:min_width]
                                         if (s.activation_function_in[
                                                 input_width - 1] == 0) else 0)
        else:
            @update
            def upblk3():
                s.internal_out0 @= (s.activation_function_in[0:min_width]
                                    if (s.activation_function_in[
                                            input_width - 1] == 0) else 0)
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
    def construct(s, count=1, function="RELU", input_width=1,
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
            assert (function == "RELU"), \
                "NON-RELU functions not currently created automatically." + \
                " If required, change the RELU module in the generated .v."
            curr_inst = RELU(input_width, output_width, registered)
            setattr(s, function + '_inst_' + str(i), curr_inst)

            for port in curr_inst.get_input_value_ports():
                utils.connect_in_to_top(s, port, port._dsl.my_name + "_" +
                                        str(i))
            for port in curr_inst.get_output_value_ports():
                utils.connect_out_to_top(s, port, port._dsl.my_name + "_" +
                                         str(i))


class MLB_Wrapper(Component):
    """" This module represents some block, as specified in a json description
         The point is to provide the interface of the actual hardware block
         so that it can be instantiated in other modules, and a simulation
         model.
         Simulation type specifies which sim model to use.
         Otherwise, the contents of this block is
         empty, and all outputs are tied to zero.
         One port is added for each port listed in the json port list.
         The module will end up being named "HWB_Sim__<block_name>"
    """
    def construct(s, spec={}, projs={}, sim=True, fast_gen=False):
        """ Constructor for HWB

         :param spec: Dictionary describing hardware block ports and
                      functionality
         :type spec: dict
         :param proj: Dictionary describing projection of computations
                            onto ML block
         :type proj: dict
        """
        copy_projs = copy.deepcopy(projs)
        ports_by_type = {}
        special_outs = []
        if "simulation_model" in spec:
            special_outs = ["DATA", "W", "I", "O", "AVALON_READDATA",
                            "AVALON_WAITREQUEST", "AVALON_READDATAVALID"]

        for port in spec['ports']:
            if not port["type"] in ("CLK", "RESET"):
                if (port["direction"] == "in"):
                    newport = utils.AddInPort(s, port["width"], port["name"])
                else:
                    if port["name"] not in s.__dict__.keys():
                        newport = utils.AddOutPort(s, port["width"],
                                                   port["name"])
                        if port["type"] not in special_outs:
                            newport //= newport._dsl.Type(0)
                typename = port["type"] + "_" + port["direction"]
                if typename in ports_by_type:
                    ports_by_type[typename] += [[port, newport]]
                else:
                    ports_by_type[typename] = [[port, newport]]
        s._dsl.args = [spec.get('block_name', "unnamed")]

        assert(spec.get("simulation_model", "") == "MLB" or
               spec.get("simulation_model", "") == "ML_Block")
        assert len(projs) > 0, "Empty projection specification"
        for req_port in ["W_in", "W_out", "I_in", "I_out",
                         "O_in", "O_out"]:
            assert req_port in ports_by_type, \
                "To run simulation, you need port of type " + req_port
        for req_port in ["W_EN_in", "I_EN_in", "ACC_EN_in"]:
            assert req_port in ports_by_type, \
                "To run simulation, you need port of type " + req_port + \
                " in definition of " + spec["block_name"]
            assert len(ports_by_type[req_port]) == 1

        inner_projs = [proj['inner_projection'] for proj in copy_projs]
        s.sim_model = module_helper_classes.MLB(copy_projs, sim=sim,
                                                fast_gen=fast_gen)
        MAC_datatypes = ['W', 'I', 'O']
        inner_bus_counts = {
            dtype: [utils.get_proj_stream_count(inner_proj, dtype)
                    for inner_proj in inner_projs]
            for dtype in MAC_datatypes}
        inner_bus_widths = {dtype: [inner_bus_count *
                                    proj['data_widths'][dtype]
                                    for (proj, inner_bus_count) in
                                    zip(copy_projs, inner_bus_counts[dtype])]
                            for dtype in MAC_datatypes}
        assert(ports_by_type["I_out"][0][0]['width'] ==
               ports_by_type["I_in"][0][0]['width']), \
            "Input and output stream widths should be equal (MLB I ports)"
        i_out = ports_by_type["I_out"][0][1]
        assert(ports_by_type["W_out"][0][0]['width'] ==
               ports_by_type["W_in"][0][0]['width']), \
            "Input and output stream widths should be equal (MLB W ports)"
        assert(ports_by_type["O_out"][0][0]['width'] ==
               ports_by_type["O_in"][0][0]['width']), \
            "Input and output stream widths should be equal (MLB O ports)"
        o_out = ports_by_type["O_out"][0][1]
        assert(max(inner_bus_widths['W']) <=
               ports_by_type["W_in"][0][0]['width']), \
            "Specified MLB port width not wide enough for desired mapping"
        w_in = ports_by_type["W_in"][0][1]
        assert(max(inner_bus_widths['I']) <=
               ports_by_type["I_in"][0][0]['width']), \
            "Specified MLB port width not wide enough for desired mappings"
        i_in = ports_by_type["I_in"][0][1]
        assert(max(inner_bus_widths['O']) <=
               ports_by_type["O_in"][0][0]['width']), \
            "Specified MLB port width not wide enough for desired mappings"
        o_in = ports_by_type["O_in"][0][1]
        connect(ports_by_type["W_EN_in"][0][1], s.sim_model.W_EN)
        connect(ports_by_type["W_out"][0][1][0:max(inner_bus_widths['W'])],
                s.sim_model.W_OUT)
        connect(ports_by_type["I_EN_in"][0][1], s.sim_model.I_EN)
        connect(ports_by_type["ACC_EN_in"][0][1], s.sim_model.ACC_EN)
        if ("MODE_in" in ports_by_type):
            connect(ports_by_type["MODE_in"][0][1], s.sim_model.sel)
        else:
            s.sim_model.sel //= 0
        inner_projs = [proj['inner_projection'] for proj in projs]
        connect(i_in[0:max(inner_bus_widths['I'])], s.sim_model.I_IN)
        connect(i_out[0:max(inner_bus_widths['I'])], s.sim_model.I_OUT)
        connect(w_in[0:max(inner_bus_widths['W'])], s.sim_model.W_IN)
        connect(o_in[0:max(inner_bus_widths['O'])], s.sim_model.O_IN)
        connect(o_out[0:max(inner_bus_widths['O'])], s.sim_model.O_OUT)


class MLB_Wrapper_added_logic(Component):
    """" This module represents some block, as specified in a json description
         The point is to provide the interface of the actual hardware block
         so that it can be instantiated in other modules, and a simulation
         model.
         Simulation type specifies which sim model to use.
         Otherwise, the contents of this block is
         empty, and all outputs are tied to zero.
         One port is added for each port listed in the json port list.
         The module will end up being named "HWB_Sim__<block_name>"
    """
    def construct(s, spec={}, projs={}, fast_gen=False):
        """ Constructor for HWB

         :param spec: Dictionary describing hardware block ports and
                      functionality
         :type spec: dict
         :param proj: Dictionary describing projection of computations
                            onto ML block
         :type proj: dict
        """
        copy_projs = copy.deepcopy(projs)
        s._dsl.args = [spec.get('block_name', "unnamed")]

        inner_projs = [proj['inner_projection'] for proj in copy_projs]

        if ("access_patterns" in spec):
            spec_keys = copy.deepcopy(spec["access_patterns"])
            for ip in inner_projs:
                if (ip["RX"] > spec_keys["AP1"]):
                    utils.print_warning(il, "Adding additional soft" +
                                        "logic to perform windowing " +
                                        "of adjacent input values")
                    ip["RY"] = ip["RY"]*ip["RX"]
                    ip["RX"] = 1

                assert(ip["RY"]*ip["C"] <= spec_keys["AP2"])

                if (ip["B"]*ip["PX"]*ip["PY"] > spec_keys["AP4"]):
                    assert(spec_keys["AP4"] == 1)
                    assert (ip["B"]*ip["PX"]*ip["PY"] <= spec_keys["AP5"])
                    ip["G"] = ip["B"]*ip["PX"]*ip["PY"]*ip["G"]
                    ip["B"] = 1
                    ip["PX"] = 1
                    ip["PY"] = 1

                if (ip["E"] > spec_keys["AP3"]):
                    assert(spec_keys["AP3"] == 1)
                    assert (ip["E"] <= spec_keys["AP5"])
                    ip["G"] = ip["E"]*ip["G"]
                    ip["E"] = 1

                assert(ip["G"] <= spec_keys["AP5"])

        s.curr_inst = MLB_Wrapper(spec, copy_projs, sim=True,
                                  fast_gen=fast_gen)
        i_out_inner = None
        i_in_inner = None
        i_out_outer = None
        i_in_outer = None
        w_out_inner = None
        w_in_inner = None
        w_out_outer = None
        w_in_outer = None
        for port in spec['ports']:
            if ((port['type'] == 'C' or port['type'] == 'ADDRESS' or
                 port['type'] == 'W_EN' or port['type'] == 'I_EN' or
                 port['type'] == 'ACC_EN' or port['type'] == 'MODE')
                    and port["direction"] == "in"):
                instport = getattr(s.curr_inst, port["name"])
                new_p = utils.AddInPort(s,  port['width'], port["name"])
                instport //= new_p
            elif port['type'] not in ('CLK', 'RESET'):
                if (port['direction'] == "in"):
                    if (port['type'] == 'I'):
                        i_in_inner = getattr(s.curr_inst, port["name"])
                        i_in_width = port["width"]
                        i_in_outer = utils.AddInPort(s, port['width'],
                                                     port["name"])
                    elif (port['type'] == 'W'):
                        w_in_inner = getattr(s.curr_inst, port["name"])
                        w_in_width = port["width"]
                        w_in_outer = utils.AddInPort(s, port['width'],
                                                     port["name"])
                    else:
                        utils.connect_in_to_top(s, getattr(s.curr_inst,
                                                port["name"]), port["name"])
                else:
                    if (port['type'] == 'I'):
                        i_out_inner = getattr(s.curr_inst, port["name"])
                        i_out_outer = utils.AddOutPort(s, port['width'],
                                                       port["name"])
                    elif (port['type'] == 'W'):
                        w_out_inner = getattr(s.curr_inst, port["name"])
                        w_out_outer = utils.AddOutPort(s, port['width'],
                                                       port["name"])
                    else:
                        utils.connect_out_to_top(s, getattr(s.curr_inst,
                                                 port["name"]), port["name"])

        inner_projs = [proj['inner_projection'] for proj in projs]
        inner_projs_new = [proj['inner_projection'] for proj in copy_projs]
        if ("access_patterns" in spec):
            ip = inner_projs[0]
            ip_new = inner_projs_new[0]
            dataw = projs[0]['data_widths']['I']

            if (ip["E"] > spec_keys["AP3"]):
                reqd_ue = ip["E"]
                assert(spec_keys["AP3"] == 1)
                s.ue_in = Wire(i_in_width)
                for (urny, urnc, ug, ubb, ubx, uby, a, c) in utils.range8D(
                        ip["RY"], ip["C"], ip["G"], ip["B"], ip["PX"],
                        ip["PY"]):
                    in_chain = utils.get_overall_idx_new(
                        ip, {'RY': urny, 'C': urnc, 'PX': ubx, 'PY': uby,
                             'B': ubb, 'G': ug}, order=utils.input_order)
                    for ue in range(reqd_ue):
                        out_chain = utils.get_overall_idx_new(
                            ip_new, {'RY': urny, 'C': urnc, 'PX': ubx,
                                     'PY': uby, 'B': ubb,
                                     'G': (ug * reqd_ue + ue)},
                            order=utils.input_order)
                        s.ue_in[out_chain * dataw:
                                (out_chain + 1) * dataw] //= \
                            i_in_outer[(in_chain * dataw):
                                       (in_chain + 1) * dataw]
                i_in_outer = s.ue_in

            dataw = projs[0]['data_widths']['W']
            if (ip["B"]*ip["PX"]*ip["PY"] > spec_keys["AP4"]):
                reqd_ub = ip["B"]*ip["PX"]*ip["PY"]
                assert(spec_keys["AP4"] == 1)
                s.ub_in = Wire(w_in_width)
                for (urnx, urny, urnc, ug, ue, a, b, c) in utils.range8D(
                        ip["RX"], ip["RY"], ip["C"], ip["G"], ip["E"]):
                    in_chain = utils.get_overall_idx(
                        ip, {'RX': urnx, 'C': urnc, 'RY': urny, 'E': ue,
                             'G': ug})
                    for ub in range(ip["B"] * ip["PX"] * ip["PY"]):
                        out_chain = utils.get_overall_idx(
                            ip_new, {'RX': urnx, 'C': urnc, 'RY': urny,
                                     'E': ue, 'G': (ug * reqd_ub + ub)})
                        s.ub_in[out_chain * dataw:
                                (out_chain + 1) * dataw] //= \
                            w_in_outer[(in_chain * dataw):
                                       (in_chain + 1) * dataw]
                w_in_outer = s.ub_in

        i_in_inner //= i_in_outer
        i_out_outer //= i_out_inner
        w_in_inner //= w_in_outer
        w_out_outer //= w_out_inner


class HWB_Sim(Component):
    """" This module represents some block, as specified in a json description
         The point is to provide the interface of the actual hardware block
         so that it can be instantiated in other modules, and a simulation
         model.
         Simulation type specifies which sim model to use.
         Otherwise, the contents of this block is
         empty, and all outputs are tied to zero.
         One port is added for each port listed in the json port list.
         The module will end up being named "HWB_Sim__<block_name>"
    """
    def construct(s, spec={}, projs={}, sim=True, fast_gen=False):
        """ Constructor for HWB

         :param spec: Dictionary describing hardware block ports and
                      functionality
         :type spec: dict
         :param proj: Dictionary describing projection of computations
                            onto ML block
         :type proj: dict
        """
        ports_by_type = {}
        special_outs = []
        if "simulation_model" in spec:
            special_outs = ["DATA", "W", "I", "O", "AVALON_READDATA",
                            "AVALON_WAITREQUEST", "AVALON_READDATAVALID"]
        assert 'ports' in spec
        for port in spec['ports']:
            if not port["type"] in ("CLK", "RESET"):
                if (port["direction"] == "in"):
                    newport = utils.AddInPort(s, port["width"], port["name"])
                else:
                    if port["name"] not in s.__dict__.keys():
                        newport = utils.AddOutPort(s, port["width"],
                                                   port["name"])
                        if port["type"] not in special_outs:
                            newport //= newport._dsl.Type(0)
                typename = port["type"] + "_" + port["direction"]
                if typename in ports_by_type:
                    ports_by_type[typename] += [[port, newport]]
                else:
                    ports_by_type[typename] = [[port, newport]]
        s._dsl.args = [spec.get('block_name', "unnamed")]

        if "simulation_model" in spec:
            if spec.get("simulation_model", "") == "Buffer":
                for req in ["ADDRESS_in", "WEN_in", "DATA_in", "DATA_out"]:
                    assert req in ports_by_type, \
                        "To run simulation, you need port of type " + \
                        req + " in definition of " + spec["block_name"]
                assert len(ports_by_type["ADDRESS_in"]) == 1  # Todo
                assert (len(ports_by_type["ADDRESS_in"]) ==
                        len(ports_by_type["DATA_out"]))
                assert (len(ports_by_type["ADDRESS_in"]) ==
                        len(ports_by_type["WEN_in"]))
                assert (len(ports_by_type["ADDRESS_in"]) ==
                        len(ports_by_type["DATA_in"]))
                for buffer_inst in range(len(ports_by_type["ADDRESS_in"])):
                    assert ports_by_type["DATA_out"][buffer_inst][0]["width"] \
                        == ports_by_type["DATA_in"][buffer_inst][0]["width"]
                    assert ports_by_type["WEN_in"][buffer_inst][0]["width"] \
                        == 1
                    datalen = ports_by_type["DATA_in"][buffer_inst][0]["width"]
                    addrlen = \
                        ports_by_type["ADDRESS_in"][buffer_inst][0]["width"]
                    size = 2**addrlen
                    sim_model = module_helper_classes.Buffer(
                        datalen, size, sim=sim,
                        fast_gen=fast_gen)
                    setattr(s, "sim_model_inst" + str(buffer_inst), sim_model)
                    connect(ports_by_type["DATA_in"][buffer_inst][1],
                            sim_model.datain)
                    connect(ports_by_type["DATA_out"][buffer_inst][1],
                            sim_model.dataout)
                    connect(ports_by_type["ADDRESS_in"][buffer_inst][1],
                            sim_model.address)
                    connect(ports_by_type["WEN_in"][buffer_inst][1],
                            sim_model.wen)
            elif spec.get("simulation_model", "") == "EMIF":
                for req_port in ["AVALON_ADDRESS_in", "AVALON_READDATA_out",
                                 "AVALON_WRITEDATA_in",
                                 "AVALON_READDATAVALID_out",
                                 "AVALON_WAITREQUEST_out",
                                 "AVALON_READ_in", "AVALON_WRITE_in"]:
                    assert req_port in ports_by_type, \
                        "To run simulation, you need port type " + req_port + \
                        " in definition of " + spec["block_name"]
                    assert len(ports_by_type[req_port]) == 1
                wd_width = ports_by_type["AVALON_WRITEDATA_in"][0][0]["width"]
                ad_width = ports_by_type["AVALON_ADDRESS_in"][0][0]["width"]
                s.sim_model = module_helper_classes.EMIF(
                    datawidth=wd_width, length=2**ad_width, startaddr=0,
                    preload_vector=spec.get('fill', False),
                    pipelined=spec.get('pipelined', False),
                    max_pipeline_transfers=spec.get(
                        'max_pipeline_transfers', {}).get(
                            'max_pipeline_transfers', 4),
                    sim=True, fast_gen=fast_gen)
                connect(ports_by_type["AVALON_ADDRESS_in"][0][1],
                        s.sim_model.avalon_address)
                connect(ports_by_type["AVALON_WRITEDATA_in"][0][1],
                        s.sim_model.avalon_writedata)
                connect(s.sim_model.avalon_readdata,
                        ports_by_type["AVALON_READDATA_out"][0][1])
                connect(ports_by_type["AVALON_READ_in"][0][1],
                        s.sim_model.avalon_read,)
                connect(ports_by_type["AVALON_WRITE_in"][0][1],
                        s.sim_model.avalon_write)
                connect(s.sim_model.avalon_readdatavalid,
                        ports_by_type["AVALON_READDATAVALID_out"][0][1])
                connect(s.sim_model.avalon_waitrequest,
                        ports_by_type["AVALON_WAITREQUEST_out"][0][1])


class HWB_Wrapper(Component):
    """" This module wraps several instantiations of some specified block
         (``spec``). Input ports with datatype "C" (config) and "ADDRESS" are
         shared between all instances. All other ports are duplicated one for
         each instance on the top level, and named
         <instance_port_name>_<instance>. Clock and reset are common.
    """
    def construct(s, spec={}, count=1, name="_v1", projections={},
                  fast_gen=False):
        """ Constructor for HWB_Wrapper

         :param spec: Dictionary describing hardware block ports and
                      functionality
         :type spec: dict
         :param count: Number of blocks to instantiate
         :type count: int
         :param name: String appended to resultant module name to avoid name
                      collision bug.
         :type count: string
        """
        # Add ports shared between instances to the top level
        for i in range(count):
            if (spec.get("simulation_model", "") == "MLB" or
                    spec.get("simulation_model", "") == "ML_Block"):
                curr_inst = MLB_Wrapper_added_logic(spec, projections,
                                                    fast_gen=fast_gen)
            else:
                curr_inst = HWB_Sim(spec, projections, sim=True,
                                    fast_gen=fast_gen)
            setattr(s, spec.get('block_name', "unnamed") + '_inst_' + str(i),
                    curr_inst)
            for port in spec['ports']:
                if ((port['type'] == 'C' or port['type'] == 'ADDRESS' or
                     port['type'] == 'W_EN' or port['type'] == 'I_EN' or
                     port['type'] == 'ACC_EN' or port['type'] == 'MODE')
                        and port["direction"] == "in"):
                    instport = getattr(curr_inst, port["name"])
                    instport //= utils.AddInPort(s,  port['width'],
                                                 port["name"])
                elif port['type'] not in ('CLK', 'RESET'):
                    if (port['direction'] == "in"):
                        utils.connect_in_to_top(s, getattr(curr_inst,
                                                           port["name"]),
                                                port["name"] + "_" + str(i))
                    else:
                        utils.connect_out_to_top(s, getattr(curr_inst,
                                                            port["name"]),
                                                 port["name"] + "_" + str(i))
        utils.tie_off_clk_reset(s)


class InputBufferWrapper(Component):
    """" This module wraps several instantiations of some specified block
         (``spec``). Input ports with datatype "C" (config) and "ADDRESS" are
         shared between all instances. All other ports are duplicated one for
         each instance on the top level, and named
         <instance_port_name>_<instance>. Clock and reset are common.

         This module instantiates buffers for all the input and output
         activations.
         For each layer, these buffers are divided into input buffers and
         output buffers. There are X banks of Y input buffers and Z banks of
         T output buffers.
    """
    def construct(s, spec={}, bcounts_i=[1], bcounts_o=[1], name="_v1",
                  projections={}, mux=True, fast_gen=False, add_SR=False,
                  input_width=-1, buffer_start_idxs=[0]):
        # Add ports shared between instances to the top level
        # How many address muxes are required? URNYxURNB
        muxes = [[] for proj in projections]
        max_total_bcount = max([sum(x) for x in zip(bcounts_i, bcounts_o)])
        addr_width = 0
        addri1 = None

        # Add three addresses - two input addresses and an output address
        for port in spec['ports']:
            if (port['type'] == 'ADDRESS'):
                addr_width = port['width']
                addri1 = utils.AddInPort(s,  port['width'], port["name"])
                addri2 = utils.AddInPort(s,  port['width'], port["name"]
                                         + "_b")
                addro = utils.AddInPort(s,  port['width'], port["name"]
                                        + "_o")

        # A signal to select which layer
        layer_sel = utils.AddInPort(s, math.ceil(math.log(max(len(projections),
                                                              2), 2)),
                                    "sel")

        # Mux between two input addresses for efficient convolution.
        mux_sizes = [proj['outer_projection']['PY'] *
                     proj['inner_projection']['PY'] *
                     proj['inner_projection']['RY'] *
                     proj['outer_projection']['RY']
                     for proj in projections]
        utils.AddInPort(s, math.ceil(math.log(max(max(mux_sizes), 2), 2)),
                        "addr_sel")
        j = 0
        for proj in projections:
            k = 0
            if ((proj['inner_projection']['RY'] *
                 proj['outer_projection']['RY'] > 0)):
                for mux in range(mux_sizes[j]):
                    newmux = module_helper_classes.MUX2(
                        addr_width,
                        math.ceil(math.log(max(mux_sizes[j], 2), 2)), k)
                    setattr(s, "mux_addr" + str(j) + "_" + str(k), newmux)
                    if (mux_sizes[j] > 1):
                        newmux.sel //= \
                            s.addr_sel[0:math.ceil(math.log(max(mux_sizes[j],
                                                                2), 2))]
                    else:
                        newmux.sel //= 0
                    newmux.in0 //= addri1
                    newmux.in1 //= addri2
                    muxes[j] += [newmux]
                    k = k + 1
            j += 1
        s.omux = module_helper_classes.MUX1(addr_width)
        s.omux.in0 //= addro

        for i in range(max_total_bcount):
            curr_inst = HWB_Sim(spec, sim=True, fast_gen=fast_gen)
            setattr(s, spec.get('block_name', "unnamed") + '_inst_' + str(i),
                    curr_inst)

            for port in spec['ports']:
                if ((port['type'] == 'C' or port['type'] == 'MODE')
                        and port["direction"] == "in"):
                    instport = getattr(curr_inst, port["name"])
                    instport //= utils.AddInPort(s,  port['width'],
                                                 port["name"])
                elif (port['type'] == 'ADDRESS'):
                    instport = getattr(curr_inst, port["name"])
                    muxs = []
                    for pj in range(len(projections)):
                        buffer_idxs = utils.map_buffer_idx_to_y_idx(
                            projections[pj], spec)
                        i_order = (i - buffer_start_idxs[pj]) % \
                            max_total_bcount
                        if (i_order < bcounts_i[pj]):
                            assert i_order < len(buffer_idxs)
                            muxs += [muxes[pj][buffer_idxs[i_order]]]
                        else:
                            muxs += [s.omux]
                    # Which muxes to connect to?
                    utils.mux_ports_by_name(s, muxs, "out", curr_inst,
                                            port['name'], insel=layer_sel,
                                            sim=False, idx=str(i))
                elif (port['type'] == 'DATA') and \
                        (port["direction"] == "out"):
                    utils.connect_out_to_top(
                        s, getattr(curr_inst, port["name"]),
                        port["name"] + "_out_" + str(i))

                    if (add_SR):
                        assert(input_width > 0)
                        urwv = projections[0]['inner_projection']['RX']
                        outport = utils.AddOutPort(s, port["width"] * urwv,
                                                   port["name"] + "_" + str(i))
                        for vali in range(math.floor(port['width'] /
                                                     input_width)):
                            curr_shift_reg = \
                                module_helper_classes.ShiftRegister(
                                    reg_width=input_width,
                                    length=urwv - 1, sim=False)
                            assert urwv - 1 > 0
                            setattr(s, "SR" + str(i) + "_" + str(vali),
                                    curr_shift_reg)
                            dataout = getattr(curr_inst, port["name"])
                            curr_shift_reg.input_data //= dataout[vali *
                                                                  input_width:
                                                                  input_width *
                                                                  (vali + 1)]
                            curr_shift_reg.ena //= 1
                            datain = getattr(curr_inst, port["name"])
                            curridx = vali * \
                                projections[0]['inner_projection']['RX']
                            outport[input_width * curridx:
                                    input_width * (curridx + 1)] //= \
                                datain[vali * input_width:
                                       input_width * (vali + 1)]
                            wv = projections[0]['inner_projection']['RX']
                            for outi in range(1, wv):
                                curridx = vali * wv + outi
                                outport[input_width * curridx:
                                        input_width * (curridx + 1)] //= \
                                    getattr(curr_shift_reg, "out" +
                                            str(outi - 1))
                    else:
                        utils.connect_out_to_top(s, getattr(curr_inst,
                                                            port["name"]),
                                                 port["name"] + "_" + str(i))
                elif (port['type'] == 'DATA') and \
                        (port["direction"] == "in"):
                    instport = utils.AddInPort(s, port['width'],
                                               port["name"] + "_" + str(i))
                    instport_new = utils.AddInPort(s, port['width'],
                                                   port["name"] + "_" +
                                                   str(i) + "_out")
                    datain = getattr(curr_inst, port['name'])
                    newmux = module_helper_classes.MUX2(port['width'], 1)
                    setattr(s, "input_data_mux" + str(i), newmux)
                    newmux.in0 //= instport_new
                    newmux.in1 //= instport
                    datain //= newmux.out
                elif port['type'] not in ('CLK', 'RESET', 'WEN'):
                    if (port['direction'] == "in"):
                        utils.connect_in_to_top(s, getattr(curr_inst,
                                                           port["name"]),
                                                port["name"] + "_" + str(i))
                    else:
                        utils.connect_out_to_top(s, getattr(curr_inst,
                                                            port["name"]),
                                                 port["name"] + "_" + str(i))

            for port in spec['ports']:
                # Mux between the input buffer write enable signal
                # and the output buffer write enable signal depending on
                # whether this is an input or output buffer
                if (port['type'] == 'WEN'):
                    assert(port["direction"] == "in")
                    # Add ports for both the input and output buffers.
                    instport = utils.AddInPort(s, port['width'],
                                               port["name"] + "_" + str(i))
                    instport_new = utils.AddInPort(s, port['width'],
                                                   port["name"] + "_" +
                                                   str(i) + "_out")
                    wen_in = getattr(curr_inst, port['name'])

                    # Mux between the different layers, connecting either
                    # the input activation WEN or output activation WEN.
                    newmux = module_helper_classes.MUXN(1, len(bcounts_i))
                    setattr(s, "input_wen_mux" + str(i), newmux)
                    for tt in range(len(bcounts_i)):
                        currin = getattr(newmux, "in" + str(tt))
                        i_order = (i - buffer_start_idxs[tt]) % \
                            max_total_bcount
                        if (i_order < bcounts_i[tt]):
                            currin //= instport
                        else:
                            currin //= instport_new
                    newmux.sel //= layer_sel
                    wen_in //= newmux.out

                    currmux = getattr(s, "input_data_mux" + str(i))
                    currmux.sel //= instport
        utils.tie_off_clk_reset(s)


class MergeBusses(Component):
    """" This module connects narrow input busses to wider output busses.
         ``ins_per_out`` busses of width ``in_width`` are merged into output
        busses of width ``out_width``. Unconnected outputs are tied to 0.

         :param input_<i>: Input port
         :type input_<i>: Component class
         :param output_<i>: Output port
         :type output_<i>: Component class
    """
    def construct(s, in_width=1, num_ins=1, out_width=1, num_outs=1,
                  start_bus=0, ins_per_out=0, sim=False):
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
        num_ins_used = min(ins_per_out*num_outs, num_ins)

        # Add outputs to activation functions
        utils.add_n_inputs(s, num_ins, in_width, "input_")
        utils.add_n_outputs(s, num_outs, out_width, "output_")

        # Add input and output ports from each MLB
        for inp in range(num_ins_used):
            bus_idx = (start_bus + math.floor(inp/ins_per_out)) % num_outs
            bus_start = (inp % ins_per_out) * in_width
            bus_end = ((inp % ins_per_out)+1) * in_width
            input_bus = getattr(s, "input_"+str(inp))
            output_bus = getattr(s, "output_"+str(bus_idx))
            connect(input_bus[0:in_width], output_bus[bus_start:bus_end])

        for i in range(num_outs):
            output_bus = getattr(s, "output_" + str(i))
            i_order = (i - start_bus) % num_outs
            if i_order >= math.ceil(num_ins_used / ins_per_out):
                output_bus //= 0
            elif ((ins_per_out*in_width < out_width)):
                output_bus[ins_per_out * in_width:out_width] //= 0
        utils.tie_off_clk_reset(s)


class WeightInterconnect(Component):
    """" This module connects the weight ports between the inner instances and
         the buffers.

         Two options:
         1) Connect weight buffers to MLBs
            Assume that entire input bus of each MLB should connect to the
            same buffer to simplify things. One buffer can connect to many
            MLBs though if the output is wide enough.
         2) "PRELOAD": Given N inputs, split all MLBs into N groups
            and chain together their weights

         TODO: Allow for directly connecting weights between instances

         :param inputs_from_buffer_<i>: Input port from weight buffer for i
                                        from 0 to ``num_buffers``
         :type inputs_from_buffer_<i>: Component class
         :param inputs_from_mlb_<i>: Input port from MLB (currently
                                     disconnected)
         :type inputs_from_mlb_<i>: Component class
         :param outputs_to_mlb_<i>: Output port to MLB
         :type outputs_to_mlb_<i>: Component class
         :param outputs_to_next_<i>: Outputs of chains of weights
         :type outputs_to_next_<i>: Outputs of chains of weights
    """
    def construct(s, buffer_width=1, mlb_width=-1, mlb_width_used=1,
                  num_buffers=1, num_mlbs=1, projection={}, sim=False,
                  num_mlbs_used=-1, inner_projection={}, dilx=1,
                  num_banks=1):
        """ Constructor for WeightInterconnect

         :param buffer_width: Bit-width of buffer datain/dataout ports
         :type buffer_width: int
         :param mlb_width: Total bit-width of the weight ports of the inner
                           instances.
         :type mlb_width: int
         :param mlb_width_used: Bit-width of the weight stream connecting to
                                each inner instance for this projection.
         :type mlb_width_used: int
         :param num_buffers: Total number of weight buffers
         :type num_buffers: int
         :param num_mlbs: Total number of inner instances
         :type num_mlbs: int
         :param projection: Projection specification
         :type projection: dict
        """
        # Validate inputs
        if mlb_width < 0:
            mlb_width = mlb_width_used
        streams_per_buffer = math.floor(buffer_width / mlb_width_used)

        buffers_per_stream = math.ceil(mlb_width_used / buffer_width)
        assert mlb_width_used <= mlb_width
        assert num_mlbs >= utils.get_var_product(
            projection, [['G'], ['E'], ['C'], ['B'], ['PX'], ['PY'],
                         ['RY'], ['RX']]), \
            "Insufficient number of MLBs"
        if (num_mlbs_used < 0):
            num_mlbs_used = num_mlbs

        preload = False
        preload_bus_count = 0
        if "PRELOAD" in projection:
            for pload_type in projection["PRELOAD"]:
                if pload_type["dtype"] == 'W':
                    preload = True
                    preload_bus_count = pload_type["bus_count"]

        assert preload or num_buffers >= math.ceil(
            utils.get_var_product(projection,
                                  [['G'], ['E'], ['C'], ['RY'], ['RX']]) *
            mlb_width_used / buffer_width),\
            "Insufficient number of weight buffers"

        # Add inputs from buffers
        utils.add_n_inputs(s, num_buffers*num_banks, buffer_width,
                           "inputs_from_buffer_")
        if (num_banks > 1):
            bank_sel = utils.AddInPort(s, math.ceil(math.log(num_banks, 2)),
                                       "bank_sel")

        if preload:
            assert mlb_width_used * preload_bus_count <= \
                num_buffers * buffer_width
            # It doesn't matter in which order they are connected if things
            # are preloaded - just connect them in chains.
            chain_len = math.ceil(num_mlbs_used / num_buffers)
            for chain in range(num_buffers):
                start_idx = chain * chain_len
                end_idx = min(num_mlbs_used - 1, start_idx + chain_len - 1)
                newout, newin = utils.chain_ports(s, start_idx, end_idx,
                                                  "inputs_from_mlb_{}",
                                                  "outputs_to_mlb_{}",
                                                  mlb_width)

                # Then connect each chain input
                assert streams_per_buffer > 0, "If preloading weights," + \
                    "the weight buffer data width must be at least as " + \
                    "wide as the ML block weight input"
                input_bus_idx = math.floor(chain / streams_per_buffer)
                if (num_banks > 1):
                    newmux = module_helper_classes.MUXN(buffer_width,
                                                        num_banks)
                    setattr(s, "bank_mux" + str(input_bus_idx), newmux)
                    for mm in range(num_banks):
                        currin = getattr(newmux, "in" + str(mm))
                        inbus_mm = getattr(s, "inputs_from_buffer_" +
                                           str(input_bus_idx + mm*num_buffers))
                        currin //= inbus_mm
                    input_bus = newmux.out
                    newmux.sel //= bank_sel
                else:
                    input_bus = getattr(s, "inputs_from_buffer_" +
                                        str(input_bus_idx))
                section_idx = chain % streams_per_buffer
                input_bus_start = section_idx * mlb_width_used
                input_bus_end = (section_idx + 1) * mlb_width_used
                connect(newout[0:mlb_width_used],
                        input_bus[input_bus_start:input_bus_end])

                # Then connect each chain output
                output_bus = utils.AddOutPort(s, buffer_width,
                                              "outputs_to_buffer_" +
                                              str(input_bus_idx))
                connect(newin[0:mlb_width_used],
                        output_bus[input_bus_start:input_bus_end])

        else:
            for (ug, ue, ubb, urny, urnc, urw, ubx, uby) in utils.range8D(
                    projection['G'], projection['E'], projection['B'],
                    projection['RY'], projection['C'], projection['RX'],
                    projection['PX'], projection['PY']):
                # Get instance number of the MLB
                out_idx = utils.get_overall_idx(
                    projection, {'RX': urw, 'C': urnc, 'RY': urny, 'B': ubb,
                                 'E': ue, 'G': ug, 'PX': ubx, 'PY': uby})

                # Create ports to and from the MLB
                newout = utils.AddOutPort(s, mlb_width,
                                          "outputs_to_mlb_" + str(out_idx))
                newin = utils.AddInPort(s, mlb_width,
                                        "inputs_from_mlb_" + str(out_idx))

                # Connect all MLB weight inputs to buffers
                stream_idx = utils.get_overall_idx(
                    projection, {'RX': urw, 'C': urnc, 'RY': urny, 'E': ue,
                                 'G': ug})
                for buf_idx in range(buffers_per_stream):
                    if (buffers_per_stream > 1):
                        input_bus_idx = stream_idx * buffers_per_stream + \
                            buf_idx
                        section_idx = 0
                        output_bus_start = buf_idx * buffer_width
                    else:
                        input_bus_idx = math.floor(stream_idx /
                                                   streams_per_buffer)
                        section_idx = stream_idx % streams_per_buffer
                        output_bus_start = 0
                    if (num_banks > 1):
                        newmux = module_helper_classes.MUXN(buffer_width,
                                                            num_banks)
                        setattr(s, "bank_mux" + str(input_bus_idx), newmux)
                        for mm in range(num_banks):
                            currin = getattr(newmux, "in" + str(mm))
                            inbus_mm = getattr(s, "inputs_from_buffer_" +
                                               str(input_bus_idx*num_banks))
                            currin //= inbus_mm
                        input_bus = newmux.out
                        newmux.sel //= bank_sel
                    else:
                        input_bus = getattr(s, "inputs_from_buffer_" +
                                            str(input_bus_idx))
                    input_bus_start = section_idx * mlb_width_used
                    input_bus_end = (section_idx + 1) * \
                        min(mlb_width_used, buffer_width)
                    output_bus_end = output_bus_start + min(mlb_width_used,
                                                            buffer_width)
                    if (dilx > 1):
                        assert(inner_projection)
                        num_weight_ins = utils.get_var_product(
                            inner_projection, [['G'], ['E'], ['RY'], ['C']])
                        for input_gen in range(num_weight_ins):
                            urwx = inner_projection.get('RX')
                            for weight_x in range(urwx):
                                input_w = input_gen * urwx + weight_x
                                w_width = int(mlb_width_used /
                                              (urwx * num_weight_ins))
                                start_w_idx = output_bus_start + \
                                    w_width * input_w
                                end_w_idx = output_bus_start + \
                                    w_width * (input_w + 1)
                                total_urw = (inner_projection.get('RX') *
                                             urw + weight_x)
                                if (total_urw % dilx == 0):
                                    connect(newout[start_w_idx:end_w_idx],
                                            input_bus[input_bus_start +
                                                      start_w_idx:
                                                      input_bus_start +
                                                      end_w_idx])
                                else:
                                    newout[start_w_idx:end_w_idx] //= 0
                    else:
                        connect(newout[output_bus_start:output_bus_end],
                                input_bus[input_bus_start:input_bus_end])

                    # Then connect each chain output
                    if (ubb + ubx + uby == 0):
                        output_bus = utils.AddOutPort(s, buffer_width,
                                                      "outputs_to_buffer_" +
                                                      str(input_bus_idx))
                        connect(newin[output_bus_start:output_bus_end],
                                output_bus[input_bus_start:input_bus_end])

        # Tie disconnected MLBs to 0
        for i in range(num_mlbs):
            if (("outputs_to_mlb_" + str(i)) not in s.__dict__.keys()):
                newout = OutPort(mlb_width)
                setattr(s, "outputs_to_mlb_" + str(i), newout)
                newout //= 0
            utils.AddInPort(s, mlb_width, "inputs_from_mlb_" + str(i))
        utils.tie_off_clk_reset(s)


class InputInterconnect(Component):
    """" This module connects the input ports between the inner instances and
         the buffers.

         1) Connect input activation buffers to MLBs
            Assume that entire input bus of each MLB chain should connect to
            the same buffer to simplify things. One buffer can connect to
            many MLBs though if the output is wide enough.

         1) Connect MLBs to each other
            Chains of URW MLBs have the same input. Connect the inputs between
            these sets of blocks.

         :param inputs_from_buffer_<i>: Input port from weight buffer for i
                                        from 0 to ``num_buffers``
         :type inputs_from_buffer_<i>: Component class
         :param inputs_from_mlb_<i>: Input port from MLB
                                     (currently disconnected)
         :type inputs_from_mlb_<i>: Component class
         :param outputs_to_mlb_<i>: Output port to MLB
         :type outputs_to_mlb_<i>: Component class
    """
    def construct(s, buffer_width=1, mlb_width=-1, mlb_width_used=1,
                  num_buffers=1, num_mlbs=1, projection={},
                  inner_projection={}, inner_width=1, mux_urn=False,
                  sim=False, dily=1, buffer_start_idx=0):
        """ Constructor for InputInterconnect

         :param buffer_width: Bit-width of buffer datain/dataout ports
         :type buffer_width: int
         :param mlb_width: Total bit-width of the weight ports of the inner
                           instances.
         :type mlb_width: int
         :param mlb_width_used: Bit-width of the weight stream connecting
                                to each inner instance for this projection.
         :type mlb_width_used: int
         :param num_buffers: Total number of weight buffers
         :type num_buffers: int
         :param num_mlbs: Total number of inner instances
         :type num_mlbs: int
         :param projection: Projection specification
         :type projection: dict
        """
        if ("outer_projection" in projection):
            full_projection = projection
            inner_projection = projection['inner_projection']
            projection = projection['outer_projection']
        else:
            full_projection = {"outer_projection": projection,
                               "inner_projection": inner_projection}
        if mlb_width < 0:
            mlb_width = mlb_width_used
        streams_per_buffer = buffer_width/mlb_width_used
        assert mlb_width_used <= mlb_width
        assert num_mlbs >= utils.get_var_product(projection,
                                                 [['G'], ['E'], ['B'], ['PX'],
                                                  ['PY'], ['C'], ['RY'],
                                                  ['RX']]), \
            "Insufficient number of MLBs"

        buffers_per_stream = math.ceil(1 / streams_per_buffer)
        full_buffer_width = buffer_width
        ins_per_buffer = (mlb_width_used / inner_width) * streams_per_buffer
        if (num_buffers > 1):
            buffer_width = utils.get_max_input_bus_width(buffer_width,
                                                         full_projection, 'I',
                                                         inner_width)
            ins_per_buffer = buffer_width / inner_width
            buffers_per_stream = math.ceil((mlb_width_used / inner_width) /
                                           ins_per_buffer)
            streams_per_buffer = buffer_width / mlb_width_used

        # Add inputs from buffers
        utils.add_n_inputs(s, num_buffers, full_buffer_width,
                           "inputs_from_buffer_")
        total_urn = inner_projection.get('RY', 1) * projection['RY']
        mux_size = projection['PY'] * \
            inner_projection.get('PY', 1) * total_urn
        max_ubbi = int(inner_projection.get('B', 1) *
                       inner_projection.get('PX', 1) *
                       inner_projection.get('PY', 1))
        max_unci = int(inner_projection.get('C', 1))
        if (total_urn == 1):
            mux_size = 1
            max_unci = 1
            max_ubbi = 1
        if (mux_urn and mux_size > 1):
            s.urn_sel = InPort(math.ceil(math.log(max(mux_size, 2), 2)))
            utils.tie_off_port(s, s.urn_sel)

        # Add input and output ports from each MLB
        mux_count = 0
        mlb_chains = []

        for (ug, ue, ubb, urnc, ubx, b, c, d) in utils.range8D(
                projection['G'], projection['E'], projection['B'],
                projection['C'], projection['PX']):
            muxs = []
            if (mux_size > 1) and mux_urn:
                for mi in range(inner_projection['G'] * max_ubbi * max_unci):
                    newmux = module_helper_classes.MUX_NXN(inner_width,
                                                           mux_size)
                    muxs += [newmux]
                    setattr(s, "mux" + str(mux_count), newmux)
                    newmux.sel //= s.urn_sel
                    mux_count += 1

            for (uby, urny) in utils.range2D(projection['PY'],
                                             projection['RY']):
                chain_idx = utils.get_overall_idx(
                    projection, {'RY': urny, 'C': urnc, 'B': ubb,
                                 'PX': ubx, 'PY': uby, 'G': ug, 'E': ue})
                start_idx = chain_idx * projection['RX']
                end_idx = start_idx + projection['RX'] - 1
                newout, newin = utils.chain_ports(s, start_idx, end_idx,
                                                  "inputs_from_mlb_{}",
                                                  "outputs_to_mlb_{}",
                                                  mlb_width)
                mlb_chains += [list(range(start_idx, end_idx + 1))]

                # Connect the chain's input
                stream_idx = utils.get_overall_idx_new(
                    projection, {'RY': urny, 'C': urnc, 'PY': uby,
                                 'PX': ubx, 'B': ubb, 'G': ug},
                    order=utils.input_order)
                streams_per_buf_int = math.floor(streams_per_buffer)
                for buf in range(buffers_per_stream):
                    input_bus_idx = stream_idx * buffers_per_stream + buf
                    input_bus_start = 0
                    if (streams_per_buffer > 1):
                        input_bus_idx = math.floor(
                            stream_idx / streams_per_buf_int)
                        section_idx = stream_idx % streams_per_buf_int
                        input_bus_start = section_idx * mlb_width_used
                    i_order = (input_bus_idx + buffer_start_idx) % num_buffers
                    input_bus = getattr(s, "inputs_from_buffer_" +
                                        str(i_order))
                    if mux_urn and (mux_size > 1):
                        # For each separate input image connected to this MLB:
                        for (ugi, ubbi, unci, d) in utils.range4D(
                                inner_projection['G'], max_ubbi, max_unci):
                            # For each different input from that image,
                            # Connect it to the mux.
                            currmux = muxs[ugi * max_ubbi * max_unci +
                                           ubbi * max_unci + unci]
                            for (ubyi, unyi) in utils.range2D(
                                    inner_projection['PY'],
                                    inner_projection['RY']):
                                mlb_in_idx = utils.get_overall_idx_new(
                                    inner_projection,
                                    {'RY': unyi, 'C': unci, 'PX': 0,
                                     'PY': ubyi, 'B': ubbi, 'G': ugi},
                                    order=utils.input_order)
                                if (math.floor(mlb_in_idx / ins_per_buffer) ==
                                        buf):
                                    curr_uby = uby * inner_projection['PY'] + \
                                        ubyi
                                    total_uny = inner_projection['RY'] * \
                                        projection['RY']
                                    curr_uny = urny * \
                                        inner_projection['RY'] + unyi
                                    mux_in_idx = curr_uby * total_uny + \
                                        curr_uny

                                    muxin = getattr(currmux, "in" +
                                                    str(mux_in_idx))
                                    muxout = getattr(currmux, "out" +
                                                     str(mux_in_idx))
                                    total_idx = input_bus_start + \
                                        math.floor(mlb_in_idx %
                                                   ins_per_buffer) * \
                                        inner_width
                                    connect(input_bus[total_idx:total_idx +
                                                      inner_width], muxin)
                                    if (curr_uny % dily == 0):
                                        connect(muxout,
                                                newout[mlb_in_idx *
                                                       inner_width:
                                                       (mlb_in_idx + 1) *
                                                       inner_width])
                                    else:
                                        newout[mlb_in_idx * inner_width:
                                               (mlb_in_idx + 1) *
                                               inner_width] //= 0
                    else:
                        section_w = min(buffer_width, mlb_width_used)
                        buf_start_idx = buf * section_w
                        connection_width = min(mlb_width,
                                               buf_start_idx + section_w) - \
                            buf_start_idx
                        connect(newout[buf_start_idx:buf_start_idx +
                                       connection_width],
                                input_bus[input_bus_start:
                                          input_bus_start + connection_width])
                    # And one of the outputs
                    if (ue == 0):
                        output_bus = utils.AddOutPort(s, buffer_width,
                                                      "outputs_to_buffer_" +
                                                      str(input_bus_idx))
                        section_w = min(buffer_width, mlb_width_used)
                        buf_start_idx = buf*section_w
                        connection_width = min(mlb_width, buf_start_idx +
                                               (section_w)) - buf_start_idx
                        connect(newin[buf_start_idx:
                                      buf_start_idx + connection_width],
                                output_bus[input_bus_start:
                                           input_bus_start + connection_width])

        # Tie disconnected MLBs to 0
        for i in range(num_mlbs):
            if ("outputs_to_mlb_" + str(i) not in s.__dict__.keys()):
                newout = OutPort(mlb_width)
                setattr(s, "outputs_to_mlb_" + str(i), newout)
                newout //= 0
            utils.AddInPort(s, mlb_width, "inputs_from_mlb_" + str(i))
        utils.tie_off_clk_reset(s)


class OutputPSInterconnect(Component):
    """" This module connects the output ports between the inner instances and
         the activation functions

         1) Connect MLBs to activation functions
            Split the outputs of chains of MLBs into activations, and connect
            them to the corresponding activation functions.

         1) Connect MLBs to each other
            Chains of URW*URN MLBs have the same output. Connect the outputs
            between these sets of blocks.

         TODO: Allow for weight stationary flow
         TODO: Allow of loading partial sums from a buffer.
         TODO: Send partial sums to a buffer.

         :param outputs_to_afs_<i>: Output ports connecting to activation
                                    functions
         :type outputs_to_afs_<i>: Component class
         :param inputs_from_mlb_<i>: Input port from MLB
                                     (currently disconnected)
         :type inputs_from_mlb_<i>: Component class
         :param outputs_to_mlb_<i>: Output port to MLB
         :type outputs_to_mlb_<i>: Component class
    """
    def construct(s, af_width=1, mlb_width=-1, mlb_width_used=1, num_afs=1,
                  num_mlbs=1, projection={}, sim=False, input_buf_width=0,
                  num_input_bufs=0):
        """ Constructor for OutputInterconnect

         :param af_width: Bit-width of activation function input
         :type af_width: int
         :param mlb_width: Total bit-width of the weight ports of the inner
                           instances.
         :type mlb_width: int
         :param mlb_width_used: Bit-width of the weight stream connecting to
                                each inner instance for this projection.
         :type mlb_width_used: int
         :param num_afs: Total number of activation functions available
         :type num_afs: int
         :param num_mlbs: Total number of inner instances
         :type num_mlbs: int
         :param projection: Projection specification
         :type projection: dict
        """
        if mlb_width < 0:
            mlb_width = mlb_width_used
        acts_per_stream = math.floor(mlb_width_used / af_width)
        assert mlb_width_used <= mlb_width
        assert mlb_width_used % af_width == 0, \
            "The activation input width should be a factor of the total " + \
            "output stream width"
        assert acts_per_stream > 0, "Activation function width too wide"
        assert num_mlbs >= utils.get_var_product(
            projection, [['G'], ['E'], ['B'], ['PX'], ['PY'],
                         ['RY'], ['RX'], ['C']]), "Insufficient # of MLBs"
        assert num_afs >= math.ceil(utils.get_var_product(
            projection, [['G'], ['B'], ['PX'], ['PY'], ['E']])
            * acts_per_stream), \
            "Insufficient number of activation functions"

        # Add outputs to activation functions
        outs_to_afs = utils.add_n_outputs(s, num_afs, af_width,
                                          "outputs_to_afs_")
        if (num_input_bufs > 0):
            utils.add_n_inputs(s, num_input_bufs, input_buf_width,
                               "ps_inputs_from_buffer_")

        # Add input and output ports from each MLB
        connected_outs = []
        output_chains = []
        for (ug, ue, ubb, ubx, uby, a, b, c) in utils.range8D(
                projection['G'], projection['E'], projection['B'],
                projection['PX'], projection['PY']):
            chain_idx = utils.get_overall_idx(
                projection, {'B': ubb, 'G': ug, 'E': ue, 'PX': ubx, 'PY': uby})
            chain_len = projection['RX'] * projection['C'] * \
                projection['RY']
            start_idx = chain_idx * chain_len
            end_idx = start_idx + chain_len - 1
            newout, newin = utils.chain_ports(s, start_idx, end_idx,
                                              "inputs_from_mlb_{}",
                                              "outputs_to_mlb_{}", mlb_width)
            output_chains += [list(range(start_idx, end_idx+1))]

            if (num_input_bufs == 0):
                newout[0:mlb_width_used] //= 0
            output_bus_idx = chain_idx * acts_per_stream

            # Connect input stream.
            if (num_input_bufs > 0):
                assert input_buf_width >= mlb_width_used
                streams_per_buffer = math.floor(input_buf_width /
                                                mlb_width_used)
                input_bus_idx = math.floor(chain_idx / streams_per_buffer)
                section_idx = chain_idx % streams_per_buffer
                input_bus_start = section_idx * mlb_width_used
                input_bus_end = (section_idx + 1) * mlb_width_used
                input_bus = getattr(s, "ps_inputs_from_buffer_" +
                                    str(input_bus_idx))
                connect(input_bus[input_bus_start:input_bus_end],
                        newout[0:mlb_width_used])

            for out_part in range(acts_per_stream):
                output_bus = getattr(s, "outputs_to_afs_" +
                                     str(output_bus_idx + out_part))
                connected_outs += [output_bus]
                output_bus_start = out_part * af_width
                output_bus_end = (out_part + 1) * af_width
                connect(output_bus, newin[output_bus_start:output_bus_end])

        # Tie disconnected MLBs to 0
        for i in range(num_mlbs):
            if ("outputs_to_mlb_" + str(i) not in s.__dict__.keys()):
                newout = OutPort(mlb_width)
                setattr(s, "outputs_to_mlb_" + str(i), newout)
                newout //= 0
            newin = utils.AddInPort(s, mlb_width, "inputs_from_mlb_" +
                                    str(i))
        # Tie disconnected outs to 0
        for out in outs_to_afs:
            if (out not in connected_outs):
                out //= 0
        utils.tie_off_clk_reset(s)

        with open('chain_list_for_placement.yaml', 'w') as file:
            yaml.dump(output_chains, file)


class Datapath(Component):
    """" This module includes the whole datapath:

         :param mlb_modules: Contains all MLB modules
         :type mlb_modules: HWB_Wrapper Component
         :param weight_modules: Contains all weight buffers
         :type weight_modules: HWB_Wrapper Component
         :param input_act_modules: Contains all input activation buffers
         :type input_act_modules: HWB_Wrapper Component
         :param output_act_modules: Contains all output activation buffers
         :type output_act_modules: HWB_Wrapper Component
         :param activation_function_modules: Contains all activation functions
         :type activation_function_modules: ActivationWrapper Component
         :param weight_interconnect: Interconnect for weights
         :type weight_interconnect: WeightInterconnect Component
         :param input_interconnect: Interconnect for inputs
         :type input_interconnect: InputInterconnect Component
         :param output_ps_interconnect: Interconnect for partial sums
         :type output_ps_interconnect: OutputPSInterconnect Component
         :param output_interconnect: Interconnect for activation outputs
         :type output_interconnect: MargeBusses Component
    """
    def construct(s, mlb_spec={}, wb_spec={}, ib_spec={}, ob_spec={},
                  proj_specs=[], fast_gen=False, pingpong_w=False):
        """ Constructor for Datapath

         :param af_width: Bit-width of activation function input
         :type af_width: int
         :param mlb_width: Total bit-width of the weight ports of the inner
                           instances.
         :type mlb_width: int
         :param mlb_width_used: Bit-width of the weight stream connecting
                                to each inner instance for this projection.
         :type mlb_width_used: int
         :param num_afs: Total number of activation functions available
         :type num_afs: int
         :param num_mlbs: Total number of inner instances
         :type num_mlbs: int
         :param projection: Projection specification
         :type projection: dict
        """
        utils.printi(il, "{:=^60}".format(
            "> Constructing Datapath with MLB block " +
            str(mlb_spec.get('block_name', "unnamed") + " <")))
        MAC_datatypes = ['W', 'I', 'O']
        buffer_specs = {'W': wb_spec, 'I': ib_spec, 'O': ob_spec}

        # Calculate required MLB interface widths and print information
        inner_projs = [proj_spec['inner_projection']
                       for proj_spec in proj_specs]
        MAC_counts = [utils.get_mlb_count(inner_proj)
                      for inner_proj in inner_projs]
        inner_bus_counts = {dtype: [utils.get_proj_stream_count(inner_proj,
                                                                dtype)
                                    for inner_proj in inner_projs]
                            for dtype in MAC_datatypes}
        inner_data_widths = {dtype: [proj_spec['data_widths'][dtype]
                                     for proj_spec in proj_specs]
                             for dtype in MAC_datatypes}
        inner_bus_widths = {dtype: [inner_bus_count * inner_data_width
                                    for (inner_bus_count, inner_data_width)
                                    in zip(inner_bus_counts[dtype],
                                           inner_data_widths[dtype])]
                            for dtype in MAC_datatypes}

        for (proj_spec, MAC_count, inner_bus_count, inner_data_width,
             inner_bus_width) in zip(proj_specs, MAC_counts, inner_bus_counts,
                                     inner_data_widths, inner_bus_widths):
            print(utils.print_table("ML Block Details, Projection " +
                                    proj_spec.get("name", "unnamed"),
                                    [["Num MACs", MAC_count,
                                      "(MACs within each MLB)"],
                                     ["values/MLB, by type", inner_bus_counts,
                                      "(number of in and out values / MLB)"],
                                     ["data widths by type", inner_data_widths,
                                      "(bit-width of each value)"],
                                     ["total bus width, by type",
                                      inner_bus_widths,
                                      "(bit-width of MLB interface)"]], il) +
                  "\n")

        # Check that this configuration is supported by the hardware model
        assert MAC_count <= mlb_spec['MAC_info']['num_units']
        for dtype in MAC_datatypes:
            for i in range(len(inner_bus_widths[dtype])):
                assert inner_bus_widths[dtype][i] <= \
                    utils.get_sum_datatype_width(mlb_spec, dtype)
                assert (inner_data_widths[dtype][i] <=
                        mlb_spec['MAC_info']['data_widths'][dtype]), \
                    "MLB width insufficient for inner projection"

        # Calculate required number of MLBs, IO streams, activations
        outer_projs = [proj_spec['outer_projection']
                       for proj_spec in proj_specs]
        MLB_counts = [utils.get_mlb_count(outer_proj)
                      for outer_proj in outer_projs]
        outer_bus_counts = {dtype: [utils.get_proj_stream_count(outer_proj,
                                                                dtype)
                                    for outer_proj in outer_projs]
                            for dtype in MAC_datatypes}
        outer_bus_widths = {dtype: [outer_bus_count * inner_bus_width
                                    for (outer_bus_count, inner_bus_width) in
                                    zip(outer_bus_counts[dtype],
                                        inner_bus_widths[dtype])]
                            for dtype in MAC_datatypes}
        total_bus_counts = {dtype: [outer_bus_count * inner_bus_count
                            for (outer_bus_count, inner_bus_count) in
                                    zip(outer_bus_counts[dtype],
                                        inner_bus_counts[dtype])]
                            for dtype in MAC_datatypes}
        buffer_counts = {}
        num_w_banks = 2 if (pingpong_w) else 1
        if (num_w_banks > 1):
            utils.AddInPort(s, math.ceil(math.log(num_w_banks, 2)), "bank_sel")

        buffer_counts['W'] = [utils.get_num_buffers_reqd(buffer_specs['W'],
                                                         outer_bus_count,
                                                         inner_bus_width)
                              for (outer_bus_count, inner_bus_width) in
                              zip(outer_bus_counts['W'],
                                  inner_bus_widths['W'])]
        max_input_buf_widths = [utils.get_max_input_bus_width(
            utils.get_sum_datatype_width(buffer_specs['I'], "DATA", ["in"]),
            proj, 'I') for proj in proj_specs]
        buffer_counts['I'] = [utils.get_num_buffers_reqd(buffer_specs['I'],
                                                         outer_bus_count,
                                                         inner_bus_width, mw)
                              for (outer_bus_count, inner_bus_width, mw) in
                              zip(outer_bus_counts['I'],
                                  inner_bus_widths['I'],
                                  max_input_buf_widths)]
        buffer_counts['O'] = [utils.get_num_buffers_reqd(buffer_specs['O'],
                                                         outer_bus_counto *
                                                         inner_bus_counto,
                                                         inner_data_widthi)
                              for (outer_bus_counto, inner_bus_counto,
                                   inner_data_widthi) in
                              zip(outer_bus_counts["O"],
                                  inner_bus_counts["O"],
                                  inner_data_widths["I"])]

        for (proj_spec, MAC_count, outer_bus_width, total_bus_count) in \
                zip(proj_specs, MAC_counts, outer_bus_widths,
                    total_bus_counts):
            print(utils.print_table("Dataflow Details, Projection " +
                                    proj_spec.get("name", "unnamed"),
                                    [["Num MLBs", MAC_count,
                                      "(Number of MLBs reqd for projection)"],
                                     ["total data widths by type",
                                      outer_bus_widths,
                                      "(total data width from buffers)"],
                                     ["total values, by type",
                                      total_bus_counts,
                                      "(total # values from buffers)"]],
                                    il) + "\n")

        # Instantiate MLBs, buffers
        # Allocate buffers for each layer
        buffer_counts['I'] = [utils.get_num_buffers_reqd(buffer_specs['I'],
                                                         outer_bus_count,
                                                         inner_bus_width, mw)
                              for (outer_bus_count, inner_bus_width, mw) in
                              zip(outer_bus_counts['I'],
                                  inner_bus_widths['I'],
                                  max_input_buf_widths)]

        total_num_buffers = max(sum(x) for x in zip(buffer_counts['I'],
                                                    buffer_counts['O']))
        buffer_start_idxs = [sum(buffer_counts['I'][0:i+1]) %
                             total_num_buffers
                             for i in range(len(buffer_counts['I']) - 1)]
        ibuffer_start_idxs = [0] + buffer_start_idxs
        obuffer_start_idxs = [(a + b) % total_num_buffers
                              for (a, b) in zip(ibuffer_start_idxs,
                                                buffer_counts['I'])]

        s.sel = InPort(math.ceil(math.log(max(len(proj_specs), 2), 2)))
        utils.tie_off_port(s, s.sel)
        s.weight_modules = HWB_Wrapper(buffer_specs['W'],
                                       max(buffer_counts['W'])*num_w_banks,
                                       fast_gen=fast_gen)
        s.input_act_modules = InputBufferWrapper(
            buffer_specs['I'], buffer_counts['I'], buffer_counts['O'],
            projections=proj_specs, fast_gen=fast_gen,
            add_SR=(("access_patterns" in mlb_spec) and
                    (mlb_spec["access_patterns"]["AP1"] <
                     inner_projs[i]["RX"]) and (inner_projs[i]["RX"] > 1)),
            input_width=max(inner_data_widths['I']),
            buffer_start_idxs=ibuffer_start_idxs)
        s.mlb_modules = HWB_Wrapper(mlb_spec, max(MLB_counts),
                                    projections=proj_specs, fast_gen=fast_gen)
        s.input_act_modules.sel //= s.sel
        activation_function_modules = []
        for i in range(len(proj_specs)):
            new_act_modules = ActivationWrapper(
                count=max(total_bus_counts['O']),
                function=utils.get_activation_function_name(proj_specs[i]),
                input_width=max(inner_data_widths['O']),
                output_width=max(inner_data_widths['I']),
                registered=False)
            if (i > 0):
                newname = proj_specs[i].get("name", i)
            else:
                newname = ""
            activation_function_modules += [new_act_modules]
            setattr(s, "activation_function_modules" + newname,
                    new_act_modules)

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
            weight_interconnect = WeightInterconnect(
                buffer_width=utils.get_sum_datatype_width(buffer_specs['W'],
                                                          'DATA', ["in"]),
                mlb_width=utils.get_sum_datatype_width(mlb_spec, 'W', ["in"]),
                mlb_width_used=inner_bus_widths['W'][i],
                num_buffers=max(buffer_counts['W']),
                num_mlbs=max(MLB_counts),
                projection=outer_projs[i],
                inner_projection=inner_projs[i],
                dilx=proj_specs[i].get("dilation", {}).get("x", 1),
                num_banks=num_w_banks
            )
            weight_interconnects += [weight_interconnect]
            setattr(s, "weight_interconnect" + newname, weight_interconnect)
            if (num_w_banks > 1):
                weight_interconnect.bank_sel //= s.bank_sel
            input_buf_width = utils.get_sum_datatype_width(buffer_specs['I'],
                                                           'DATA', ["in"])
            mlb_width_used = inner_bus_widths['I'][i]
            mlb_width = utils.get_sum_datatype_width(mlb_spec, 'I', ["in"])
            inner_width = inner_data_widths['I'][i]
            buf_width = utils.get_sum_datatype_width(buffer_specs['I'],
                                                     'DATA', ["in"])

            if (("access_patterns" in mlb_spec) and
                    (mlb_spec["access_patterns"]["AP1"] <
                     inner_projs[i]["RX"])):
                input_buf_width = input_buf_width * inner_projs[i]["RX"]
                mlb_width_used = mlb_width_used * inner_projs[i]["RX"]
                inner_width = inner_width * inner_projs[i]["RX"]
                buf_width = buf_width * inner_projs[i]["RX"]

            input_interconnect = InputInterconnect(
                buffer_width=buf_width,
                mlb_width=max(mlb_width, mlb_width_used),
                mlb_width_used=mlb_width_used,
                num_buffers=max(sum(x) for x in zip(buffer_counts['I'],
                                                    buffer_counts['O'])),
                num_mlbs=max(MLB_counts),
                projection=proj_specs[i],
                inner_projection=inner_projs[i],
                inner_width=inner_width,
                mux_urn=True,
                dily=proj_specs[i].get("dilation", {}).get("y", 1),
                buffer_start_idx=ibuffer_start_idxs[i]
            )
            setattr(s, "input_interconnect" + newname, input_interconnect)
            input_interconnects += [input_interconnect]
            output_ps_interconnect = OutputPSInterconnect(
                af_width=inner_data_widths['O'][i],
                mlb_width=utils.get_sum_datatype_width(mlb_spec, 'O', ["in"]),
                mlb_width_used=inner_bus_widths['O'][i],
                num_afs=max(total_bus_counts['O']),
                num_mlbs=max(MLB_counts),
                projection=outer_projs[i])
            output_ps_interconnects += [output_ps_interconnect]
            setattr(s, "output_ps_interconnect" + newname,
                    output_ps_interconnect)
            output_interconnect = MergeBusses(
                in_width=inner_data_widths['I'][i],
                num_ins=max(total_bus_counts['O']),
                out_width=utils.
                get_sum_datatype_width(buffer_specs['O'], 'DATA', ["in"]),
                num_outs=max(sum(x) for x in zip(buffer_counts['I'],
                                                 buffer_counts['O'])),
                start_bus=obuffer_start_idxs[i])
            output_interconnects += [output_interconnect]
            setattr(s, "output_interconnect" + newname, output_interconnect)

        # Connect MLB sel
        modeports = list(utils.get_ports_of_type(mlb_spec, 'MODE', ["in"]))
        connected_ins = []
        if (len(modeports) > 0):
            mlb_sel_port = getattr(s.mlb_modules, modeports[0]["name"])
            connected_ins += [mlb_sel_port]
            mlb_sel_port //= s.sel

        # Connect weight interconnect
        for portname in utils.get_ports_of_type(mlb_spec, 'W', ["out"]):
            for i in range(len(proj_specs)):
                weight_interconnect = weight_interconnects[i]
                connected_ins += utils.connect_ports_by_name(
                    s.mlb_modules, portname["name"] + r"_(\d+)",
                    weight_interconnect, r"inputs_from_mlb_(\d+)")

        for portname in utils.get_ports_of_type(mlb_spec, 'W', ["in"]):
            connected_ins += utils.mux_ports_by_name(
                s, weight_interconnects, r"outputs_to_mlb_(\d+)",
                s.mlb_modules, portname["name"] + r"_(\d+)", insel=s.sel)

        for portname in utils.get_ports_of_type(buffer_specs['W'], 'DATA',
                                                ["out"]):
            for i in range(len(proj_specs)):
                weight_interconnect = weight_interconnects[i]
                connected_ins += utils.connect_ports_by_name(
                    s.weight_modules, portname["name"] + r"_(\d+)",
                    weight_interconnect, r"inputs_from_buffer_(\d+)")

        # Connect input interconnect
        for portname in utils.get_ports_of_type(mlb_spec, 'I', ["out"]):
            for i in range(len(proj_specs)):
                input_interconnect = input_interconnects[i]
                connected_ins += utils.connect_ports_by_name(
                    s.mlb_modules,  portname["name"] + r"_(\d+)",
                    input_interconnect, r"inputs_from_mlb_(\d+)")

        for portname in utils.get_ports_of_type(mlb_spec, 'I', ["in"]):
            connected_ins += utils.mux_ports_by_name(s, input_interconnects,
                                                     r"outputs_to_mlb_(\d+)",
                                                     s.mlb_modules,
                                                     portname["name"] +
                                                     r"_(\d+)",
                                                     insel=s.sel)

        for portname in utils.get_ports_of_type(buffer_specs['I'], 'DATA',
                                                ["out"]):
            for i in range(len(proj_specs)):
                input_interconnect = input_interconnects[i]
                connected_ins += utils.connect_ports_by_name(
                    s.input_act_modules, portname["name"] + r"_(\d+)",
                    input_interconnect, r"inputs_from_buffer_(\d+)")

        # Connect partial sum interconnect
        for portname in utils.get_ports_of_type(mlb_spec, 'O', ["out"]):
            for i in range(len(proj_specs)):
                output_ps_interconnect = output_ps_interconnects[i]
                connected_ins += utils.connect_ports_by_name(
                    s.mlb_modules, portname["name"] + r"_(\d+)",
                    output_ps_interconnect, r"inputs_from_mlb_(\d+)")
        for portname in utils.get_ports_of_type(mlb_spec, 'O', ["in"]):
            connected_ins += utils.mux_ports_by_name(
                s, output_ps_interconnects, r"outputs_to_mlb_(\d+)",
                s.mlb_modules, portname["name"] + r"_(\d+)", insel=s.sel)

        for i in range(len(proj_specs)):
            output_ps_interconnect = output_ps_interconnects[i]
            output_interconnect = output_interconnects[i]
            activation_functions = activation_function_modules[i]
            connected_ins += utils.connect_ports_by_name(
                output_ps_interconnect, r"outputs_to_afs_(\d+)",
                activation_functions, r"activation_function_in_(\d+)")
            connected_ins += utils.connect_ports_by_name(
                activation_functions, r"activation_function_out_(\d+)",
                output_interconnect, r"input_(\d+)")

        for portname in utils.get_ports_of_type(buffer_specs['O'], 'DATA',
                                                ["in"]):
            connected_ins += utils.mux_ports_by_name(
                s, output_interconnects, r"output_(\d+)", s.input_act_modules,
                portname["name"] + r"_(\d+)_out", insel=s.sel)

        # Connect output buffers to top
        for port in s.input_act_modules.get_output_value_ports():
            for dout in utils.get_ports_of_type(buffer_specs['O'], 'DATA',
                                                ["out"]):
                if dout["name"] in port._dsl.my_name:
                    utils.connect_out_to_top(s, port, port._dsl.my_name)

        # Connect input and weight datain to a common port
        utils.AddInPort(s, utils.get_sum_datatype_width(buffer_specs['I'],
                                                        'DATA', ["in"]),
                        "input_datain")
        for port in utils.get_ports_of_type(buffer_specs['I'], 'DATA', ["in"]):
            connected_ins += utils.connect_inst_ports_by_name(
                s, "input_datain", s.input_act_modules, port["name"])
        utils.AddInPort(s, utils.get_sum_datatype_width(buffer_specs['W'],
                                                        'DATA', ["in"]),
                        "weight_datain")
        for port in utils.get_ports_of_type(buffer_specs['W'], 'DATA', ["in"]):
            connected_ins += utils.connect_inst_ports_by_name(s,
                                                              "weight_datain",
                                                              s.weight_modules,
                                                              port["name"])
        for inst in [s.activation_function_modules, s.mlb_modules,
                     s.mlb_modules, s.input_act_modules,
                     s.weight_interconnect, s.output_ps_interconnect,
                     s.input_interconnect, s.weight_modules]:
            for port in (inst.get_input_value_ports()):
                if (port._dsl.my_name not in s.__dict__.keys()) and \
                   (port not in connected_ins):
                    utils.connect_in_to_top(s, port, inst._dsl.my_name + "_" +
                                            port._dsl.my_name + "_top")
