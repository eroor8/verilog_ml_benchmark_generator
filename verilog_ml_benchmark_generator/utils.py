"""Utility functions"""
import math
import re
import sys
import module_helper_classes
from pymtl3 import connect, Wire, InPort, OutPort
input_order = [['C'], ['B'], ['PX'], ['G'], ['RY'], ['PY']]


"""
=============================================================================
Utility functions for printing warnings, info messages etc.
=============================================================================
"""


def print_warning(indent, string):
    """ Print a warning in orange

        :param indent: level of indentation to print at
        :param string: string to print
    """
    printi(indent, "WARNING: " + string, "ORANGE")


def print_mapping(mapping, indent):
    """ Print a layer nicely

        :param mapping: unrolling factors to be printed
        :param indent: level of indentation to print at
    """
    i = {"RX": mapping["RXI"],
         "RY": mapping["RYI"],
         "E": mapping["EI"],
         "C": mapping["CI"],
         "B": mapping["BI"],
         "PX": mapping["PXI"],
         "PY": mapping["PYI"]}
    o = {"RX": mapping["RXO"],
         "RY": mapping["RYO"],
         "E": mapping["EO"],
         "C": mapping["CO"],
         "B": mapping["BO"],
         "PX": mapping["PXO"],
         "PY": mapping["PYO"]}
    t = {"RX": mapping["RXT"],
         "RY": mapping["RYT"],
         "E": mapping["ET"],
         "C": mapping["CT"],
         "B": mapping["BT"],
         "PX": mapping["PXT"],
         "PY": mapping["PYT"]}

    return ("\n" + "\t" * indent + "Intra-PE unrolling factors: " + str(i) +
            "\n" + "\t" * indent + "Inter-PE unrolling factors: " + str(o) +
            "\n" + "\t" * indent + "Temporal tiling factors: " + str(t))


def print_heading(string, step=-1):
    """ Print something out on the terminal in green with underlines

        :param string: heading to be printed
        :param step: step number
    """
    to_print = '\n===> '
    if (step > -1):
        to_print += "Step " + str(step) + ": "
    to_print += string + "\n"
    to_print += ("=" * len(to_print))
    printi(1, to_print, "BLUE")


def printi(level, string, colour="None"):
    """ Print something out with indents

        :param level: intentation level to print at
        :param string: string to be printed
        :param colour: colour to print with
    """
    col_code = ''
    if (colour) == "ORANGE":
        col_code = '\033[31m'
    elif (colour) == 'GREEN':
        col_code = '\033[32m'
    # elif (colour) == 'RED':
    #     col_code = '\033[31m'
    elif (colour) == 'BLUE':
        col_code = '\033[34m'
    print(('\t' * level) + col_code + str(string) + '\033[0m')


def print_table(title, table, indent=0, col_width=0):
    """ Print a table nicely on the command line.

        :param table: table to be printed
        :param title: table name
        :param col_width: Fixed column width, optional
    """

    # Function to print a line in the table
    def printline(row, col_widths):
        orig_str = ""
        prev_width = 0
        for j in range(max(map(lambda i: 1 if (type(i) in [int, str])
                               else len(i), row))):
            for i in range(min(len(row), len(col_widths))):
                orig_str += '{:<' + str(col_widths[i])+'s}'
                if type(row[i]) in [int, str]:
                    if j == 0:
                        orig_str = orig_str.format(str(row[i]))
                    else:
                        orig_str = orig_str.format("")
                else:
                    curr_key = str(list(row[i].keys())[j])
                    curr_value = str(row[i][curr_key])
                    orig_str = orig_str.format(curr_key + " = " +
                                               curr_value)
                prev_width += col_widths[i]
            orig_str += "\n" + "\t" * indent
        return orig_str

    def get_item_len(item):
        if type(item) in [int, str]:
            return len(str(item))
        else:
            return max(map(lambda i: len(str(i)), item)) + 3

    widths = [col_width]*max(map(lambda i: len(str(i)), table))
    if (col_width == 0):
        for line in table:
            for i in range(len(line)):
                itemlen = get_item_len(line[i])
                if (itemlen+4) > widths[i]:
                    widths[i] = itemlen+4
    string_to_print = ("\t"*indent + '-' * sum(widths)) + "\n"
    string_to_print = "\n" + string_to_print + "\t"*indent + title + "\n" + \
                      string_to_print + "\t"*indent
    for line in table:
        string_to_print += printline(line, widths)
    return string_to_print


"""
=============================================================================
Utility functions for extracting information from projection vectors
=============================================================================
"""


def get_var_product(projection, var_array):
    """ Multiply a subset of projection factors (new version)

        :param projection: unrolling factor vector
        :param var_array: keys to multiply
    """
    product = 1
    for var in var_array:
        if var[0] in projection:
            assert var[0] in projection
            if var[0] in projection:
                if len(var) == 1:
                    product *= int(projection[var[0]])
                elif var[1] in projection[var[0]]:
                    product *= int(projection[var[0]][var[1]])
    return product


def get_mlb_count(projection):
    """ Calculate the number of ML Blocks required based on the projection
        definition (``projections``)

        :param projection: unrolling factor vector
    """
    return get_var_product(projection, [['RY'], ['C'], ['RX'],
                                        ['B'], ['PX'], ['PY'], ['E'],
                                        ['G']])


def get_proj_stream_count(projection, dtype='WIO'):
    """ Calculate the number of input or output streams required in total.
        (The total width is this number * the total width of that datatype
        at the inner instance boundary)

        :param projection: unrolling factor vector
        :param dtype: datatype considered - all by default
    """
    assert re.search(r"^[WIO]*$", dtype), "Unknown type " + dtype
    sum = 0
    if ('W' in dtype):
        ws = get_var_product(projection, [['RX'], ['RY'], ['C'],
                                          ['E'], ['G']])
        if 'PRELOAD' in projection:
            for pload in projection["PRELOAD"]:
                if pload["dtype"] == 'W':
                    ws = pload.get("bus_count", 1)
        sum += ws
    if ('I' in dtype):
        wi = get_var_product(projection, [['RY'], ['C'], ['B'], ['PX'], ['PY'],
                                          ['G']])
        if 'PRELOAD' in projection:
            for pload in projection["PRELOAD"]:
                if pload["dtype"] == 'I':
                    wi = pload.get("bus_count", 1)
        sum += wi
    if ('O' in dtype):
        wo = get_var_product(projection, [['E'], ['B'], ['PX'], ['PY'],
                                          ['G']])
        if 'PRELOAD' in projection:
            for pload in projection["PRELOAD"]:
                if pload["dtype"] == 'O':
                    wo = pload.get("bus_count", 1)
        sum += wo
    return sum


def get_activation_function_name(projection):
    """ Lookup the name of the activation function in the projection yaml
        schema (``projections``).

        :param projection: Projection definition
    """
    assert 'activation_function' in projection, \
        "Key activation_function not found in projection definition"
    return projection['activation_function']


def get_overall_idx_new(projection, idxs,
                        order=['RX', 'RY', 'C', 'E', 'B', 'PX', 'PY', 'G']):
    """ Calculate the inner block instance number based on loop unrolling
        factors

        :param projection: unrolling factor vector
        :param idxs: Unrolling factors to specify some instance
    """
    product = 1
    total = 0
    for item in order:
        if item[0] in idxs:
            curr_val = idxs[item[0]]
            projection_val = projection.get(item[0], 1)
            assert curr_val >= 0
            total += product * curr_val
            product *= projection_val
    return total


def get_overall_idx(projection, idxs,
                    order=['RX', 'RY', 'C', 'E', 'B', 'PX', 'PY', 'G']):
    """ Calculate the inner block instance number based on loop unrolling
        factors

        :param projection: unrolling factor vector
        :param idxs: Unrolling factors to specify some instance
    """
    product = 1
    total = 0
    for item in idxs:
        assert item in order
    for item in order:
        if item in idxs:
            assert item in projection
            val = projection[item]
            assert idxs[item] < val
            assert idxs[item] >= 0
            total += product*idxs[item]
            product *= val
    return total


"""
=============================================================================
Utility functions for extracting information from block specifications
=============================================================================
"""


def get_ports_of_type(hw_spec, type, dir=["in", "out"]):
    """ Find and return all input ports of given type from the hardware
        specification (``hw_spec``)

        :param hw_spec: Hardware definition
        :param type: Type to match
        :param dir: port directions to include - both by default
    """
    return filter(lambda port: (port['type'] == type) and
                  (port["direction"] in dir), hw_spec['ports'])


def get_sum_datatype_width(hw_spec, type, dir=['in', 'out']):
    """ Calculate the total sum of widths of ports of type ``type`` and
        with direction ``dir``

        :param hw_spec: Hardware definition
        :param type: Type to match
        :param dir: port directions to include - both by default
    """
    return sum(map(lambda port: port['width']
                   if ((port['type'] == type) and
                       (port['direction'] in dir)) else 0,
                   hw_spec['ports']))


def get_num_buffers_reqd(buffer_spec, stream_count, stream_width,
                         max_buf_width=10000000):
    """ Calculate the number of buffers required given IO requirements

    :param buffer_spec: Buffer hardware definition (according to hardware
                        yaml schema)
    :param stream_count: Number of streams that need to connect to buffers.
                         Each stream must connect to a single buffer and
                         isn't split between them.
    :param stream_width: Bit-width of each stream
    """
    buffer_width = get_sum_datatype_width(buffer_spec, "DATA", ["in"])
    buffer_width = min(max_buf_width, buffer_width)

    streams_per_buffer = math.floor(buffer_width / stream_width)
    buf_per_stream = math.ceil(stream_width / buffer_width)
    if (streams_per_buffer > 0):
        buf_count = math.ceil(stream_count / streams_per_buffer)
    else:
        buf_count = buf_per_stream * stream_count
    assert get_sum_datatype_width(buffer_spec, 'DATA', ["in"]) > 0, \
        "Buffer DATAOUT port width of zero"
    assert(buf_count > 0)
    return buf_count


def get_max_input_bus_width(buf_width, projection, data_type, inner_width=-1):
    """ Calculate the width of the values stored in each input buffer.
        Typically this would be the width of the buffer, unless convolving
        over y-indices requires for them to be stored in different buffers.

        :param buf_width: Width of the buffers
        :param projection: Width of the input buffers
        :param datatype: Type of data stored in the buffers
        :param inner_width: Width of the values stored in buffers
    """
    max_vals_per_buf = get_var_product(
        projection.get("inner_projection", {}),
        [['B'], ['PY'], ['PX'], ['G'], ['C']])
    if inner_width < 0:
        inner_width = projection["data_widths"][data_type]
    urny = projection.get('inner_projection', {}).get('RY', 1)
    if (urny > 1) and (data_type == "I"):
        buf_width = min(inner_width*max_vals_per_buf, buf_width)
    return buf_width


def get_iw_buffer_dimensions(buf_spec, projection, data_type):
    """ Find the required number and sizes of buffers """
    stream_count = get_proj_stream_count(projection["outer_projection"],
                                         data_type)
    values_per_stream = get_proj_stream_count(projection["inner_projection"],
                                              data_type)

    stream_bitwidth = values_per_stream * \
        projection["data_widths"][data_type]
    buf_width = get_max_input_bus_width(get_sum_datatype_width(buf_spec,
                                                               "DATA",
                                                               ["in"]),
                                        projection, data_type)
    streams_per_buf = math.floor(buf_width / stream_bitwidth)
    buf_per_stream = math.ceil(stream_bitwidth / buf_width)
    if (streams_per_buf > 0):
        buf_count = math.ceil(stream_count / streams_per_buf)
        values_per_buf = min(streams_per_buf * values_per_stream,
                             values_per_stream * stream_count)
    else:
        buf_count = buf_per_stream * stream_count
        values_per_buf = math.ceil(values_per_stream / buf_per_stream)
    buf_len = 2 ** get_sum_datatype_width(buf_spec, "ADDRESS", ["in"])
    return (values_per_buf, buf_len, buf_count)


def get_obuffer_dimensions(buf_spec, projection):
    """ Find the required number and sizes of output buffers """
    stream_count = \
        get_proj_stream_count(projection["outer_projection"], 'O') * \
        get_proj_stream_count(projection["inner_projection"], 'O')
    activation_width = projection["data_widths"]["I"]
    values_per_buf = (math.floor(get_sum_datatype_width(buf_spec, "DATA",
                                                        ["in"]) /
                                 activation_width))
    buf_count = math.ceil(stream_count / values_per_buf)
    buf_len = 2 ** get_sum_datatype_width(buf_spec, "ADDRESS", ["in"])
    return (values_per_buf, buf_len, buf_count)


def read_out_stored_values_from_emif(emif_inst, bytes_per_word,
                                     emif_size, dwidth, startaddr=0,
                                     words_per_buffer=sys.maxsize):
    """ Return the contents of the emif instance """
    emif_array = [0 for i in range(emif_size + startaddr)]
    for i in range(startaddr, emif_size + startaddr):
        currvalue = getattr(emif_inst, "V" + str(i))
        emif_array[i] = int(currvalue.dataout)
    return read_out_stored_values_from_array(emif_array, bytes_per_word,
                                             emif_size, dwidth, startaddr,
                                             words_per_buffer)


"""
=============================================================================
Utility functions for adding to pyMTL models
=============================================================================
"""


def AddWire(s, width, newname):
    """ Create a new wire at level ``s`` (if it doesn't already
        exist), and name it ``newname``
        Do nothing for clk and reset ports.

        :param s: Module at which to create a new port
        :param newname: Name of port to be created
    """
    if newname in s.__dict__.keys():
        return getattr(s, newname)
    else:
        neww = Wire(width)
        setattr(s, newname, neww)
        return neww


def AddInPort(s, width, newname):
    """ Create a new input port at level ``s`` (if it doesn't already
        exist), and name it ``newname``
        Do nothing for clk and reset ports.

        :param s: Module at which to create a new port
        :param newname: Name of port to be created
    """
    if newname in s.__dict__.keys():
        return getattr(s, newname)
    else:
        newinport = InPort(width)
        setattr(s, newname, newinport)
        # Tie it off in case we don't use it
        tie_off_port(s, newinport)
        return newinport


def AddOutPort(s, width, newname):
    """ Create a new output port at level ``s`` (if it doesn't already
        exist), and name it ``newname``
        Do nothing for clk and reset ports.

        :param s: Module at which to create a new port
        :param newname: Name of port to be created
    """
    if newname in s.__dict__.keys():
        return getattr(s, newname)
    else:
        newoutport = OutPort(width)
        setattr(s, newname, newoutport)
        return newoutport


def connect_in_to_top(s, port, newname):
    """ Create a new input port at level ``s`` (if it doesn't already
        exist), and connect ``port`` to the new top level port.
        Do nothing for clk and reset ports.

        :param s: Module at which to create a new port
        :param port: port of module instance instantiated within ``s``
        :param newname: Name of port to be created
    """
    if port._dsl.my_name == "clk" or port._dsl.my_name == "reset":
        return
    newinport = AddInPort(s, port._dsl.Type, newname)
    port //= newinport


def connect_out_to_top(s, port, newname):
    """ Create a new output port at level ``s``, and connect ``port``
        to the new top level port, unless it already exists

        :param s: Module at which to create a new port
        :param port: port of module instance instantiated within ``s``
        :param newname: Name of port to be created
    """
    newoutport = AddOutPort(s, port._dsl.Type, newname)
    newoutport //= port


def add_n_wires(s, n, width, prefix, start_idx=0):
    """ Create ``n`` new inputs on module ``s``, with width ``width``.
        Ports are named ``prefix``_i, where i begins at ``start_idx``.

        :param s: Module at which to create new ports
        :param n: Number of ports to add
        :param width: Bit-width of new ports
        :param start_idx: Index at which to start for port naming
        :param prefix: Prefix of names of new ports
    """
    added_w = []
    for i in range(start_idx, start_idx+n):
        added_w += [AddWire(s, width, prefix + str(i))]
    return added_w


def add_n_inputs(s, n, width, prefix, start_idx=0):
    """ Create ``n`` new inputs on module ``s``, with width ``width``.
        Ports are named ``prefix``_i, where i begins at ``start_idx``.

        :param s: Module at which to create new ports
        :param n: Number of ports to add
        :param width: Bit-width of new ports
        :param start_idx: Index at which to start for port naming
        :param prefix: Prefix of names of new ports
    """
    added_ins = []
    for i in range(start_idx, start_idx+n):
        added_ins += [AddInPort(s, width, prefix + str(i))]
    return added_ins


def add_n_outputs(s, n, width, prefix, start_idx=0):
    """ Create ``n`` new outputs on module ``s``, with width ``width``.
        Ports are named ``prefix``_i, where i begins at ``start_idx``.

        :param s: Module at which to create new ports
        :param n: Number of ports to add
        :param width: Bit-width of new ports
        :param start_idx: Index at which to start for port naming
        :param prefix: Prefix of names of new ports
    """
    added_outs = []
    for i in range(start_idx, start_idx+n):
        added_outs += [AddOutPort(s, width, prefix + str(i))]
    return added_outs


def tie_off_clk_reset(s):
    """ Create internal ports and connect them to clk and reset inputs
        This is helpful because ODIN will error out otherwise.

        :param s: Module at which to tie off clk and reset
    """
    tie_off_port(s, s.clk)
    tie_off_port(s, s.reset)


def chain_ports(s, start_idx, end_idx, in_name, out_name, width=8):
    """" Chain ports together by name
         eg. in_1 -> out2, in2 -> out3 and so on.

        :param s: Parent pyMTL module
        :param start_idx: index of port to start chain at
        :param end_idx: index of port to end chain at
        :param in_name: name of input port
        :param out_name: name of output port
        :param width: port widths
    """
    for idx in range(start_idx, end_idx):
        p1name = in_name.format(str(idx))
        p2name = out_name.format(str(idx+1))
        p1 = AddInPort(s, width, p1name)
        p2 = AddOutPort(s, width, p2name)
        connect(p1, p2)
    p1name = out_name.format(str(start_idx))
    p2name = in_name.format(str(end_idx))
    return AddOutPort(s, width, p1name), AddInPort(s, width, p2name)


def tie_off_port(s, port):
    """ Create internal wire and connect it to ``port``
        This is helpful because ODIN will error out otherwise.

        :param s: Module at which to tie off clk and reset
        :param port: Port to tie off
    """
    newwire = Wire(port._dsl.Type.nbits)
    setattr(s, port._dsl.my_name + "_tieoff", newwire)
    newwire //= port


def flatten_array(input_array, data_width):
    """ Flatten a multi-dimensional array

        :param input_array: array to be flattened
        :param data_width: width of values in flattened array
    """
    return [sum((lambda i: inner[i] * (2 ** (i * data_width)))(i)
                for i in range(len(inner)))
            for outer in input_array
            for inner in outer]


def mux_ports_by_name(s, srcs, name1, inst2, name2, factor1=1, factor2=1,
                      insel={}, sim=False, idx=''):
    """ Mux between ports named ``name1``_<#*``factor1``> on ``src``
        to ports named ``name2``_<#*``factor2``> on inst2.

        :param src: Module instance with output ports to be connected
        :param inst2: Module instance with input ports to be connected
        :param name1: Prefix of names of output ports of ``src``
        :param name2: Prefix of names of input ports of ``inst2``
        :param factor1: Factor used to match port indexes
                        (p1[i*factor1] <==> p2[i*factor2])
        :param factor2: Factor used to match port indexes
                        (p1[i*factor1] <==> p2[i*factor2])
    """
    match_dict = {}
    connected_ins = []
    connected_outs = []
    common_ports = []

    # Make a list of matching source ports.
    for src in srcs:
        for port in src.get_output_value_ports():
            port1name = port._dsl.my_name
            foundname1 = re.search("^" + name1+r"$", port1name)
            if foundname1:
                if (len(foundname1.groups()) > 0):
                    matching_name = str(int(foundname1.group(1))*factor1)
                    if matching_name in match_dict:
                        match_dict[matching_name] += [port]
                    else:
                        match_dict[matching_name] = [port]
                else:
                    common_ports += [port]

    assert (len(match_dict) > 0) or (len(common_ports) == len(srcs)), \
        "Should have found outputs with name " + name1 + " in " + \
        str(srcs[0].get_output_value_ports()) + " and " + str(common_ports)

    # For each output, add a mux between all inputs and connect it.
    for port in inst2.get_input_value_ports():
        port2name = port._dsl.my_name
        foundname2 = re.search("^" + name2+r"$", port2name)

        if foundname2:
            if (len(foundname2.groups()) > 0):
                name_to_get = str(int(foundname2.group(1))*factor2)
            else:
                name_to_get = -1
            inports = match_dict.get(name_to_get, common_ports)
            assert(len(inports) > 0)
            if (len(inports) > 1):
                muxn_inst = module_helper_classes.MUXN(port._dsl.Type.nbits,
                                                       len(inports), sim=sim)
                setattr(s, port2name+"mux"+idx, muxn_inst)
                assert (len(inports) > 1)
                # if (len(inports) < 2):
                #     muxn_inst.sel //= 0
                # else:
                muxn_inst.sel //= insel
                for i in range(len(inports)):
                    muxin = getattr(muxn_inst, "in"+str(i))
                    muxin //= inports[i]
                connectport = muxn_inst.out
            else:
                connectport = inports[0]
            connect(connectport, port)
            connected_ins += [port]
            connected_outs += [connectport]
            if (name_to_get in match_dict):
                del match_dict[name_to_get]
    assert len(match_dict) == 0, "Missing matches for ports " + \
        str(match_dict) + " in list " + str(inst2.get_input_value_ports())
    return connected_ins + connected_outs


def connect_ports_by_name(inst1, name1, inst2, name2, factor1=1, factor2=1):
    """ Connect ports named ``name1``_<#*``factor1``> on ``inst1``
        to ports named ``name2``_<#*``factor2``> on

        :param inst1: Module instance with output ports to be connected
        :param inst2: Module instance with input ports to be connected
        :param name1: Prefix of names of output ports of ``inst1``
        :param name2: Prefix of names of input ports of ``inst2``
        :param factor1: Factor used to match port indexes
                        (p1[i*factor1] <==> p2[i*factor2])
        :param factor2: Factor used to match port indexes
                        (p1[i*factor1] <==> p2[i*factor2])
    """
    match_dict = {}
    connected_ins = []
    connected_outs = []
    common = None

    # Iterate through ports on the first instance and add them to a list.
    for port in inst1.get_output_value_ports():
        port1name = port._dsl.my_name
        foundname1 = re.search("^" + name1+r"$", port1name)
        if foundname1:
            if (len(foundname1.groups()) > 0):
                match_dict[str(int(foundname1.group(1)) * factor1)] = port
            else:
                common = port

    assert (len(match_dict) > 0) or common, \
        "Should have found outputs with name " + \
        name1 + " in " + str(inst1.get_output_value_ports()) + \
        " and " + str(common)

    # Iterate through ports on the second instance to find matches.
    for port in inst2.get_input_value_ports():
        port2name = port._dsl.my_name
        foundname2 = re.search("^" + name2+r"$", port2name)
        if foundname2:
            matching_name = str(int(foundname2.group(1)) * factor2)
            assert (matching_name in match_dict) or common, \
                "Should have found output with name equivalent to " + \
                port2name + " in " + str(match_dict)
            connectport = match_dict.get(matching_name, common)
            connectport //= port
            connected_ins += [port]
            connected_outs += [connectport]
            if (matching_name in match_dict):
                del match_dict[matching_name]
    assert len(match_dict) == 0, "Missing matches for ports " + \
        str(match_dict) + " in list " + str(inst2.get_input_value_ports())
    return connected_ins + connected_outs


def connect_inst_ports_by_name(parent, namep, inst, namei,
                               factor1=1, factor2=1, parent_in=1):
    """ Connect ports named ``namei``_<#*``factor1``> on ``inst``
        to ports named ``name2``_<#*``factor2``> on the top level.

        :param inst: Module instance with output ports to be connected
        :param parent: Parent module
        :param namei: Prefix of names of output ports of ``inst``
        :param namep: Prefix of names of input ports of ``parent``
        :param factor1: Factor used to match port indexes
                        (p1[i*factor1] <==> p2[i*factor2])
        :param factor2: Factor used to match port indexes
                        (p1[i*factor1] <==> p2[i*factor2])
    """
    instports = list(inst.get_output_value_ports()) + \
        list(inst.get_input_value_ports())
    foundport = 0
    connected_ins = []

    # Iterate through ports of the instance and find the corresponding
    # port at the parent level.
    for port in instports:
        foundname1 = re.search("^" + namei+r"_(\d+)$", port._dsl.my_name)
        if foundname1:
            foundport = True
            parentport = None
            # Check if there is a matching port on the parent.
            if namep + "_" + foundname1.group(1) in parent.__dict__.keys():
                parentport = getattr(parent, namep + "_" +
                                     foundname1.group(1))
            elif namep in parent.__dict__.keys():
                parentport = getattr(parent, namep)

            # If there is no matching port, add one.
            if not (parentport):
                newname = namep + "_" + foundname1.group(1)
                if (parent_in):
                    parentport = AddInPort(parent, port._dsl.Type.nbits,
                                           newname)
                else:
                    parentport = AddOutPort(parent, port._dsl.Type.nbits,
                                            newname)
            if (port._dsl.Type.nbits > 1):
                connect(parentport[0:port._dsl.Type.nbits], port)
            else:
                connect(parentport, port)
            connected_ins += [port]
    assert foundport, "Port " + namei + " not found in " + str(instports)
    return connected_ins


def mux_inst_ports_by_name(inst2, name2, srcs, name1, factor1=1, factor2=1,
                           insel={}, idx='', sim=True):
    """ Mux between ports named ``name1``_<#*``factor1``> on ``srcs``
        to ports named ``name2``_<#*``factor2``>.

        :param src: Module instance with output ports to be connected
        :param inst2: Module instance with input ports to be connected
        :param name1: Prefix of names of output ports of ``src``
        :param name2: Prefix of names of input ports of ``inst2``
        :param factor1: Factor used to match port indexes
                        (p1[i*factor1] <==> p2[i*factor2])
        :param factor2: Factor used to match port indexes
                        (p1[i*factor1] <==> p2[i*factor2])
    """
    match_dict = {}
    connected_ins = []
    connected_outs = []
    for src in srcs:
        for port in src.get_output_value_ports():
            port1name = port._dsl.my_name
            foundname1 = re.search("^" + name1+r"$", port1name)
            if foundname1:
                if (len(foundname1.groups()) > 0):
                    matching_name = str(int(foundname1.group(1))*factor1)
                    if matching_name in match_dict:
                        match_dict[matching_name] += [port]
                    else:
                        match_dict[matching_name] = [port]
                else:
                    if "c" in match_dict:
                        match_dict["c"] += [port]
                    else:
                        match_dict["c"] = [port]
    assert (len(match_dict) > 0), "Should have found outputs with name " + \
        name1 + " in " + str(srcs[0].get_output_value_ports())

    for port in match_dict:
        parentname = name2+"_"+port
        parentport = None
        # if parentname in inst2.__dict__.keys():
        #     parentport = getattr(inst2, name2 + "_" + foundname1.group(1))
        # elif name2 in inst2.__dict__.keys():
        assert(name2 in inst2.__dict__.keys())
        parentport = getattr(inst2, name2)
        inports = match_dict[port]
        muxn_inst = module_helper_classes.MUXN(parentport._dsl.Type.nbits,
                                               len(inports), sim)
        setattr(inst2, parentname+"mux"+idx, muxn_inst)
        if (len(inports) < 2):
            muxn_inst.sel //= 0
        else:
            muxn_inst.sel //= insel
        for i in range(len(inports)):
            muxin = getattr(muxn_inst, "in"+str(i))
            inportw = inports[i]._dsl.Type.nbits
            muxin[0:inportw] //= inports[i]
        connectport = muxn_inst.out
        connect(connectport, parentport[0:parentport._dsl.Type.nbits])
        connected_ins += [parentport]
        connected_outs += [connectport]
    return connected_ins + connected_outs


"""
=============================================================================
Utility functions for validating outputs and buffer contents
=============================================================================
"""


def read_out_stored_values_from_array(array,
                                      values_per_word,
                                      emif_size,
                                      dwidth, startaddr=0,
                                      words_per_buffer=sys.maxsize):
    """ Return the contents of the EMIF instance

        :param array: array to read out from
        :param values_per_word: values in each word of array
        :param dwidth: Width of stored data
        :param startaddr: Address to start reading at
        :param words_per_buffer: Number of words to read
    """
    buffer_values = []
    curr_buffer = []
    for i in range(startaddr, emif_size + startaddr):
        curr_obuf_out = array[i]
        curr_buffer_vals = []
        for section in range(values_per_word):
            curr_buffer_vals += [int(curr_obuf_out % (2 ** dwidth))]
            curr_obuf_out = curr_obuf_out // (2 ** dwidth)
        curr_buffer += [curr_buffer_vals]
        if ((i - startaddr + 1) % words_per_buffer == 0):
            buffer_values += [curr_buffer]
            curr_buffer = []

    if (words_per_buffer == sys.maxsize):
        return curr_buffer
    else:
        return buffer_values + [curr_buffer]


def compute_layer(inputs, weights, layer):
    """ Calculate expected outputs of given FC or convolutional layer

        :param inputs: Input activation array
        :param weights: Filter weight array
        :param layer: Information about FC or conv layer (ie dimensions)
    """
    urx = layer.get("filter_x", 1)
    ury = layer.get("filter_y", 1)
    urc = layer.get("in_chans", 1)
    ue = layer.get("out_chans", 1)
    ub = layer.get("batches", 1)
    ubx = layer.get("image_x", 1)
    uby = layer.get("image_y", 1)
    ug = layer.get("group", 1)
    stridex = layer.get("stridex", 1)
    stridey = layer.get("stridey", 1)
    dilx = layer.get("dilx", 1)
    dily = layer.get("dily", 1)

    outputs = [[[[[0 for k in range(int(ubx / stridex))]  # x
                  for i in range(int(uby / stridey))]     # y
                 for j in range(ue)]                      # chans
                for p in range(ub)]                       # batch
               for t in range(ug)]                        # group

    for (ugi, ubi, uei, urci, urxi, uryi, g, h) in range8D(
            ug, ub, ue, urc, urx, ury):
        for ubxi in range(urx-1, ubx, stridex):           # px x
            for ubyi in range(ury - 1, uby, stridey):     # px y
                if ((urxi % dilx == 0) and (uryi % dily == 0)):
                    y_idx_in = ubyi + 1 - ury + uryi
                    inact = inputs[ugi][ubi][urci][y_idx_in][ubxi - urxi]
                    weight = weights[ugi][uei][urci][uryi][urxi]
                    y_idx_out = int((ubyi - ury + 1) / stridey)
                    x_idx_out = int((ubxi - urx + 1) / stridex)
                    wi = inact * weight
                    outputs[ugi][ubi][uei][y_idx_out][x_idx_out] += wi
    return outputs


def get_expected_outputs(obuf, ostreams_per_buf, wbuf, ibuf,
                         ivalues_per_buf, projection):
    """ Calculate the expected contents of the output buffer
        based on the contents of the input and weight buffers.

        :param obuf: output buffer to be filled
        :param ostreams_per_buf: number of output streams per buffer
        :param wbuf: array of filter weights
        :param ibuf: array of input activations
        :param ivalues_per_buf: number of streams per input buffer
        :param projection: unrolling factor vector
    """
    obuf_len = len(obuf[0])
    wbuf_len = len(wbuf[0])
    ibuf_len = len(ibuf[0])

    # Get unrolling factors
    inner_uw = projection["inner_projection"]["RX"]
    inner_un = projection["inner_projection"]["RY"] * \
        projection["inner_projection"]["C"]
    inner_ue = projection["inner_projection"]["E"]
    inner_ub = projection["inner_projection"]["B"] * \
        projection["inner_projection"]["PX"] * \
        projection["inner_projection"]["PY"]
    inner_ug = projection["inner_projection"]["G"]
    outer_uw = projection["outer_projection"]["RX"]
    outer_un = projection["outer_projection"]["RY"] * \
        projection["outer_projection"]["C"]
    outer_ue = projection["outer_projection"]["E"]
    outer_ub = projection["outer_projection"]["B"] * \
        projection["outer_projection"]["PX"] * \
        projection["outer_projection"]["PY"]
    outer_ug = projection["outer_projection"]["G"]
    temp_proj = projection.get("temporal_projection", {})
    temp_ug = temp_proj.get("G", 1)
    temp_ub = temp_proj.get("B", obuf_len) * temp_proj.get("PX", 1) * \
        temp_proj.get("PY", 1)
    temp_un = temp_proj.get("RY", 1) * temp_proj.get("C", 1)
    temp_ue = temp_proj.get("E", 1)
    for (ugt, uet, ugo, ubo, ueo, ugi, ubi, uei) in range8D(
            temp_ug, temp_ue, outer_ug, outer_ub, outer_ue, inner_ug,
            inner_ub, inner_ue):
        for ubt in range(outer_uw * inner_uw - 1, temp_ub):
            # Accumulate a partial sum
            correct_sum = 0
            for (urno, urni, urnt, urwo, urwi, f, g, h) in range8D(
                    outer_un, inner_un, temp_un, outer_uw, inner_uw):
                # Find the corresponding weight in the weight buffers
                if ("PRELOAD" in projection["inner_projection"]):
                    mlb_chain_len = inner_ug * inner_ue * inner_un * inner_uw
                    w_buf_inst_idx = \
                        ugi * inner_ue * inner_un * inner_uw + \
                        uei * inner_un * inner_uw + urni * inner_uw + urwi
                    bus_idx = 0
                    stream_width = 1
                else:
                    mlb_chain_len = 1
                    w_buf_inst_idx = 0
                    bus_idx = ugi * inner_ue * inner_un * inner_uw + \
                        uei * inner_un * inner_uw + urni * inner_uw + urwi
                    stream_width = inner_ug * inner_ue * inner_un * inner_uw

                if ("PRELOAD" in projection["outer_projection"]):
                    w_buf_inst_idx = (ugo * outer_ue * outer_un * outer_uw +
                                      ueo * outer_un * outer_uw +
                                      urno * outer_uw +
                                      urwo) * mlb_chain_len + \
                        w_buf_inst_idx
                    outer_chain_len = (outer_ug * outer_ue * outer_uw *
                                       outer_un)
                    buffer_cnt = 0
                else:
                    outer_chain_len = 1
                    stream_idx = ugo * outer_ue * outer_un * outer_uw + \
                        ueo * outer_un * outer_uw + \
                        urno * outer_uw + \
                        urwo
                    streams_per_buffer = math.floor(len(wbuf[0][0]) /
                                                    stream_width)
                    buffer_cnt = math.floor(stream_idx / streams_per_buffer)
                    bus_idx = (stream_idx % streams_per_buffer) * \
                        stream_width + bus_idx

                urw = urwo * inner_uw + urwi
                buffer_idx = (outer_chain_len * mlb_chain_len -
                              w_buf_inst_idx - 1)
                buffer_idx += ugt * temp_ue * temp_un + uet * temp_un
                total_b_idx = (buffer_idx + urnt) % wbuf_len
                w = wbuf[buffer_cnt][total_b_idx][bus_idx]

                # Now find the corresponding input activation value
                i_stream_idx = get_overall_idx_new(
                    projection["outer_projection"],
                    {'C': urno, 'B': ubo, 'G': ugo}, order=input_order)
                i_value_idx = i_stream_idx * get_proj_stream_count(
                    projection["inner_projection"], 'I') + \
                    get_overall_idx_new(projection["inner_projection"],
                                        {'C': urni, 'B': ubi, 'G': ugi},
                                        order=input_order)
                ibuf_idx = math.floor(i_value_idx / ivalues_per_buf)
                iv_idx = i_value_idx % ivalues_per_buf

                # Add to the current partial sum
                it_idx = (ugt * temp_ub * temp_un+ubt*temp_un + urnt -
                          urw) % ibuf_len
                correct_sum += (ibuf[ibuf_idx][it_idx][iv_idx] * w)

            # Find the corresponding location in the output buffers
            out_act_idx = ugo * outer_ub * outer_ue * inner_ug * \
                inner_ub * inner_ue + \
                ubo * outer_ue * inner_ug * inner_ub * inner_ue + \
                ueo * inner_ug * inner_ub * inner_ue + \
                ugi * inner_ub * inner_ue + ubi * inner_ue + uei
            obuf_idx = math.floor(out_act_idx / ostreams_per_buf)
            os_idx = out_act_idx % ostreams_per_buf
            ot_idx = ugt * temp_ub * temp_ue + uet * temp_ub + ubt - \
                outer_uw * inner_uw + 1
            obuf[obuf_idx][ot_idx][os_idx] = correct_sum % \
                (2 ** projection["data_widths"]["I"])
    return obuf


def merge_bus(v, width):
    """ Given a list of values, combine them into a single value

        :param obuf: list of values to flatten
        :param width: bitwidth of each value
    """
    sum = 0
    for i in range(len(v)):
        sum += v[i] * (2 ** (width * i))
    return sum


"""
=============================================================================
Other
=============================================================================
"""


def map_buffer_idx_to_y_idx(proj_yaml, ab_yaml=None, ibuf_count=0,
                            ivalues_per_buf=0):
    """ Map each input activation buffer to the y-indices of
        the input image that it contains.
        This is useful when connecting buffers to the appropriate
        embedded block inputs

        :param proj_yaml: unrolling factor vectors
        :param ab_yaml: specification of input buffer
        :param ibuf_count: number of input buffers
        :param ivalues_per_buf: number of streams per input buffer
    """
    if (ab_yaml):
        ivalues_per_buf, ibuf_len, ibuf_count = \
            get_iw_buffer_dimensions(ab_yaml, proj_yaml, 'I')

    output_map = [0 for buf in range(ibuf_count)]

    # Get Intra-EB unrolling factors
    i_proj = proj_yaml.get("inner_projection", {})
    inner_ug = i_proj.get("G", 1)
    inner_ubb = i_proj.get("B", 1)
    inner_ubx = i_proj.get("PX", 1)
    inner_uby = i_proj.get("PY", 1)
    inner_unc = i_proj.get("C", 1)
    inner_uny = i_proj.get("RY", 1)

    # Get Inter-EB unrolling factors
    o_proj = proj_yaml.get("outer_projection", {})
    outer_ug = o_proj.get("G", 1)
    outer_ubb = o_proj.get("B", 1)
    outer_ubx = o_proj.get("PX", 1)
    outer_uby = o_proj.get("PY", 1)
    outer_unc = o_proj.get("C", 1)
    outer_uny = o_proj.get("RY", 1)

    # Get other required values
    i_stream_count = get_proj_stream_count(proj_yaml["inner_projection"], 'I')

    # Iterate through each MLB input and map the buffer to the y-idx
    for (ugo, ubox, uboy, ubob, urnoc, urnoy, g, h) in range8D(
            outer_ug, outer_ubx, outer_uby, outer_ubb, outer_unc, outer_uny):
        for (ugi, ubix, ubiy, ubib, urnic, urniy, g, h) in range8D(
                inner_ug, inner_ubx, inner_uby, inner_ubb, inner_unc,
                inner_uny):
            for urniy in range(inner_uny):
                # Find the corresponding buffer
                i_stream_idx = get_overall_idx_new(
                    proj_yaml["outer_projection"],
                    {'RY': urnoy, 'C': urnoc, 'PX': ubox, 'PY': uboy,
                     'B': ubob, 'G': ugo}, order=input_order)
                idx_within_stream = get_overall_idx_new(
                    proj_yaml["inner_projection"],
                    {'RY': urniy, 'C': urnic, 'PX': ubix, 'PY': ubiy,
                     'B': ubib, 'G': ugi}, order=input_order)
                i_value_idx = i_stream_idx * i_stream_count + \
                    idx_within_stream
                buf_idx = math.floor(i_value_idx / ivalues_per_buf)

                # Find the corresponding y index
                inner_y = get_overall_idx_new(proj_yaml["inner_projection"],
                                              {'RY': urniy, 'PY': ubiy},
                                              order=[['RY'], ['PY']])
                outer_y = get_overall_idx_new(proj_yaml["outer_projection"],
                                              {'RY': urnoy, 'PY': uboy},
                                              order=[['RY'], ['PY']])
                total_y = outer_y * inner_uby * inner_uny + inner_y
                output_map[buf_idx] = total_y
    return output_map


def range8D(a=1, b=1, c=1, d=1, e=1, f=1, g=1, h=1):
    """ Helper function for iterating through deep nested loops

        :param a-f: loop dimensions
    """
    for ai in range(a):
        for bi in range(b):
            for ci in range(c):
                for di in range(d):
                    for ei in range(e):
                        for fi in range(f):
                            for gi in range(g):
                                for hi in range(h):
                                    yield(ai, bi, ci, di, ei, fi, gi, hi)


def range4D(a=1, b=1, c=1, d=1):
    """ Helper function for iterating through deep nested loops

        :param a-f: loop dimensions
    """
    for ai in range(a):
        for bi in range(b):
            for ci in range(c):
                for di in range(d):
                    yield(ai, bi, ci, di)


def range2D(a=1, b=1):
    """ Helper function for iterating through deep nested loops

        :param a-f: loop dimensions
    """
    for ai in range(a):
        for bi in range(b):
            yield(ai, bi)
