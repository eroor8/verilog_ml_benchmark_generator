"""Utility functions"""
import math
import re
import sys
from pymtl3 import connect, Wire, InPort, OutPort


def get_var_product(projection, var_array):
    """ Multiply a subset of projection factors
    """
    product = 1
    for var in var_array:
        assert var in projection, "Key " + var + \
               " not found in projection definition."
        product *= int(projection[var]['value'])
    return product


def get_mlb_count(projection):
    """ Calculate the number of ML Blocks required based on the projection
        definition (``projections``)
    """
    return get_var_product(projection, ['URN', 'URW', 'UB', 'UE', 'UG'])


def get_proj_chain_length(projection, dtype):
    """ Calculate cascade chain length
    """
    assert re.search(r"^[WIO]*$", dtype), "Unknown type " + dtype
    if ('W' == dtype):
        return get_var_product(projection, ['URW'])
    elif ('I' == dtype):
        return get_var_product(projection, ['URW'])
    elif ('O' in dtype):
        return get_var_product(projection, ['URW', 'URN'])
    return 0


def get_proj_stream_count(projection, dtype='WIO'):
    """ Calculate the number of input or output streams required in total.
        (The total width is this number * the total width of that datatype
        at the inner instance boundary)

    :param projection: Projection definition
                       (according to projection yaml schema)
    :type projection: dict
    :param dtype: datatype considered - all by default
    :type dtype: string
    :return: dtype
    :rtype: int
    """
    assert re.search(r"^[WIO]*$", dtype), "Unknown type " + dtype
    sum = 0
    if ('W' in dtype):
        ws = get_var_product(projection, ['URW', 'URN', 'UE', 'UG'])
        if 'PRELOAD' in projection:
            for pload in projection["PRELOAD"]:
                if pload["dtype"] == 'W':
                    ws = pload.get("bus_count", 1)
        sum += ws
    if ('I' in dtype):
        wi = get_var_product(projection, ['URN', 'UB', 'UG'])
        if 'PRELOAD' in projection:
            for pload in projection["PRELOAD"]:
                if pload["dtype"] == 'I':
                    wi = pload.get("bus_count", 1)
        sum += wi
    if ('O' in dtype):
        wo = get_var_product(projection, ['UE', 'UB', 'UG'])
        if 'PRELOAD' in projection:
            for pload in projection["PRELOAD"]:
                if pload["dtype"] == 'O':
                    wo = pload.get("bus_count", 1)
        sum += wo
    return sum


def get_activation_function_name(projection):
    """ Lookup the name of the activation function in the projection yaml
        schema (``projections``).

    :return: Activation function name
    :rtype: string
    """
    assert 'activation_function' in projection, \
        "Key activation_function not found in projection definition"
    return projection['activation_function']


def get_ports_of_type(hw_spec, type, dir=["in", "out"]):
    """ Find and return all input ports of given type from the hardware
        specification (``hw_spec``)

    :param hw_spec: Hardware definition (according to hardware yaml schema)
    :type hw_spec: dict
    :param type: Type to match
    :type type: string
    :param dir: port directions to include - both by default
    :type dir: list
    :return: Ports with type ``type``
    :rtype: list
    """
    return filter(lambda port: (port['type'] == type) and
                  (port["direction"] in dir), hw_spec['ports'])


def get_sum_datatype_width(hw_spec, type, dir=['in', 'out']):
    """ Calculate the total sum of widths of ports of type ``type`` and
        with direction ``dir``

    :param hw_spec: Hardware definition (according to hardware yaml schema)
    :type hw_spec: dict
    :param type: Type to match
    :type type: string
    :param dir: port directions to include - both by default
    :type dir: list
    :return: Sum of port widths
    :rtype: int
    """
    return sum(map(lambda port: port['width']
                   if ((port['type'] == type) and
                       (port['direction'] in dir)) else 0,
                   hw_spec['ports']))


def get_num_buffers_reqd(buffer_spec, stream_count, stream_width):
    """ Calculate the number of buffers required given IO requirements

    :param buffer_spec: Buffer hardware definition (according to hardware
                        yaml schema)
    :type buffer_spec: dict
    :param stream_count: Number of streams that need to connect to buffers.
                         Each stream must connect to a single buffer and
                         isn't split between them.
    :type stream_count: int
    :param stream_width: Bit-width of each stream
    :type stream_width: int
    :return: Number of buffers to be instantiated
    :rtype: int
    """
    streams_per_buffer = math.floor(get_sum_datatype_width(
        buffer_spec, 'DATA', ["in"]) / stream_width)
    
    assert get_sum_datatype_width(buffer_spec, 'DATA', ["in"]) > 0, \
        "Buffer DATAOUT port width of zero"
    assert streams_per_buffer > 0, "Streams per buffer=" + \
        str(get_sum_datatype_width(buffer_spec, 'DATA', ["in"])) + "/" + \
        str(stream_width)
    return math.ceil(stream_count/streams_per_buffer)


def get_overall_idx(projection, idxs):
    """ Calculate the inner block instance number based on loop unrolling
        factors

    :param projection: Projection definition
                       (according to projection yaml schema)
    :type projection: dict
    :param idxs: Unrolling factors to specify some instance
    :type idxs: dict
    :return: Instance number of specified instance
    :rtype: int
    """
    product = 1
    total = 0
    for item in idxs:
        assert item in ['URW', 'URN', 'UE', 'UB', 'UG']
    for item in ['URW', 'URN', 'UE', 'UB', 'UG']:
        if item in idxs:
            assert item in projection
            assert idxs[item] < projection[item]['value']
            assert idxs[item] >= 0
            total += product*idxs[item]
            product *= projection[item]['value']
    return total

def AddWire(s, width, newname):
    """ Create a new wire at level ``s`` (if it doesn't already
    exist), and name it ``newname``
    Do nothing for clk and reset ports.

    :param s: Module at which to create a new port
    :type s: Component class
    :param newname: Name of port to be created
    :type newname: string
    :return: The newly created port
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
    :type s: Component class
    :param newname: Name of port to be created
    :type newname: string
    :return: The newly created port
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
    :type s: Component class
    :param newname: Name of port to be created
    :type newname: string
    :return: The newly created port
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
    :type s: Component class
    :param port: port of module instance instantiated within ``s``
    :type port: Component class
    :param newname: Name of port to be created
    :type newname: string
    """
    if port._dsl.my_name == "clk" or port._dsl.my_name == "reset":
        return
    newinport = AddInPort(s, port._dsl.Type, newname)
    port //= newinport


def connect_out_to_top(s, port, newname):
    """ Create a new output port at level ``s``, and connect ``port``
    to the new top level port, unless it already exists

    :param s: Module at which to create a new port
    :type s: Component class
    :param port: port of module instance instantiated within ``s``
    :type port: Component class
    :param newname: Name of port to be created
    :type newname: string
    """
    newoutport = AddOutPort(s, port._dsl.Type, newname)
    newoutport //= port


def add_n_wires(s, n, width, prefix, start_idx=0):
    """ Create ``n`` new inputs on module ``s``, with width ``width``.
    Ports are named ``prefix``_i, where i begins at ``start_idx``.

    :param s: Module at which to create new ports
    :type s: Component class
    :param n: Number of ports to add
    :type n: int
    :param width: Bit-width of new ports
    :type width: int
    :param start_idx: Index at which to start for port naming
    :type start_idx: int
    :param prefix: Prefix of names of new ports
    :type prefix: string
    """
    added_w = []
    for i in range(start_idx, start_idx+n):
        added_w += [AddWire(s, width, prefix + str(i))]
    return added_w

def add_n_inputs(s, n, width, prefix, start_idx=0):
    """ Create ``n`` new inputs on module ``s``, with width ``width``.
    Ports are named ``prefix``_i, where i begins at ``start_idx``.

    :param s: Module at which to create new ports
    :type s: Component class
    :param n: Number of ports to add
    :type n: int
    :param width: Bit-width of new ports
    :type width: int
    :param start_idx: Index at which to start for port naming
    :type start_idx: int
    :param prefix: Prefix of names of new ports
    :type prefix: string
    """
    added_ins = []
    for i in range(start_idx, start_idx+n):
        added_ins += [AddInPort(s, width, prefix + str(i))]
    return added_ins


def add_n_outputs(s, n, width, prefix, start_idx=0):
    """ Create ``n`` new outputs on module ``s``, with width ``width``.
    Ports are named ``prefix``_i, where i begins at ``start_idx``.

    :param s: Module at which to create new ports
    :type s: Component class
    :param n: Number of ports to add
    :type n: int
    :param width: Bit-width of new ports
    :type width: int
    :param start_idx: Index at which to start for port naming
    :type start_idx: int
    :param prefix: Prefix of names of new ports
    :type prefix: string
    """
    added_outs = []
    for i in range(start_idx, start_idx+n):
        added_outs += [AddOutPort(s, width, prefix + str(i))]
    return added_outs


def tie_off_clk_reset(s):
    """ Create internal ports and connect them to clk and reset inputs
        This is helpful because ODIN will error out otherwise.

    :param s: Module at which to tie off clk and reset
    :type s: Component class
    """
    tie_off_port(s, s.clk)
    tie_off_port(s, s.reset)

def chain_ports(s, start_idx, end_idx, in_name, out_name, width=8):
    """" Chain ports together by name
         eg. in_1 -> out2, in2 -> out3 and so on. """
    for idx in range(start_idx, end_idx):
        p1name = in_name.format(str(idx))
        p2name = out_name.format(str(idx+1))
        p1 = AddInPort(s, width, p1name)
        p2 = AddOutPort(s, width, p2name)
        connect(p1, p2)
    p1name = out_name.format(str(start_idx))
    p2name = in_name.format(str(end_idx))
    return AddOutPort(s, width, p1name), AddInPort(s, width, p2name)

def print_table(title, table, indent=0, col_width=0):
    """ Print a table nicely on the command line.

    :param table: table to be printed
    :type table: array of arrays
    :param title: table name
    :type title: string
    :param col_width: Fixed column width, optional
    :type col_width: int
    """
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
            orig_str += "\n" + "\t"*indent 
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


def tie_off_port(s, port):
    """ Create internal wire and connect it to ``port``
        This is helpful because ODIN will error out otherwise.

    :param s: Module at which to tie off clk and reset
    :type s: Component class
    :param port: Port to tie off
    :type port: Component class
    """
    newwire = Wire(port._dsl.Type.nbits)
    setattr(s, port._dsl.my_name + "_tieoff", newwire)
    newwire //= port

def get_port_name(plist, t):
    """ From a list of ports, get the name of port with type t
    """
    count = 0
    retname = ""
    for port in plist:
        if (port["type"] == t):
            count += 1
            retname = port["name"]
    assert (count == 1)
    return retname

def connect_ports_by_name(inst1, name1, inst2, name2, factor1=1, factor2=1):
    """ Connect ports named ``name1``_<#*``factor1``> on ``inst1``
        to ports named ``name2``_<#*``factor2``> on

    :param inst1: Module instance with output ports to be connected
    :type inst1: Component class
    :param inst2: Module instance with input ports to be connected
    :type inst2: Component class
    :param name1: Prefix of names of output ports of ``inst1``
    :type name1: string
    :param name2: Prefix of names of input ports of ``inst2``
    :type name2: string
    :param factor1: Factor used to match port indexes
                    (p1[i*factor1] <==> p2[i*factor2])
    :type factor1: string
    :param factor2: Factor used to match port indexes
                    (p1[i*factor1] <==> p2[i*factor2])
    :type factor2: string
    """
    match_dict = {}
    connected_ins = []
    connected_outs = []
    common = None
    for port in inst1.get_output_value_ports():
        port1name = port._dsl.my_name
        foundname1 = re.search("^" + name1+r"$", port1name)
        if foundname1:
            try:
                match_dict[str(int(foundname1.group(1))*factor1)] = port
            except:
                common = port
                
    assert (len(match_dict) > 0) or common, \
        "Should have found outputs with name " + \
        name1 + " in " + str(inst1.get_output_value_ports()) + " and " + str(common)

    for port in inst2.get_input_value_ports():
        port2name = port._dsl.my_name
        foundname2 = re.search("^" + name2+r"$", port2name)
        if foundname2:
            assert (str(int(foundname2.group(1))*factor2) in match_dict) or common, \
                "Should have found output with name " + name2 + " in " + \
                str(match_dict)
            name_to_get = str(int(foundname2.group(1))*factor2)
            connectport = match_dict.get(name_to_get, common)
            connect(connectport, port)
            connected_ins += [port]
            connected_outs += [connectport]
            if (str(int(foundname2.group(1))*factor2) in match_dict):
                del match_dict[str(int(foundname2.group(1))*factor2)]
    assert len(match_dict) == 0, "Missing matches for ports " + \
        str(match_dict) + " in list " + str(inst2.get_input_value_ports())
    return connected_ins + connected_outs

def connect_inst_ports_by_name(parent, namep, inst, namei, \
                               factor1=1, factor2=1):
    """ Connect ports named ``namei``_<#*``factor1``> on ``inst``
        to ports named ``name2``_<#*``factor2``> on the top level.

    :param inst: Module instance with output ports to be connected
    :type inst: Component class
    :param parent: Parent module
    :type parent: Component class
    :param namei: Prefix of names of output ports of ``inst``
    :type namei: string
    :param namep: Prefix of names of input ports of ``parent``
    :type namep: string
    :param factor1: Factor used to match port indexes
                    (p1[i*factor1] <==> p2[i*factor2])
    :type factor1: string
    :param factor2: Factor used to match port indexes
                    (p1[i*factor1] <==> p2[i*factor2])
    :type factor2: string
    """
    instports = list(inst.get_output_value_ports()) + \
                 list(inst.get_input_value_ports())
    foundport = 0
    connected_ins = []
    for port in instports:
        port1name = port._dsl.my_name
        foundname1 = re.search("^" + namei+r"_(\d+)$", port1name)
        if foundname1:
            foundport = True
            parentport = None
            if namep + "_" + foundname1.group(1) in parent.__dict__.keys():
                parentport = getattr(parent, namep + "_" + \
                                     foundname1.group(1) )
            elif namep in parent.__dict__.keys():
                parentport = getattr(parent, namep)
            assert(parentport)
            connect(parentport, port)
            connected_ins += [port]
    assert foundport, "Port " + namei + " not found in " + str(instports)
    return connected_ins


def get_iw_buffer_dimensions(buf_spec, projection, data_type):
    """ Find the required number and sizes of buffers """
    
    stream_count = get_proj_stream_count(projection["outer_projection"],
                                         data_type)
    values_per_stream = get_proj_stream_count(projection["inner_projection"],
                                              data_type)
    stream_bitwidth = values_per_stream * projection["stream_info"][data_type]
    streams_per_buf = math.floor(
         get_sum_datatype_width(buf_spec, "DATA", ["in"]) / stream_bitwidth)
    assert (streams_per_buf > 0)
    buf_count = math.ceil(stream_count / streams_per_buf)
    values_per_buf = min(streams_per_buf * values_per_stream,
                         values_per_stream * stream_count)
    buf_len = 2 ** get_sum_datatype_width(buf_spec, "ADDRESS", ["in"])
    return (values_per_buf, buf_len, buf_count)

def get_obuffer_dimensions(buf_spec, projection):
    """ Find the required number and sizes of output buffers """
    stream_count = \
        get_proj_stream_count(projection["outer_projection"], 'O') * \
        get_proj_stream_count(projection["inner_projection"], 'O') 
    activation_width = projection["stream_info"]["I"]
    streams_per_buf = math.floor(
        get_sum_datatype_width(buf_spec, "DATA", ["in"]) / activation_width)
    buf_count = math.ceil(stream_count / streams_per_buf)
    buf_len = 2 ** get_sum_datatype_width(buf_spec, "ADDRESS", ["in"])
    return (streams_per_buf, buf_len, buf_count)

def read_out_stored_values_from_emif(emif_inst,
                                     bytes_per_word,
                                     emif_size,
                                     dwidth, startaddr=0,
                                     words_per_buffer=sys.maxsize):
    """ Return the contents of the emif instance """
    emif_array = [0 for i in range(emif_size+startaddr)]
    for i in range(startaddr,emif_size+startaddr):
        currvalue = getattr(emif_inst, "V"+str(i))
        emif_array[i] = int(currvalue.dataout)
    return read_out_stored_values_from_array(emif_array, bytes_per_word, emif_size, dwidth, startaddr, words_per_buffer)

def read_out_stored_values_from_array(array,
                                     values_per_word,
                                     emif_size,
                                      dwidth, startaddr=0,
                                      words_per_buffer=sys.maxsize):
    """ Return the contents of the emif instance """
    buffer_values = []
    curr_buffer = []
    for i in range(startaddr, emif_size + startaddr):
        curr_obuf_out = array[i]
        curr_buffer_vals = []
        for section in range(values_per_word):
            curr_buffer_vals += [int(curr_obuf_out%(2**dwidth))]
            curr_obuf_out = curr_obuf_out // (2**dwidth)
        curr_buffer += [curr_buffer_vals]
        if ((i - startaddr + 1) % words_per_buffer == 0):
            buffer_values += [curr_buffer]
            curr_buffer = []

    if (words_per_buffer == sys.maxsize):
        return curr_buffer
    else:
        return buffer_values + [curr_buffer]


def get_expected_outputs(obuf, ostreams_per_buf, wbuf, ibuf, ivalues_per_buf, projection):
    obuf_len = len(obuf[0])
    wbuf_len = len(wbuf[0])
    ibuf_len = len(ibuf[0])
    inner_uw = projection["inner_projection"]["URW"]["value"]
    inner_un = projection["inner_projection"]["URN"]["value"]
    inner_ue = projection["inner_projection"]["UE"]["value"]
    inner_ub = projection["inner_projection"]["UB"]["value"]
    inner_ug = projection["inner_projection"]["UG"]["value"]
    outer_uw = projection["outer_projection"]["URW"]["value"]
    outer_un = projection["outer_projection"]["URN"]["value"]
    outer_ue = projection["outer_projection"]["UE"]["value"]
    outer_ub = projection["outer_projection"]["UB"]["value"]
    outer_ug = projection["outer_projection"]["UG"]["value"]
    temp_ug = projection.get("temporal_projection",{}).get("UG",{}).get("value", 1)
    temp_ub = projection.get("temporal_projection",{}).get("UB",{}).get("value", obuf_len)
    temp_un = projection.get("temporal_projection",{}).get("URN",{}).get("value", 1)
    mlb_count = get_mlb_count(projection["outer_projection"])
    mac_count = get_mlb_count(projection["inner_projection"])
    for ugt in range(temp_ug):
        for ugo in range(outer_ug): 
            for ugi in range(inner_ug):
                for ubo in range(outer_ub): 
                    for ubi in range(inner_ub):
                        for ubt in range(temp_ub):
                            for ueo in range(outer_ue):
                                for uei in range(inner_ue):
                                    correct_sum = 0
                                    for urno in range(outer_un):
                                        for urni in range(inner_un):
                                            for urnt in range(temp_un):
                                                for urwo in range(outer_uw):
                                                    for urwi in range(inner_uw):
                                                        urw = urwo*inner_uw + urwi
                                                        mlb_inst = ugo*outer_ub*outer_ue*outer_un*outer_uw + \
                                                                   ubo*outer_ue*outer_un*outer_uw + \
                                                                   ueo*outer_un*outer_uw + \
                                                                   urno*outer_uw + \
                                                                   urwo
                                                        mac_idx = mlb_inst*mac_count + \
                                                                  ugi*inner_ub*inner_ue*inner_uw*inner_un + \
                                                                  ubi*inner_ue*inner_uw*inner_un + \
                                                                  uei*inner_uw*inner_un + \
                                                                  urni*inner_uw + \
                                                                  urwi
                                                        
                                                        print("MLB:"+ str(mlb_inst) + "  and mac:" + str(mac_idx))
                                                        w_buf_inst_idx = 0
                                                        buffer_idx = 0
                                                        buffer_cnt = 0
                                                        stream_width = inner_ug*inner_ue*inner_un*inner_uw
                                                        bus_idx=0
                                                        mlb_chain_len=1
                                                        outer_chain_len=1
                                                        
                                                        if ("PRELOAD" in projection["inner_projection"]):
                                                            mlb_chain_len=inner_ug*inner_ue*inner_un*inner_uw
                                                            w_buf_inst_idx = \
                                                                ugi*inner_ue*inner_un*inner_uw + \
                                                                uei*inner_un*inner_uw + \
                                                                urni*inner_uw + \
                                                                urwi
                                                            stream_width = 1
                                                        else:
                                                            bus_idx = ugi*inner_ue*inner_un*inner_uw + \
                                                                      uei*inner_un*inner_uw + \
                                                                      urni*inner_uw + \
                                                                      urwi
                                                            stream_width=inner_ug*inner_ue*inner_un*inner_uw
                                                        if ("PRELOAD" in projection["outer_projection"]):
                                                            w_buf_inst_idx = \
                                                                (ugo*outer_ue*outer_un*outer_uw + \
                                                                ueo*outer_un*outer_uw + \
                                                                urno*outer_uw + \
                                                                urwo)*mlb_chain_len + \
                                                                w_buf_inst_idx
                                                            outer_chain_len = outer_ug*outer_ue*outer_uw*outer_un
                                                        else:
                                                            stream_idx = ugo*outer_ue*outer_un*outer_uw + \
                                                                ueo*outer_un*outer_uw + \
                                                                urno*outer_uw + \
                                                                urwo
                                                            streams_per_buffer = math.floor(len(wbuf[0][0]) / stream_width)
                                                            buffer_cnt = math.floor(stream_idx / streams_per_buffer)
                                                            bus_idx = (stream_idx % streams_per_buffer)*stream_width + bus_idx
                                                        buffer_idx = (outer_chain_len*mlb_chain_len - w_buf_inst_idx - 1)
                                                        if (temp_un > 1):
                                                            buffer_idx += ugt*temp_un
                                                        
                                                            
                                                        w = wbuf[buffer_cnt][(buffer_idx + urnt) % wbuf_len][bus_idx]
                                                        print("WBuffer[" + str(buffer_cnt) + "]["+str(buffer_idx+urnt)+"][" + str(bus_idx) + "] = " + str(w))
                                                        if ((ubt - urw) >= 0) and \
                                                           ((ubt - urw) < ibuf_len):
                                                            i_stream_idx = (outer_ub*outer_un*ugo + \
                                                                            ubo*outer_un + \
                                                                            urno)
                                                            i_value_idx = i_stream_idx*get_proj_stream_count(projection["inner_projection"], 'I') + \
                                                                          (inner_ub*inner_un*ugi + \
                                                                           ubi*inner_un + \
                                                                           urni)
                                                            ibuf_idx = math.floor(i_value_idx / ivalues_per_buf)
                                                            iv_idx = i_value_idx % ivalues_per_buf
                                                            print("IBuffer[" + str(ibuf_idx) + "]["+str(ugt*temp_ub*temp_un+ubt*temp_un+urnt-urw)+"][" + str(iv_idx) + "] = " + str(ibuf[ibuf_idx][(ugt*temp_ub+ubt+urnt-urw)%ibuf_len][iv_idx]))
                                                        
                                                            correct_sum += (ibuf[ibuf_idx][(ugt*temp_ub*temp_un+ubt*temp_un + urnt - urw)%ibuf_len][iv_idx] * w)
                                    out_act_idx = ugo*outer_ub*outer_ue*inner_ug*inner_ub*inner_ue + \
                                                  ubo*outer_ue*inner_ug*inner_ub*inner_ue + \
                                                  ueo*inner_ug*inner_ub*inner_ue + \
                                                  ugi*inner_ub*inner_ue + \
                                                  ubi*inner_ue + \
                                                  uei
                                    obuf_idx = math.floor(out_act_idx/ostreams_per_buf)
                                    os_idx = out_act_idx % ostreams_per_buf
                                    obuf[obuf_idx][ugt*temp_ub+ubt][os_idx] = correct_sum%(2**projection["stream_info"]["I"])
                                    print("MLB:"+ str(mlb_inst) + "  and mac:" + str(mac_idx) + " -> " + str(obuf[obuf_idx][ugt][os_idx]))
    return obuf


def merge_bus(v,width):
    """ Given a list of values, combine them into a single value """
    sum = 0
    for i in range(len(v)):
        sum += v[i] * (2 ** (width * i))
    return sum

def print_heading(string, step = -1):
    """ Print something out on the terminal in green with underlines """
    to_print = '\n===> '
    if (step > -1):
        to_print += "Step " + str(step) + ": "
    to_print += string + "\n"
    to_print += ("=" * len(to_print))
    print(to_print)

def printi(level, string):
    """ Print something out with indents """
    print(('\t' * level) + str(string))
