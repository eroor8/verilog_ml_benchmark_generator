"""Utility functions"""
import math
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import module_helper_classes
from pymtl3 import connect, Wire, InPort, OutPort
input_order = [['URN','chans'], ['UB','batches'], ['UG','value'], ['URN','y'], ['UB','y']]

def print_warning(indent, string):
    printi(indent, "WARNING: " + string, "ORANGE")

def print_mapping(mapping, indent):
    """ Print a layer nicely
    """
    i = {"RX": mapping["RXI"],
         "RY": mapping["RYI"],
         "E": mapping["EI"],
         "C": mapping["CI"],
         "B": mapping["BI"],
         "PX": mapping["PXI"],
         "PY": mapping["PYI"]
    }
    o = {"RX": mapping["RXO"],
         "RY": mapping["RYO"],
         "E": mapping["EO"],
         "C": mapping["CO"],
         "B": mapping["BO"],
         "PX": mapping["PXO"],
         "PY": mapping["PYO"]
    }
    t = {"RX": mapping["RXT"],
         "RY": mapping["RYT"],
         "E": mapping["ET"],
         "C": mapping["CT"],
         "B": mapping["BT"],
         "PX": mapping["PXT"],
         "PY": mapping["PYT"]
    }
    
    return ("\n" + "\t"*indent + "Intra-PE unrolling factors: " + str(i) +
            "\n" + "\t"*indent + "Iner-PE unrolling factors: " + str(o) +
            "\n" + "\t"*indent + "Temporal tiling factors: " + str(t)) 

def get_var_product(projection, var_array):
    """ Multiply a subset of projection factors
    """
    product = 1
    for var in var_array:
        #assert var in projection, "Key " + var + \
        #       " not found in projection definition."
        product *= int(projection.get(var,{}).get('value',1))
    return product

def get_var_product_new(projection, var_array, defaults=[]):
    """ Multiply a subset of projection factors
    """
    product = 1
    for var in var_array:
        if var[0] in projection:
            assert var[0] in projection
            if var[0] in projection:
                if var[1] in projection[var[0]]:
                    product *= int(projection[var[0]][var[1]])
                elif var[1] in defaults:
                    product *= int(projection[var[0]]['value'])
            
            #assert var[0] in projection, "Key " + var + \
            #   " not found in projection definition."
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


def get_num_buffers_reqd(buffer_spec, stream_count, stream_width, max_buf_width=10000000):
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
    buf_width = get_sum_datatype_width(buffer_spec, "DATA", ["in"])
    buf_width = min(max_buf_width, buf_width)
    
    streams_per_buffer = math.floor(buf_width / stream_width)
    buf_per_stream = math.ceil( stream_width / buf_width)
    if (streams_per_buffer > 0):
        buf_count = math.ceil(stream_count / streams_per_buffer)
    else:
        buf_count = buf_per_stream * stream_count
    assert get_sum_datatype_width(buffer_spec, 'DATA', ["in"]) > 0, \
        "Buffer DATAOUT port width of zero"
    assert(buf_count > 0)
    return buf_count


def get_overall_idx_new(projection, idxs, order=['URW', 'URN', 'UE', 'UB', 'UG'], default=[]):
    product = 1
    total = 0
    #for item in idxs:
    #    assert item in order
    for item in order:
        if item[0] in idxs and item[1] in idxs[item[0]]:
            curr_val = idxs[item[0]][item[1]]     
            #assert item[0] in projection
            if item[0] in projection and item[1] not in projection[item[0]]:
                if item[1] in default:
                    projection_val = projection[item[0]]['value']
                else:
                    projection_val = 1
            else:  
                projection_val = projection.get(item[0],{}).get(item[1],1)
            #assert curr_val < projection_val
            assert curr_val >= 0
            total += product*curr_val
            product *= projection_val
    return total


def get_overall_idx(projection, idxs, order=['URW', 'URN', 'UE', 'UB', 'UG']):
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
        assert item in order
    for item in order:
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

def flatten_array(input_array, data_width):
    return [sum((lambda i: inner[i] * (2 ** (i * data_width)))(i)
                for i in range(len(inner)))
            for outer in input_array
            for inner in outer]

def mux_ports_by_name(s, srcs, name1, inst2, name2, factor1=1, factor2=1, insel={}, sim=False, idx=''):
    """ Connect ports named ``name1``_<#*``factor1``> on ``src``
        to ports named ``name2``_<#*``factor2``> on

    :param src: Module instance with output ports to be connected
    :type src: Component class
    :param inst2: Module instance with input ports to be connected
    :type inst2: Component class
    :param name1: Prefix of names of output ports of ``src``
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
    common_ports = []
    for src in srcs:
        for port in src.get_output_value_ports():
            port1name = port._dsl.my_name
            foundname1 = re.search("^" + name1+r"$", port1name)
            if foundname1:
                try:
                    if str(int(foundname1.group(1))*factor1) in match_dict:
                        match_dict[str(int(foundname1.group(1))*factor1)] += [port]
                    else:
                        match_dict[str(int(foundname1.group(1))*factor1)] = [port]
                except:
                    common_ports += [port]
    assert (len(match_dict) > 0) or (len(common_ports) == len(srcs)), \
        "Should have found outputs with name " + \
        name1 + " in " + str(srcs[0].get_output_value_ports()) + " and " + str(common_ports)
    for port in inst2.get_input_value_ports():
        port2name = port._dsl.my_name
        foundname2 = re.search("^" + name2+r"$", port2name)
        
        if foundname2:
            try:
                name_to_get = str(int(foundname2.group(1))*factor2)
            except:
                name_to_get = -1
            inports = match_dict.get(name_to_get, common_ports)
            assert(len(inports) > 0)
            if (len(inports) > 1):
                muxn_inst = module_helper_classes.MUXN(port._dsl.Type.nbits, len(inports), sim=sim)
                setattr(s, port2name+"mux"+idx, muxn_inst)
                if (len(inports) < 2):
                    muxn_inst.sel //= 0
                else:
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
                "Should have found output with name equivalent to " + port2name + " in " + \
                str(match_dict)
            name_to_get = str(int(foundname2.group(1))*factor2)
            connectport = match_dict.get(name_to_get, common)
            connectport //= port
            connected_ins += [port]
            connected_outs += [connectport]
            if (str(int(foundname2.group(1))*factor2) in match_dict):
                del match_dict[str(int(foundname2.group(1))*factor2)]
    assert len(match_dict) == 0, "Missing matches for ports " + \
        str(match_dict) + " in list " + str(inst2.get_input_value_ports())
    return connected_ins + connected_outs

def connect_inst_ports_by_name(parent, namep, inst, namei, \
                               factor1=1, factor2=1, parent_in=1):
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
            #assert(parentport)
            if not (parentport):
                if (parent_in):
                    parentport = AddInPort(parent, port._dsl.Type.nbits, namep + "_" + \
                                     foundname1.group(1))
                else:
                    parentport = AddOutPort(parent, port._dsl.Type.nbits, namep + "_" + \
                                     foundname1.group(1))
            if (port._dsl.Type.nbits > 1):
                connect(parentport[0:port._dsl.Type.nbits], port)
            else:
                connect(parentport, port)
            connected_ins += [port]
    assert foundport, "Port " + namei + " not found in " + str(instports)
    return connected_ins

def mux_inst_ports_by_name(inst2, name2, srcs, name1, factor1=1, factor2=1, insel={}, idx='', sim=True):
    """ Connect ports named ``name1``_<#*``factor1``> on ``src``...
    """
    match_dict = {}
    connected_ins = []
    connected_outs = []
    for src in srcs:
        for port in src.get_output_value_ports():
            port1name = port._dsl.my_name
            foundname1 = re.search("^" + name1+r"$", port1name)
            if foundname1:
                try:
                    if str(int(foundname1.group(1))*factor1) in match_dict:
                        match_dict[str(int(foundname1.group(1))*factor1)] += [port]
                    else:
                        match_dict[str(int(foundname1.group(1))*factor1)] = [port]
                except:
                    if "c" in match_dict:
                        match_dict["c"] += [port]
                    else:
                        match_dict["c"] = [port]
    assert (len(match_dict) > 0), \
        "Should have found outputs with name " + \
        name1 + " in " + str(srcs[0].get_output_value_ports())

    for port in match_dict:
        parentname = name2+"_"+port
        parentport = None
        if parentname in inst2.__dict__.keys():
            parentport = getattr(inst2, name2 + "_" + \
                                     foundname1.group(1) )
        elif name2 in inst2.__dict__.keys():
            parentport = getattr(inst2, name2)
        assert(parentport)
        inports = match_dict[port]
        muxn_inst = module_helper_classes.MUXN(parentport._dsl.Type.nbits, len(inports), sim)
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

def get_max_input_bus_width(buf_width, projection, data_type, inner_width=-1):
    #buf_width = get_sum_datatype_width(buf_spec, "DATA", ["in"])
    max_vals_per_buf = get_var_product_new(projection.get("inner_projection",{}), [['UB','batches'], ['UG','value'], ['URN','chans']], defaults=['chans','batches'])
    if inner_width < 0:
        inner_width = projection["stream_info"][data_type]
    if ((projection.get('inner_projection',{}).get('URN',{}).get('y',1) > 1) and data_type ==  "I"):
        buf_width = min(inner_width*max_vals_per_buf, buf_width)
    #assert buf_width > 1
    return buf_width

def get_iw_buffer_dimensions(buf_spec, projection, data_type):
    """ Find the required number and sizes of buffers """
    
    stream_count = get_proj_stream_count(projection["outer_projection"],
                                         data_type)
    values_per_stream = get_proj_stream_count(projection["inner_projection"],
                                              data_type)
    
    stream_bitwidth = values_per_stream * projection["stream_info"][data_type]
    buf_width = get_max_input_bus_width(get_sum_datatype_width(buf_spec, "DATA", ["in"]),
                                        projection, data_type)
    streams_per_buf = math.floor(
         buf_width / stream_bitwidth)
    buf_per_stream = math.ceil( stream_bitwidth / buf_width)
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
    activation_width = projection["stream_info"]["I"]
    values_per_buf = math.floor(get_sum_datatype_width(buf_spec, "DATA", ["in"]) / activation_width)
    buf_count = math.ceil(stream_count / values_per_buf)
    buf_len = 2 ** get_sum_datatype_width(buf_spec, "ADDRESS", ["in"])
    return (values_per_buf, buf_len, buf_count)

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

    

def compute_layer(inputs, weights, layer):
    urx = layer.get("filter_x",1)
    ury = layer.get("filter_y",1)
    urc = layer.get("in_chans",1)
    ue = layer.get("out_chans",1)
    ub = layer.get("batches",1)
    ubx = layer.get("image_x",1) 
    uby = layer.get("image_y",1) 
    ug = layer.get("group",1) 
    stridex = layer.get("stridex",1)
    stridey = layer.get("stridey",1)
    dilx = layer.get("dilx",1)
    dily = layer.get("dily",1)
    
    outputs = [[[[[0 for k in range(int(ubx/stridex))]  # x
                 for i in range(int(uby/stridey))]      # y    
                 for j in range(ue)]       # chans
                 for l in range(ub)]       # batch
                 for t in range(ug)]       # group
    for ugi in range(ug): # group
        for ubi in range(ub): # batch
            for uei in range(ue): # out chan
                for urci in range(urc): # in chan
                    for ubxi in range(urx-1, ubx, stridex): # px x
                        for ubyi in range(ury-1, uby, stridey): # px y
                            for urxi in range(urx): # filter x
                                for uryi in range(ury): # filter y
                                    if ((urxi % dilx == 0) and (uryi % dily == 0)):
                                        inact = inputs[ugi][ubi][urci][ubyi+1-ury+uryi][ubxi-urxi]
                                        weight = weights[ugi][uei][urci][uryi][urxi]
                                        outputs[ugi][ubi][uei][int((ubyi-ury+1)/stridey)][int((ubxi-urx+1)/stridex)] += inact*weight
    return outputs

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
    temp_ue = projection.get("temporal_projection",{}).get("UE",{}).get("value", 1)
    mlb_count = get_mlb_count(projection["outer_projection"])
    mac_count = get_mlb_count(projection["inner_projection"])
    for ugt in range(temp_ug):
        for ugo in range(outer_ug): 
            for ugi in range(inner_ug):
                for ubo in range(outer_ub): 
                    for ubi in range(inner_ub):
                        for ubt in range(outer_uw*inner_uw-1,temp_ub):
                            for ueo in range(outer_ue):
                                for uei in range(inner_ue):
                                    for uet in range(temp_ue):
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
                                                            buffer_idx += ugt*temp_ue*temp_un + uet*temp_un
                                                            
                                                                
                                                            w = wbuf[buffer_cnt][(buffer_idx + urnt) % wbuf_len][bus_idx]
                                                            i_stream_idx = get_overall_idx_new(projection["outer_projection"],
                                                                                {'URN': {'chans':urno},
                                                                                 'UB': {'batches':ubo},
                                                                                 'UG': {'value': ugo}},
                                                                                 order=input_order, default=['batches','chans'])
                                                            i_value_idx = i_stream_idx*get_proj_stream_count(projection["inner_projection"], 'I') + \
                                                                                get_overall_idx_new(projection["inner_projection"],
                                                                                {'URN': {'chans':urni},
                                                                                 'UB': {'batches':ubi},
                                                                                 'UG': {'value': ugi}},
                                                                                 order=input_order, default=['batches','chans'])
                                                            ibuf_idx = math.floor(i_value_idx / ivalues_per_buf)
                                                            iv_idx = i_value_idx % ivalues_per_buf
                                                            
                                                            correct_sum += (ibuf[ibuf_idx][(ugt*temp_ub*temp_un+ubt*temp_un + urnt - urw)%ibuf_len][iv_idx] * w)
                                        out_act_idx = ugo*outer_ub*outer_ue*inner_ug*inner_ub*inner_ue + \
                                                      ubo*outer_ue*inner_ug*inner_ub*inner_ue + \
                                                      ueo*inner_ug*inner_ub*inner_ue + \
                                                      ugi*inner_ub*inner_ue + \
                                                      ubi*inner_ue + \
                                                      uei
                                        obuf_idx = math.floor(out_act_idx/ostreams_per_buf)
                                        os_idx = out_act_idx % ostreams_per_buf
                                        obuf[obuf_idx][ugt*temp_ub*temp_ue+uet*temp_ub+ubt-outer_uw*inner_uw+1][os_idx] = correct_sum%(2**projection["stream_info"]["I"])
    return obuf

def get_expected_outputs_old(obuf, ostreams_per_buf, wbuf, ibuf, ivalues_per_buf, projection):
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
    temp_ue = projection.get("temporal_projection",{}).get("UE",{}).get("value", 1)
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
                                    for uet in range(temp_ue):
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
                                                            buffer_idx += ugt*temp_ue*temp_un + uet*temp_un
                                                            
                                                                
                                                            w = wbuf[buffer_cnt][(buffer_idx + urnt) % wbuf_len][bus_idx]
                                                            if ((ubt - urw) >= 0) and ((ubt - urw) < ibuf_len):
                                                                i_stream_idx = (outer_ub*outer_un*ugo + \
                                                                                ubo*outer_un + \
                                                                                urno)
                                                                i_value_idx = i_stream_idx*get_proj_stream_count(projection["inner_projection"], 'I') + \
                                                                              (inner_ub*inner_un*ugi + \
                                                                               ubi*inner_un + \
                                                                               urni)
                                                                ibuf_idx = math.floor(i_value_idx / ivalues_per_buf)
                                                                iv_idx = i_value_idx % ivalues_per_buf
                                                                
                                                                correct_sum += (ibuf[ibuf_idx][(ugt*temp_ub*temp_un+ubt*temp_un + urnt - urw)%ibuf_len][iv_idx] * w)
                                        out_act_idx = ugo*outer_ub*outer_ue*inner_ug*inner_ub*inner_ue + \
                                                      ubo*outer_ue*inner_ug*inner_ub*inner_ue + \
                                                      ueo*inner_ug*inner_ub*inner_ue + \
                                                      ugi*inner_ub*inner_ue + \
                                                      ubi*inner_ue + \
                                                      uei
                                        obuf_idx = math.floor(out_act_idx/ostreams_per_buf)
                                        os_idx = out_act_idx % ostreams_per_buf
                                        obuf[obuf_idx][ugt*temp_ub*temp_ue+uet*temp_ub+ubt][os_idx] = correct_sum%(2**projection["stream_info"]["I"])
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
    printi(1, to_print, "BLUE")

    
def printi(level, string, colour="None"):
    """ Print something out with indents """
    col_code = ''
    if (colour) == "ORANGE":
        col_code = '\033[31m'
    elif (colour) == 'GREEN':
        col_code = '\033[32m'
    elif (colour) == 'RED':
        col_code = '\033[31m'
    elif (colour) == 'BLUE':
        col_code = '\033[34m'
    print(('\t' * level) + col_code + str(string)  + '\033[0m')

    
def map_buffer_idx_to_y_idx(proj_yaml, ab_yaml=None, ibuf_count=0, ivalues_per_buf=0):
    if (ab_yaml):
        ivalues_per_buf, ibuf_len, ibuf_count = get_iw_buffer_dimensions(
             ab_yaml, proj_yaml, 'I')
        
    output_map = [0 for buf in range(ibuf_count)]
    inner_ug = proj_yaml.get("inner_projection",{}).get("UG",{}).get("value",1)
    outer_ug = proj_yaml.get("outer_projection",{}).get("UG",{}).get("value",1)
    inner_ub = proj_yaml.get("inner_projection",{}).get("UB",{}).get("value",1)
    inner_ubb = proj_yaml.get("inner_projection",{}).get("UB",{}).get("batches",inner_ub)
    inner_ubx = proj_yaml.get("inner_projection",{}).get("UB",{}).get("x",1)
    inner_uby = proj_yaml.get("inner_projection",{}).get("UB",{}).get("y",1)
    outer_ub = proj_yaml.get("outer_projection",{}).get("UB",{}).get("value",1)
    outer_ubb = proj_yaml.get("outer_projection",{}).get("UB",{}).get("batches", outer_ub)
    outer_ubx = proj_yaml.get("outer_projection",{}).get("UB",{}).get("x",1)
    outer_uby = proj_yaml.get("outer_projection",{}).get("UB",{}).get("y",1)
    inner_un = proj_yaml.get("inner_projection",{}).get("URN",{}).get("value",1)
    inner_unc = proj_yaml.get("inner_projection",{}).get("URN",{}).get("chans",inner_un)
    #
    inner_unx = proj_yaml.get("inner_projection",{}).get("URN",{}).get("x",1)
    inner_uny = proj_yaml.get("inner_projection",{}).get("URN",{}).get("y",1)
    #
    outer_un = proj_yaml.get("outer_projection",{}).get("URN",{}).get("value",1)
    outer_unc = proj_yaml.get("outer_projection",{}).get("URN",{}).get("chans",outer_un)
    outer_unx = proj_yaml.get("outer_projection",{}).get("URN",{}).get("x",1)
    outer_uny = proj_yaml.get("outer_projection",{}).get("URN",{}).get("y",1)

    for ugo in range(outer_ug): 
        for ugi in range(inner_ug):
            for ubox in range(outer_ubx):
                for ubix in range(inner_ubx):
                    for uboy in range(outer_uby):
                        for ubiy in range(inner_uby):
                            for ubob in range(outer_ubb):
                                for ubib in range(inner_ubb):
                                    for urnoc in range(outer_unc):
                                        for urnic in range(inner_unc):
                                            for urnoy in range(outer_uny):
                                                for urniy in range(inner_uny):
                                                    #print("UGO:" + str(ugo) + " UGI" + str(ugi) +
                                                    #      " UBOX:" + str(ubox) + " UBIX" + str(ubix) +
                                                    #      " UBOY:" + str(uboy) + " UBIY" + str(ubiy) +
                                                     #     " UBOB:" + str(ubob) + " UBIB" + str(ubib))
                                                    #print(proj_yaml["outer_projection"])
                                                    i_stream_idx = get_overall_idx_new(proj_yaml["outer_projection"],
                                                                                       {'URN': {'y':urnoy, 'chans':urnoc},
                                                                                        'UB': {'y':uboy, 'x':ubox, 'batches':ubob},
                                                                                        'UG': {'value': ugo}},
                                                                  order=input_order, default=['batches','chans'])
                                                    i_value_idx = i_stream_idx*get_proj_stream_count(proj_yaml["inner_projection"], 'I') + \
                                                                  get_overall_idx_new(proj_yaml["inner_projection"],
                                                                                       {'URN': {'y':urniy, 'chans':urnic},
                                                                                        'UB': {'y':ubiy, 'x':ubix, 'batches':ubib},
                                                                                        'UG': {'value': ugi}},
                                                         order=input_order, default=['batches','chans'])
                                                   # print(i_value_idx)
                                                    #print(ivalues_per_buf)
                                                    buf_idx = math.floor(i_value_idx / ivalues_per_buf)                
                                                    inner_y = get_overall_idx_new(proj_yaml["inner_projection"],
                                                                                       {'URN': {'y':urniy}, 'UB': {'y':ubiy}},
                                                                  order=[['URN','y'], ['UB','y']], default=['batches','chans'])
                                                    outer_y = get_overall_idx_new(proj_yaml["outer_projection"],
                                                                                       {'URN': {'y':urnoy}, 'UB': {'y':uboy}},
                                                                  order=[['URN','y'], ['UB','y']], default=['batches','chans'])
                                                    output_map[buf_idx] = outer_y*inner_uby*inner_uny + inner_y
                                                    #print("MAP " + str(buf_idx) + ": " + str(outer_y) + ", " + str(inner_y))
    return output_map
