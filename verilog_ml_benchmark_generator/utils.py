"""Utility functions"""
import math

def get_var_product(projection, var_array):
    product = 1
    for var in var_array:
        product *= projection[var]['value']
    return product

def get_mlb_count(projection):
    return get_var_product(projection, ['URN','URW','UB','UE','UG'])

def get_proj_stream_count(projection, dtype):
    if (dtype == 'W'):
        return get_var_product(projection, ['URW','URN','UE','UG'])
    elif (dtype == 'I'):
        return get_var_product(projection, ['URN','UB','UG'])
    elif (dtype == 'O'):
        return get_var_product(projection, ['UE','UB','UG'])

def get_activation_function_name(projection):
    return projection['activation_function']

def get_max_datatype_width(hw_spec, type):
    return max(map(lambda port: port['width'] if (port['type'] == type) else 0,
               hw_spec['ports']))

def get_sum_datatype_width(hw_spec, type):
    return sum(map(lambda port: port['width'] if (port['type'] == type) else 0,
               hw_spec['ports']))

def get_mlb_io_count(hw_spec, type):
    dt_width = sum(map(lambda port: port['width'] if (port['type'] == type) else 0,
               hw_spec['ports']))
    
def get_num_buffers_reqd(buffer_spec, stream_count, stream_width):
    streams_per_buffer = math.floor(get_max_datatype_width(buffer_spec, 'DATAOUT')/stream_width)
    assert get_max_datatype_width(buffer_spec, 'DATAOUT') > 0, "Buffer DATAOUT port width of zero"
    return math.ceil(stream_count/streams_per_buffer)

def get_mlb_count(projection):
    return get_var_product(projection, ['URN','URW','UB','UE','UG'])

def get_overall_idx(projection, idxs={'URW':1,'URN':1,'UB':1,'UE':1,'UG':1}):
    product = 1
    total = 0
    for item in idxs:
        total += product*idxs[item]
        product *= projection[item]['value']    
    return total



    
