import copy
import pytest
import os
import yaml
import sys
import random
import jsonschema
import subprocess
from jsonschema import validate
from verilog_ml_benchmark_generator import utils
from verilog_ml_benchmark_generator import generate_modules

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_helpers import *

VTR_FLOW_PATH = os.getenv('VTR_FLOW_PATH')
NUMX = 1976
NUMI = 988
HEIGHT = 177
WIDTH = 149

filesets = [# EMIF, input, MLB interface same width
            # Preload weights from wide buffer
            # Inner: URW3, URN2, UE1, UB2, UG2
            # Outer: URW1, URN2, UE1, UB2, UG2
              ("mlb_spec_0.yaml","input_spec_0.yaml", "weight_spec_0.yaml",
               "emif_spec_0.yaml", "projection_spec_0.yaml", True),
            # EMIF wider than input, weight buffer
              ("mlb_spec_0.yaml","input_spec_0.yaml", "weight_spec_0.yaml",
               "emif_spec_1.yaml", "projection_spec_0.yaml", True),
            # Input 2x as wide as MLB interface
              ("mlb_spec_0.yaml","input_spec_1.yaml", "weight_spec_0.yaml",
               "emif_spec_1.yaml", "projection_spec_0.yaml", True),
            # Narrower weight buffer
              ("mlb_spec_0.yaml","input_spec_0.yaml", "weight_spec_1.yaml",
              "emif_spec_1.yaml", "projection_spec_0.yaml", True),
            # Inner: URW6, URN1, UE2, UB1, UG1
            # Outer: URW1, URN1, UE1, UB2, UG1
               ("mlb_spec_1.yaml","input_spec_0.yaml", "weight_spec_0.yaml",
                "emif_spec_0.yaml", "projection_spec_1.yaml", True),
            #
            # Narrower I, W, 
            # Inner: URW1, URN2, UE2, UB2, UG1
            # Outer: URW1, URN1, UE3, UB1,UG1
             ("mlb_spec_3.yaml","input_spec_1.yaml", "weight_spec_3.yaml",
                "emif_spec_1.yaml", "projection_spec_3.yaml", True),
            # Narrower I, W, 
            # Outer: URW1, URN2, UE2, UB2, UG1
            # Inner: URW1, URN1, UE3, UB1,UG1
               ("mlb_spec_3.yaml","input_spec_1.yaml", "weight_spec_3.yaml",
                 "emif_spec_1.yaml", "projection_spec_4.yaml", True),
            # Inner: URW6, URN1, UE1, UB2, UG1
            # Outer: URW2, URN2, UE2, UB1, UG1
               ("mlb_spec_3.yaml","input_spec_1.yaml", "weight_spec_3.yaml",
                 "emif_spec_1.yaml", "projection_spec_5.yaml", True),
            # Inner: All2
            # Outer: All2
                ("mlb_spec_3.yaml","input_spec_1.yaml", "weight_spec_3.yaml",
                  "emif_spec_1.yaml", "projection_spec_5.yaml", True),
           ]  # bad: URW2, URN2

@pytest.mark.parametrize(
    "mlb_file,ab_file,wb_file,emif_file,proj_files,ws", filesets
)
@pytest.mark.full_simulations
@pytest.mark.skip
def test_simulate_multiple_layers(
        mlb_file, ab_file, wb_file, emif_file, proj_files, ws, v=True):
    
    # Make sure that output gets through odin.
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            mlb_file)
    ab_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ab_file)
    wb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            wb_file)
    emif_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            emif_file)
    proj_specs = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               proj_files)
    with open(mlb_spec) as stream:
        mlb_yaml = yaml.safe_load(stream)
    with open(ab_spec) as stream:
        ab_yaml = yaml.safe_load(stream)
    with open(wb_spec) as stream:
        wb_yaml = yaml.safe_load(stream)
    with open(proj_specs) as stream:
        proj_yamls = yaml.safe_load(stream)
    with open(emif_spec) as stream:
        emif_yaml = yaml.safe_load(stream)

    wbufs_flat=[]
    wbufs=[]
    obufs_flat=[]
    waddrs = []
    iaddrs = []
    oaddrs = []
    weights = []
    layers = []
    #proj_yamls = [proj_yamls[1], proj_yamls[0]]
    for py_i in range(len(proj_yamls)):
        proj_yaml=proj_yamls[py_i]
        # Calculate buffer dimensions info
        wvalues_per_buf, wbuf_len, wbuf_count = utils.get_iw_buffer_dimensions(
            wb_yaml, proj_yaml, 'W')
        ivalues_per_buf, ibuf_len, ibuf_count = utils.get_iw_buffer_dimensions(
            ab_yaml, proj_yaml, 'I')
        ovalues_per_buf, obuf_len, obuf_count = utils.get_obuffer_dimensions(
            ab_yaml, proj_yaml)  
        
        inner_uw = proj_yaml["inner_projection"]["RX"]
        inner_uwx = proj_yaml["inner_projection"]["RX"]
        inner_uwy = 1
        outer_uw = proj_yaml["outer_projection"]["RX"]
        outer_uwx = proj_yaml["outer_projection"]["RX"]
        outer_uwy = 1
        assert((inner_uwx == 1) | (inner_uwy == 1)) # Can't window in both directions
        assert((outer_uwx == 1) | (outer_uwy == 1))
        assert((inner_uwx == 1) | (outer_uwy == 1))
        assert((outer_uwx == 1) | (inner_uwy == 1))
    
        inner_unc = proj_yaml["inner_projection"]["C"]
        inner_unx = 1
        inner_uny = proj_yaml["inner_projection"]["RY"]
        inner_un = inner_unc * inner_uny
        outer_unc = proj_yaml["outer_projection"]["C"]
        outer_unx = 1
        outer_uny = proj_yaml["outer_projection"]["RY"]
        outer_un = outer_unc * outer_uny
        temp_unc = proj_yaml.get("temporal_projection",{}).get("C",1)
        temp_unx = 1
        temp_uny = proj_yaml.get("temporal_projection",{}).get("RY",1)
        temp_un = temp_unc * temp_uny
        assert(((inner_uwx == 1) & (outer_uwx == 1)) | ((inner_unx == 1) & (outer_unx == 1)))
        assert(((inner_uwy == 1) & (outer_uwy == 1)) | ((inner_uny == 1) & (outer_uny == 1)))
        
        inner_ue = proj_yaml["inner_projection"]["E"]
        outer_ue = proj_yaml["outer_projection"]["E"]
        temp_ue = proj_yaml.get("temporal_projection",{}).get("E",1)
        
        inner_ubb = proj_yaml["inner_projection"]["B"]
        inner_ubx = proj_yaml["inner_projection"]["PX"]
        inner_uby = proj_yaml["inner_projection"]["PY"]
        inner_ub = inner_ubb * inner_ubx * inner_uby
        outer_ubb = proj_yaml["outer_projection"]["B"]
        outer_ubx = proj_yaml["outer_projection"]["PX"]
        outer_uby = proj_yaml["outer_projection"]["PY"]
        outer_ub = outer_ubb * outer_ubx * outer_uby
        temp_ubb = proj_yaml.get("temporal_projection",{}).get("B",1)
        temp_ubx = proj_yaml.get("temporal_projection",{}).get("PX", obuf_len)
        temp_uby = proj_yaml.get("temporal_projection",{}).get("PY", 1)
        temp_ub = temp_ubb * temp_ubx * temp_uby
        assert(((inner_uwx == 1) & (outer_uwx == 1)) | ((inner_ubx == 1) & (outer_ubx == 1)))
        assert(((inner_uwy == 1) & (outer_uwy == 1)) | ((inner_uby == 1) & (outer_uby == 1)))
        
        inner_ug = proj_yaml["inner_projection"]["G"]
        outer_ug = proj_yaml["outer_projection"]["G"]
        temp_ug = proj_yaml.get("temporal_projection",{}).get("G",{})
        stridex = proj_yaml.get("stride",{}).get("x",1)
        stridey = proj_yaml.get("stride",{}).get("y",1)
        
        # Figure out the layer dimensions based on the projection...
        layers += [{"group": inner_ug*outer_ug*temp_ug,
                 "batches": inner_ubb*outer_ubb*temp_ubb,
                 "out_chans": inner_ue*outer_ue*temp_ue,
                 "in_chans": inner_unc*outer_unc*temp_unc,
                 "image_x": inner_ubx*outer_ubx*temp_ubx,
                 "image_y": inner_uby*outer_uby*temp_uby,
                 "filter_x": inner_uwx*outer_uwx*inner_unx*outer_unx*temp_unx,
                 "filter_y": inner_uwy*outer_uwy*inner_uny*outer_uny*temp_uny,
                 "stridex": stridex,
                 "stridey": stridey
              }]
        print(layers)
        print("Adding to weights")
        print(layers[-1]["group"])
        # Create random input data arrays to load into EMIF
        weights_i = [[[[[random.randint(1,4) #(2**proj_yaml["data_widths"]["W"])-1)
                       for k in range(layers[-1]["filter_x"])]    # x
                       for i in range(layers[-1]["filter_y"])]    # y    
                       for j in range(layers[-1]["in_chans"])]    # ichans
                       for l in range(layers[-1]["out_chans"])]   # ochans
                       for t in range(layers[-1]["group"])]       # group
        print(weights_i)
        weights += [weights_i]
        
        # Move the weights and inputs into the EMIF in the expected order
        wbuf = [[[0 for k in range(wvalues_per_buf)]    # inner urw * urn * ue * ug
                for i in range(wbuf_len)]               # temp urn * ue * ug
                for j in range(wbuf_count)]            # outer urw * urn * ue * ug
        wbufs += [wbuf]
        for ugt in range(temp_ug):
            for ugo in range(outer_ug): 
                for ugi in range(inner_ug):
                    for ueo in range(outer_ue):
                        for uei in range(inner_ue):
                            for uet in range(temp_ue):
                                for urnoc in range(outer_unc):
                                    for urnic in range(inner_unc):
                                        for urntc in range(temp_unc):
                                            for urnox in range(outer_unx):
                                                for urnix in range(inner_unx):
                                                    for urntx in range(temp_unx):
                                                        for urnoy in range(outer_uny):
                                                            for urniy in range(inner_uny):
                                                                for urnty in range(temp_uny):
                                                                    for urwox in range(outer_uwx):
                                                                        for urwix in range(inner_uwx):
                                                                            for urwoy in range(outer_uwy):
                                                                                for urwiy in range(inner_uwy):
                                                                                    urno = urnox*outer_uny + urnoy + urnoc*outer_unx*outer_uny
                                                                                    urni = urnix*inner_uny + urniy + urnic*inner_unx*inner_uny 
                                                                                    urnt = urntx*temp_uny  + urnty + urntc*temp_unx*temp_uny
                                                                                    w = weights[py_i][ugt*outer_ug*inner_ug+ugo*inner_ug+ugi]\
                                                                                               [uet*outer_ue*inner_ue+ueo*inner_ue+uei]\
                                                                                               [urntc*outer_unc*inner_unc+urnoc*inner_unc+urnic]\
                                                                                               [urwoy*inner_uwy+urwiy]\
                                                                                               [urwox*inner_uwx+urwix]
                                                                                    urwo = max(urwox,urwoy)
                                                                                    urwi = max(urwix,urwiy)
                                                                                    w_buf_inst_idx = 0
                                                                                    buffer_cnt = 0
                                                                                    stream_width = inner_ug*inner_ue*inner_un*inner_uw
                                                                                    bus_idx=0
                                                                                    mlb_chain_len=1
                                                                                    outer_chain_len=1
                                                                                    if ("PRELOAD" in proj_yaml["inner_projection"]):
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
                                                                                    if ("PRELOAD" in proj_yaml["outer_projection"]):
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
                                                                                    wbuf[buffer_cnt][(buffer_idx + urnt) % wbuf_len][bus_idx] = w
        waddrs += [len(wbufs_flat)]
        wbufs_flat += [sum((lambda i: inner[i] * \
                          (2**(i*proj_yaml["data_widths"]["W"])))(i) \
                         for i in range(len(inner))) \
                             for outer in wbuf for inner in outer]
        if (py_i == 0):
            inputs = [[[[[random.randint(1,4)
                               for k in range(layers[0]["image_x"])]     # x
                               for i in range(layers[0]["image_y"])]     # y    
                               for j in range(layers[0]["in_chans"])]    # chans
                               for l in range(layers[0]["batches"])]     # batch
                               for t in range(layers[0]["group"])]       # group
            ibuf =  [[[0 for k in range(ivalues_per_buf)]         # values per word
                         for i in range(ibuf_len)]                   # words per buffer
                         for j in range (ibuf_count)]                # buffers
            for ugt in range(temp_ug):
                for ugo in range(outer_ug): 
                    for ugi in range(inner_ug):
                        for ubox in range(outer_ubx):
                            for ubix in range(inner_ubx):
                                for ubtx in range(temp_ubx):
                                    for uboy in range(outer_uby):
                                        for ubiy in range(inner_uby):
                                            for ubty in range(temp_uby):
                                                for ubob in range(outer_ubb):
                                                    for ubib in range(inner_ubb):
                                                        for ubtb in range(temp_ubb):
                                                            for urnoc in range(outer_unc):
                                                                for urnic in range(inner_unc):
                                                                    for urntc in range(temp_unc):
                                                                        i = inputs[ugt*outer_ug*inner_ug+ugo*inner_ug+ugi]\
                                                                                  [ubtb*outer_ubb*inner_ubb+ubob*inner_ubb+ubib]\
                                                                          [urntc*outer_unc*inner_unc+urnoc*inner_unc+urnic]\
                                                                          [ubty*outer_uby*inner_uby+uboy*inner_uby+ubiy]\
                                                                          [ubtx*outer_ubx*inner_ubx+ubox*inner_ubx+ubix]
                                                                        ubo = ubob*outer_ubx*outer_uby + ubox*outer_uby + uboy
                                                                        ubi = ubib*inner_ubx*inner_uby + ubix*inner_uby + ubiy
                                                                        ubt = ubtb*temp_ubx*temp_uby + ubty*temp_ubx + ubtx
                                                                        urno = urnoc*outer_unx*outer_uny
                                                                        urni = urnic*inner_unx*inner_uny 
                                                                        urnt = urntc*temp_unx*temp_uny
                                                                        i_stream_idx = (outer_ub*outer_un*ugo + \
                                                                                        ubo*outer_un + \
                                                                                        urno)
                                                                        i_value_idx = i_stream_idx*utils.get_proj_stream_count(proj_yaml["inner_projection"], 'I') + \
                                                                                  (inner_ub*inner_un*ugi + \
                                                                                   ubi*inner_un + \
                                                                                   urni)
                                                                        ibuf_idx = math.floor(i_value_idx / ivalues_per_buf)
                                                                        iv_idx = i_value_idx % ivalues_per_buf
                                                                        ibuf[ibuf_idx][(ugt*temp_ub*temp_un+ubt*temp_un + urnt)%ibuf_len][iv_idx] = i
            iaddrs = [0 for i in range(len(proj_yamls))] 
            ibuf_flat = [sum((lambda i: inner[i] * \
                        (2**(i*proj_yaml["data_widths"]["I"])))(i) \
                             for i in range(len(inner))) \
                                 for outer in ibuf for inner in outer]

    print(weights)
    layer_outputs_i = inputs
    for py_i in range(len(proj_yamls)):
        print("Compute")
        print(py_i)
        print(layer_outputs_i)
        print(weights[py_i])
        print(layers[py_i])
        layer_outputs_i = utils.compute_layer(layer_outputs_i, weights[py_i], layers[py_i],
                                              output_width=proj_yaml["data_widths"]["I"],
                                              activation_function=proj_yaml["activation_function"])

    layer_output = [[[[[layer_outputs_i[t][l][j][i][k]%(2**proj_yaml["data_widths"]["I"])
                             for k in range(len(layer_outputs_i[t][l][j][i]))]  # x
                             for i in range(len(layer_outputs_i[t][l][j]))]      # y    
                             for j in range(len(layer_outputs_i[t][l]))]       # chans
                             for l in range(len(layer_outputs_i[t]))]       # batch
                             for t in range(len(layer_outputs_i))]       # group o%(2**proj_yaml["data_widths"]["I"])                                                                    
    obuf =  [[[0 for k in range(ivalues_per_buf)] 
             for i in range(ibuf_len)]                  
             for j in range (ibuf_count)]  
    obufs_flat += [sum((lambda i: inner[i] * \
                (2**(i*proj_yaml["data_widths"]["I"])))(i) \
                     for i in range(len(inner))) \
                          for outer in obuf for inner in outer]            
    oaddrs = [0 for i in range(len(proj_yamls))]

    for i in range(len(iaddrs)):
        iaddrs[i] += len(wbufs_flat)
    for i in range(len(oaddrs)):
        oaddrs[i] += len(ibuf_flat) + len(wbufs_flat)
    emif_data = wbufs_flat + ibuf_flat
    oaddr = len(emif_data)
    emif_yaml["fill"] = copy.deepcopy(emif_data)
    print(waddrs)
    print(iaddrs)
    print(oaddrs)
    print(" ========> Start Simulation") 
    print("*"*50)
    outvals, testinst = generate_modules.simulate_accelerator(
        module_name="test_odin_emif_sm", 
        mlb_spec=mlb_yaml,
        wb_spec=wb_yaml,
        ab_spec=ab_yaml,
        emif_spec=emif_yaml,
        projections=proj_yamls,
        write_to_file=True,
        waddrs=waddrs,
        iaddrs=iaddrs,
        oaddrs=oaddrs,
        ws=ws,
        validate_output=v,
        layer_sel=range(len(proj_yamls)))
    print(" ========> Done Simulation") 
    
    print(" ========> Read out initial inputs from EMIF and check against expected" + str(py_i))
    ivalues_per_buf0, ibuf_len0, ibuf_count0 = utils.get_iw_buffer_dimensions(
            ab_yaml, proj_yamls[0], 'I')
    print(ivalues_per_buf)
    print(oaddrs[0]-iaddrs[0])
    print(iaddrs[0])
    print(ibuf_flat)
    print(emif_data)
    emif_vals = utils.read_out_stored_values_from_emif(
        testinst.emif_inst.sim_model.emif_inner_inst, ivalues_per_buf0, oaddrs[0]-iaddrs[0],
        proj_yaml["data_widths"]["I"], iaddrs[0])
    print(emif_vals)
    print(ibuf)
    for k in range(len(ibuf)):
        for j in range(len(ibuf[k])):
            for i in range(len(ibuf[k][j])):
                print(emif_vals[k*len(ibuf[k])+j][i])
                print(ibuf[k][j][i])
                assert emif_vals[k*len(ibuf[k])+j][i] == ibuf[k][j][i]

    for py_i in range(len(proj_yamls)):
        proj_yaml=proj_yamls[py_i]
          
        print(" ========> Read out Weights from EMIF and check against expected - L" + str(py_i))
        endaddr = waddrs[py_i+1] if (py_i < len(proj_yamls)-1) else iaddrs[0]
        emif_vals = utils.read_out_stored_values_from_emif(
            testinst.emif_inst.sim_model.emif_inner_inst, wvalues_per_buf, endaddr-waddrs[py_i],
            proj_yaml["data_widths"]["W"], waddrs[py_i])
        print(wbufs[py_i])
        print(emif_vals)
        for k in range(len(wbufs[py_i])):
            for j in range(len(wbufs[py_i][k])):
                for i in range(len(wbufs[py_i][k][j])):
                    assert emif_vals[k*len(wbufs[py_i][k])+j][i] == wbufs[py_i][k][j][i]
                
    proj_yaml=proj_yamls[len(proj_yamls)-1]
    # Calculate buffer dimensions info
    wvalues_per_buf, wbuf_len, wbuf_count = utils.get_iw_buffer_dimensions(
        wb_yaml, proj_yaml, 'W')
    ivalues_per_buf, ibuf_len, ibuf_count = utils.get_iw_buffer_dimensions(
        ab_yaml, proj_yaml, 'I')
    ovalues_per_buf, obuf_len, obuf_count = utils.get_obuffer_dimensions(
        ab_yaml, proj_yaml) 
    
    inner_uw = proj_yaml["inner_projection"]["RX"]
    inner_uwx = proj_yaml["inner_projection"]["RX"]
    inner_uwy = 1
    outer_uw = proj_yaml["outer_projection"]["RX"]
    outer_uwx = proj_yaml["outer_projection"]["RX"]
    outer_uwy = 1
    
    inner_unc = proj_yaml["inner_projection"]["C"]
    inner_unx = 1
    inner_uny = proj_yaml["inner_projection"]["RY"]
    inner_un = inner_unc * inner_uny
    outer_unc = proj_yaml["outer_projection"]["C"]
    outer_unx = 1
    outer_uny = proj_yaml["outer_projection"]["RY"]
    outer_un = outer_unc * outer_uny
    temp_unc = proj_yaml.get("temporal_projection",{}).get("C",1)
    temp_unx = 1
    temp_uny = proj_yaml.get("temporal_projection",{}).get("RY",1)
    temp_un = temp_unc * temp_uny
    
    inner_ue = proj_yaml["inner_projection"]["E"]
    outer_ue = proj_yaml["outer_projection"]["E"]
    temp_ue = proj_yaml.get("temporal_projection",{}).get("E",1)
    
    inner_ubb = proj_yaml["inner_projection"]["B"]
    inner_ubx = proj_yaml["inner_projection"]["PX"]
    inner_uby = proj_yaml["inner_projection"]["PY"]
    inner_ub = inner_ubb * inner_ubx * inner_uby
    outer_ubb = proj_yaml["outer_projection"]["B"]
    outer_ubx = proj_yaml["outer_projection"]["PX"]
    outer_uby = proj_yaml["outer_projection"]["PY"]
    outer_ub = outer_ubb * outer_ubx * outer_uby
    temp_ubb = proj_yaml.get("temporal_projection",{}).get("B",1)
    temp_ubx = proj_yaml.get("temporal_projection",{}).get("PX", obuf_len)
    temp_uby = proj_yaml.get("temporal_projection",{}).get("PY", 1)
    temp_ub = temp_ubb * temp_ubx * temp_uby
    
    inner_ug = proj_yaml["inner_projection"]["G"]
    outer_ug = proj_yaml["outer_projection"]["G"]
    temp_ug = proj_yaml.get("temporal_projection",{}).get("G",1)
    stridex = proj_yaml.get("stride",{}).get("x",1)
    stridey = proj_yaml.get("stride",{}).get("y",1)

    print(" ========> Collect actual final outputs")
    with open("final_offchip_data_contents.yaml") as outfile:
        outvals_yaml = yaml.safe_load(outfile)
    print(layer_output)
    actual_outputs = [[[[[0
                     for k in range(len(layer_output[t][l][j][i]))]  # x
                     for i in range(len(layer_output[t][l][j]))]      # y    
                     for j in range(len(layer_output[t][l]))]       # chans
                     for l in range(len(layer_output[t]))]       # batch
                     for t in range(len(layer_output))]       # group o%(2**proj_yaml["data_widths"]["I"])
    if (py_i == len(proj_yamls)-1):
        for ugt in range(temp_ug):
            for ugo in range(outer_ug): 
                for ugi in range(inner_ug):
                    for ubox in range(outer_ubx):
                        for ubix in range(inner_ubx):
                            for ubtx in range(int(temp_ubx/stridex)):
                                for uboy in range(outer_uby):
                                    for ubiy in range(inner_uby):
                                        for ubty in range(int(temp_uby/stridey)):
                                            for ubob in range(outer_ubb):
                                                for ubib in range(inner_ubb):
                                                    for ubtb in range(temp_ubb):
                                                        for ueo in range(outer_ue):
                                                            for uei in range(inner_ue):
                                                                for uet in range(temp_ue):
                                                                    ubo = ubox*outer_uby + uboy + ubob*outer_ubx*outer_uby
                                                                    ubi = ubix*inner_uby + ubiy + ubib*inner_ubx*inner_uby 
                                                                    ubt = ubty*int(temp_ubx/stridex) + ubtx + ubtb*int(temp_ubx/stridex)*int(temp_uby/stridey)
                                                                    
                                                                    out_act_idx = ugo*outer_ub*outer_ue*inner_ug*inner_ub*inner_ue + \
                                                                                  ubo*outer_ue*inner_ug*inner_ub*inner_ue + \
                                                                                  ueo*inner_ug*inner_ub*inner_ue + \
                                                                                  ugi*inner_ub*inner_ue + \
                                                                                  ubi*inner_ue + uei
                                                                    obuf_idx = math.floor(out_act_idx/ovalues_per_buf)
                                                                    os_idx = out_act_idx % ovalues_per_buf
                                                                    ubx = ubtx*outer_ubx*inner_ubx+ubox*inner_ubx+ubix
                                                                    uby = ubty*outer_uby*inner_uby+uboy*inner_uby+ubiy
                                                                    ubb = ubtb*outer_ubb*inner_ubb+ubob*inner_ubb+ubib
                                                                    max_ubx = outer_ubx*inner_ubx*temp_ubx
                                                                    max_un = inner_uwx*outer_uwx*inner_unx*outer_unx*temp_unx*inner_uwy*outer_uwy*inner_unx*outer_unx*temp_uny
                                                                    if (ubx <= (max_ubx - max_un)/stridex):
                                                                        correct_val = outvals_yaml[py_i][obuf_idx*min(obuf_len,ibuf_len) + ugt*temp_ub*temp_ue+uet*temp_ub+ubt][os_idx]                                                           
                                                                        actual_outputs[ugt*outer_ug*inner_ug+ugo*inner_ug+ugi]\
                                                                            [ubb]\
                                                                            [uet*outer_ue*inner_ue+ueo*inner_ue+uei]\
                                                                            [uby][ubx] = correct_val
       
    print(" ========> Compare outputs against expected")
    print(layers)
    print("Weights")
    print(weights)
    print("Inputs")
    print(inputs)
    print("Expected Outputs")
    print(layer_output)
    print("Actual Outputs")
    print(actual_outputs)
    
    print("Raw Outputs")
    print(outvals_yaml)
    print(waddrs)
    print(iaddrs)
    print(oaddrs)
    
    if (py_i == len(proj_yamls) - 1):
        assert actual_outputs == layer_output
  
@pytest.mark.parametrize(
    "mlb_file,ab_file,wb_file,emif_file,proj_file,ws", filesets
)
@pytest.mark.full_simulations
@pytest.mark.skip
def test_simulate_layer(
        mlb_file, ab_file, wb_file, emif_file, proj_file, ws, v=True, pp=False):
    
    # Make sure that output gets through odin.
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            mlb_file)
    ab_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ab_file)
    wb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            wb_file)
    emif_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            emif_file)
    proj_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            proj_file)
    
    with open(mlb_spec) as stream:
        mlb_yaml = yaml.safe_load(stream)
    with open(ab_spec) as stream:
        ab_yaml = yaml.safe_load(stream)
    with open(wb_spec) as stream:
        wb_yaml = yaml.safe_load(stream)
    with open(proj_spec) as stream:
        proj_yaml = yaml.safe_load(stream)
    with open(emif_spec) as stream:
        emif_yaml = yaml.safe_load(stream)

    # Calculate buffer dimensions info
    layer_type = mlb_yaml.get('MAC_info', {}).get('type', 'MAC')
    wvalues_per_buf, wbuf_len, wbuf_count = utils.get_iw_buffer_dimensions(
        wb_yaml, proj_yaml, 'W')
    ivalues_per_buf, ibuf_len, ibuf_count = utils.get_iw_buffer_dimensions(
        ab_yaml, proj_yaml, 'I')
    ovalues_per_buf, obuf_len, obuf_count = utils.get_obuffer_dimensions(
        ab_yaml, proj_yaml)  
    
    inner_uw = proj_yaml["inner_projection"]["RX"]
    inner_uwx = proj_yaml["inner_projection"]["RX"]
    inner_uwy = 1
    outer_uw = proj_yaml["outer_projection"]["RX"]
    outer_uwx = proj_yaml["outer_projection"]["RX"]
    outer_uwy = 1
    assert((inner_uwx == 1) | (inner_uwy == 1)) # Can't window in both directions
    assert((outer_uwx == 1) | (outer_uwy == 1))
    assert((inner_uwx == 1) | (outer_uwy == 1))
    assert((outer_uwx == 1) | (inner_uwy == 1))
    
    inner_unc = proj_yaml["inner_projection"]["C"]
    inner_unx = 1
    inner_uny = proj_yaml["inner_projection"]["RY"]
    inner_un = inner_unc * inner_uny
    outer_unc = proj_yaml["outer_projection"]["C"]
    outer_unx = 1
    outer_uny = proj_yaml["outer_projection"]["RY"]
    outer_un = outer_unc * outer_uny
    temp_unc = proj_yaml.get("temporal_projection",{}).get("C",1)
    temp_unx = 1
    temp_uny = proj_yaml.get("temporal_projection",{}).get("RY",1)
    temp_un = temp_unc * temp_uny
    assert(((inner_uwx == 1) & (outer_uwx == 1)) | ((inner_unx == 1) & (outer_unx == 1)))
    assert(((inner_uwy == 1) & (outer_uwy == 1)) | ((inner_uny == 1) & (outer_uny == 1)))

    inner_ue = proj_yaml["inner_projection"]["E"]
    outer_ue = proj_yaml["outer_projection"]["E"]
    temp_ue = proj_yaml.get("temporal_projection",{}).get("E",1)
    
        
    inner_ubb = proj_yaml["inner_projection"]["B"]
    inner_ubx = proj_yaml["inner_projection"]["PX"]
    inner_uby = proj_yaml["inner_projection"]["PY"]
    inner_ub = inner_ubb * inner_ubx * inner_uby
    outer_ubb = proj_yaml["outer_projection"]["B"]
    outer_ubx = proj_yaml["outer_projection"]["PX"]
    outer_uby = proj_yaml["outer_projection"]["PY"]
    outer_ub = outer_ubb * outer_ubx * outer_uby
    temp_ubb = proj_yaml.get("temporal_projection",{}).get("B",1)
    temp_ubx = proj_yaml.get("temporal_projection",{}).get("PX", obuf_len)
    temp_uby = proj_yaml.get("temporal_projection",{}).get("PY", 1)
    temp_ub = temp_ubb * temp_ubx * temp_uby
    assert(((inner_uwx == 1) & (outer_uwx == 1)) | ((inner_ubx == 1) & (outer_ubx == 1)))
    assert(((inner_uwy == 1) & (outer_uwy == 1)) | ((inner_uby == 1) & (outer_uby == 1)))
    
    inner_ug = proj_yaml["inner_projection"]["G"]
    outer_ug = proj_yaml["outer_projection"]["G"]
    temp_ug = proj_yaml.get("temporal_projection",{}).get("G",1)
    dilx = proj_yaml.get("dilation",{}).get("x",1)
    dily = proj_yaml.get("dilation",{}).get("y",1)
    stridex = proj_yaml.get("stride",{}).get("x",1)
    stridey = proj_yaml.get("stride",{}).get("y",1)
    
    # Figure out the layer dimensions based on the projection...
    layer = {"group": inner_ug*outer_ug*temp_ug,
             "batches": inner_ubb*outer_ubb*temp_ubb,
             "out_chans": inner_ue*outer_ue*temp_ue,
             "in_chans": inner_unc*outer_unc*temp_unc,
             "image_x": inner_ubx*outer_ubx*temp_ubx,
             "image_y": inner_uby*outer_uby*temp_uby,
             "filter_x": inner_uwx*outer_uwx*inner_unx*outer_unx*temp_unx,
             "filter_y": inner_uwy*outer_uwy*inner_uny*outer_uny*temp_uny,
             "stridex": stridex,
             "stridey": stridey,
             "dilx": dilx,
             "dily": dily
    }
    print("==> Layer information:")
    print(layer)

    # Create random input data arrays to load into EMIF
    weights = [[[[[random.randint(1,4) #(2**proj_yaml["data_widths"]["W"])-1)
                   for k in range(layer["filter_x"])]    # x
                   for i in range(layer["filter_y"])]    # y    
                   for j in range(layer["in_chans"])]    # ichans
                   for l in range(layer["out_chans"])]   # ochans
                   for t in range(layer["group"])]       # group
    inputs = [[[[[random.randint(0,3) #(2**proj_yaml["data_widths"]["I"])-1)
                   for k in range(layer["image_x"])]     # x
                   for i in range(layer["image_y"])]     # y    
                   for j in range(layer["in_chans"])]    # chans
                   for l in range(layer["batches"])]     # batch
                   for t in range(layer["group"])]       # group
    layer_outputs = utils.compute_layer(inputs, weights, layer, layer_type,
                                        output_width=proj_yaml["data_widths"]["O"],
                                        final_width=proj_yaml["data_widths"]["I"],
                                        activation_function=proj_yaml["activation_function"])
    layer_outputs = [[[[[layer_outputs[t][l][j][i][k]%(2**proj_yaml["data_widths"]["I"])
                         for k in range(len(layer_outputs[t][l][j][i]))]  # x
                         for i in range(len(layer_outputs[t][l][j]))]      # y    
                         for j in range(len(layer_outputs[t][l]))]       # chans
                         for l in range(len(layer_outputs[t]))]       # batch
                         for t in range(len(layer_outputs))]       # group o%(2**proj_yaml["data_widths"]["I"])
    
    # Move the weights and inputs into the EMIF in the expected order
    wbuf = reorder_weight_array(weights,proj_yaml, wb_yaml)
    ibuf = reorder_input_array(inputs,proj_yaml, ab_yaml, obuf_len)
    wbuf_flat = [sum((lambda i: inner[i] * \
                      (2**(i*proj_yaml["data_widths"]["W"])))(i) \
                     for i in range(len(inner))) \
                         for outer in wbuf for inner in outer]
    iaddr = len(wbuf_flat)
    ibuf_flat = [sum((lambda i: inner[i] * \
                (2**(i*proj_yaml["data_widths"]["I"])))(i) \
                     for i in range(len(inner))) \
                          for outer in ibuf for inner in outer]
    emif_data = wbuf_flat + ibuf_flat
    oaddr = len(emif_data)
    
    emif_yaml["fill"] = copy.deepcopy(emif_data)
    print("==> Start simulation")
    outvals, testinst = generate_modules.simulate_accelerator(
        module_name="test_odin_emif_sm", 
                                                    mlb_spec=mlb_yaml,
                                                    wb_spec=wb_yaml,
                                                    ab_spec=ab_yaml,
                                                    emif_spec=emif_yaml,
                                                    projections=proj_yaml,
                                                    write_to_file=True,
                                                    waddrs=[0],
                                                    iaddrs=[iaddr],
                                                    oaddrs=[oaddr],
                                                    ws=ws,
                                                    validate_output=v,
                                                    pingpong_w=pp)
    print("==> Done simulation")
    # Check that EMIFs have the right data
    emif_vals = utils.read_out_stored_values_from_emif(
        testinst.emif_inst.sim_model.emif_inner_inst, wvalues_per_buf, iaddr,
        proj_yaml["data_widths"]["W"], 0)
    print("==> Checking that weights are correctly stored in EMIF")
    print(wbuf)
    print(emif_vals)
    for k in range(len(wbuf)):
        for j in range(len(wbuf[k])):
            for i in range(len(wbuf[k][j])):
                assert emif_vals[k*len(wbuf[k])+j][i] == wbuf[k][j][i]
    emif_vals = utils.read_out_stored_values_from_emif(
        testinst.emif_inst.sim_model.emif_inner_inst, ivalues_per_buf, oaddr-iaddr,
        proj_yaml["data_widths"]["I"], iaddr)
    print("==> Checking that inputs are correctly stored in EMIF")
    print(emif_vals)
    for k in range(len(ibuf)):
        for j in range(len(ibuf[k])):
            for i in range(len(ibuf[k][j])):
                assert emif_vals[k*len(ibuf[k])+j][i] == ibuf[k][j][i]

    # Check that the right data got into the on-chip buffers
    print("==> Check that EMIF data gets into the on-chip buffers - weights:")
    check_buffers(testinst.datapath, testinst.datapath.weight_modules,
                  "ml_block_weights_inst_{}",
                  wbuf, proj_yaml["data_widths"]["W"], testinst)
    print("==> Check that EMIF data gets into the on-chip buffers - inputs:")
    check_buffers(testinst.datapath, testinst.datapath.input_act_modules,
                  "ml_block_input_inst_{}",
                  ibuf, proj_yaml["data_widths"]["I"], testinst)

    with open("final_offchip_data_contents.yaml") as outfile:
        outvals_yaml = yaml.safe_load(outfile)[0]
    actual_outputs = [[[[['x'
                     for k in range(len(layer_outputs[t][l][j][i]))]  # x
                     for i in range(len(layer_outputs[t][l][j]))]      # y    
                     for j in range(len(layer_outputs[t][l]))]       # chans
                     for l in range(len(layer_outputs[t]))]       # batch
                     for t in range(len(layer_outputs))]       # group o%(2**proj_yaml["data_widths"]["I"])
                            
    actual_outputs = reorder_output_array(outvals_yaml, proj_yaml, ab_yaml, actual_outputs, ibuf_len)
    print("==> Comparing outputs with expected output array:")
    print("Weights")
    print(weights)
    print("Inputs")
    print(inputs)
    print("Expected outs")
    print(layer_outputs)
    print("Actual outs")
    print(actual_outputs)
    print(outvals_yaml)
    not_all_x = False
    for j in range(len(actual_outputs)):
        for k in range(len(actual_outputs[j])):
            for l in range(len(actual_outputs[j][k])):
                for m in range(len(actual_outputs[j][k][l])):
                    for n in range(len(actual_outputs[j][k][l][m])):
                        if not (actual_outputs[j][k][l][m][n] == 'x'):
                            not_all_x = True
                            assert(actual_outputs[j][k][l][m][n] == layer_outputs[j][k][l][m][n])
    assert(not_all_x)

    

@pytest.mark.parametrize(
    "mlb_file,ab_file,wb_file,emif_file,proj_file,ws", filesets
)
@pytest.mark.full_simulations
@pytest.mark.skip
def test_simulate_emif_statemachine(
        mlb_file, ab_file, wb_file, emif_file, proj_file, ws, v=True):
    
    # Make sure that output gets through odin.
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            mlb_file)
    ab_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ab_file)
    wb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            wb_file)
    emif_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            emif_file)
    proj_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            proj_file)
    with open(mlb_spec) as stream:
        mlb_yaml = yaml.safe_load(stream)
    with open(ab_spec) as stream:
        ab_yaml = yaml.safe_load(stream)
    with open(wb_spec) as stream:
        wb_yaml = yaml.safe_load(stream)
    with open(proj_spec) as stream:
        proj_yaml = yaml.safe_load(stream)
    with open(emif_spec) as stream:
        emif_yaml = yaml.safe_load(stream)
        
    # Calculate buffer dimensions info
    wvalues_per_buf, wbuf_len, wbuf_count = utils.get_iw_buffer_dimensions(
        wb_yaml, proj_yaml, 'W')
    ivalues_per_buf, ibuf_len, ibuf_count = utils.get_iw_buffer_dimensions(
        ab_yaml, proj_yaml, 'I')
    ovalues_per_buf, obuf_len, obuf_count = utils.get_obuffer_dimensions(
        ab_yaml, proj_yaml)

    # Create random input data arrays to load into EMIF
    wbuf = [[[random.randint(0,(2**proj_yaml["data_widths"]["W"])-1)
            for k in range(wvalues_per_buf)]    # values per word
            for i in range(wbuf_len)]           # words per buffer
            for j in range(wbuf_count)]         # buffer count
    wbuf_flat = [sum((lambda i: inner[i] * \
                      (2**(i*proj_yaml["data_widths"]["W"])))(i) \
                     for i in range(len(inner))) \
                         for outer in wbuf for inner in outer]
    iaddr = len(wbuf_flat)
    ibuf = [[[random.randint(0,(2**proj_yaml["data_widths"]["I"])-1)
             for k in range(ivalues_per_buf)]            # values per word
             for i in range(ibuf_len)]                   # words per buffer
             for j in range (ibuf_count)]                # buffers
    ibuf_flat = [sum((lambda i: inner[i] * \
                (2**(i*proj_yaml["data_widths"]["I"])))(i) \
                     for i in range(len(inner))) \
                          for outer in ibuf for inner in outer]
    emif_data = wbuf_flat + ibuf_flat
    print(wbuf_flat)
    oaddr = len(emif_data)
    
    emif_yaml["fill"] = copy.deepcopy(emif_data)
    outvals, testinst = generate_modules.simulate_accelerator(
        module_name="test_odin_emif_sm", 
                                                    mlb_spec=mlb_yaml,
                                                    wb_spec=wb_yaml,
                                                    ab_spec=ab_yaml,
                                                    emif_spec=emif_yaml,
                                                    projections=proj_yaml,
                                                    write_to_file=True,
                                                    waddrs=[0],
                                                    iaddrs=[iaddr],
                                                    oaddrs=[oaddr],
                                                    ws=ws,
                                                    validate_output=v,
                                                    gen_ver=True)
    print("done simulating")
    # Check that EMIFs have the right data
    emif_vals = utils.read_out_stored_values_from_emif(
        testinst.emif_inst.sim_model.emif_inner_inst, wvalues_per_buf, iaddr,
        proj_yaml["data_widths"]["W"], 0)
    print("\n\nCOMPARE")
    print(emif_vals)
    print("WITH")
    print(wbuf)
    for k in range(len(wbuf)):
        for j in range(len(wbuf[k])):
            for i in range(len(wbuf[k][j])):
                assert emif_vals[k*len(wbuf[k])+j][i] == wbuf[k][j][i]
                
    emif_vals = utils.read_out_stored_values_from_emif(
        testinst.emif_inst.sim_model.emif_inner_inst, ivalues_per_buf, oaddr-iaddr,
        proj_yaml["data_widths"]["I"], iaddr)
    print("\n\nCOMPARE")
    print(emif_vals)
    print("WITH")
    print(ibuf)
    for k in range(len(ibuf)):
        for j in range(len(ibuf[k])):
            for i in range(len(ibuf[k][j])):
                assert emif_vals[k*len(ibuf[k])+j][i] == ibuf[k][j][i]

    # Check that the right data got into the on-chip buffers
    #check_buffers(testinst.datapath, testinst.datapath.weight_modules,
    #              "ml_block_weights_inst_{}",
    #              wbuf, proj_yaml["data_widths"]["W"], testinst)
    #check_buffers(testinst.datapath, testinst.datapath.input_act_modules,
    #              "ml_block_input_inst_{}",
    #              ibuf, proj_yaml["data_widths"]["I"], testinst)
    
    # Check that the right data is in the MLBs
    #print(testinst.datapath.mlb_modules.ml_block_inst_0.curr_inst.sim_model.mac_modules.input_out_0)
    #print(testinst.datapath.mlb_modules.ml_block_inst_0.curr_inst.sim_model.mac_modules.sum_out_0)
    #if (ws):
    #    assert(check_weight_contents(
    #        testinst.datapath, proj_yaml,
    #        "ml_block_inst_{}", "weight_out_{}", wbuf))

    print("\n\n\n\nHERE")
    obuf = [[[0 for i in range(ovalues_per_buf)]
             for i in range(obuf_len)]
             for j in range (obuf_count)]
    obuf = utils.get_expected_outputs(obuf, ovalues_per_buf,
                                wbuf,
                                ibuf, ivalues_per_buf,
                                proj_yaml)

    print("EXPECTED OUT")
    print(obuf)
    print("\nACTUAL OUT")
    #ibuf2 = [[[0 for k in range(ivalues_per_buf)]            # values per word
    #         for i in range(ibuf_len)]                   # words per buffer
    #         for j in range (2)]                # buffers

    #print("%%%%%%%%%%% Check inputs again")
    #check_buffers(testinst.datapath, testinst.datapath.input_act_modules,
    #              "ml_block_input_inst_{}",
    #              ibuf2, proj_yaml["data_widths"]["I"], testinst)
    
    with open("final_offchip_data_contents.yaml") as outfile:
        outvals_yaml = yaml.safe_load(outfile)[0]
            
    for emif_inner_inst in range(obuf_count):
        for olen in range(min(obuf_len,ibuf_len)-1):
            print(obuf[emif_inner_inst][olen])
            print(outvals_yaml[emif_inner_inst*min(obuf_len,ibuf_len) + olen])
            assert obuf[emif_inner_inst][olen] == outvals_yaml[emif_inner_inst*min(obuf_len,ibuf_len) + olen]
    
def test_simulate_emif_statemachine_unit_ws_pl():
    test_simulate_emif_statemachine("mlb_spec_3.yaml",
                               "input_spec_1.yaml",
                               "weight_spec_3.yaml",
                               "emif_spec_1.yaml",
                               "projection_spec_5.yaml", True, True)
    
def test_simulate_emif_statemachine_unit_ws_bc():
    test_simulate_emif_statemachine("mlb_spec_3.yaml",
                               "input_spec_1.yaml",
                               "weight_spec_3.yaml",
                               "emif_spec_1.yaml",
                                    "projection_spec_6.yaml", True, False)
    
def test_simulate_layer_ws_bc():
    test_simulate_layer("mlb_spec_3.yaml",
                               "input_spec_1.yaml",
                               "weight_spec_3.yaml",
                               "emif_spec_1.yaml",
                                    "projection_spec_8.yaml", True, False)
    
def test_simulate_layer_n0():
    test_simulate_layer("mlb_spec_3.yaml",
                        "input_spec_intel_8.yaml",
                        "weight_spec_3.yaml",
                        "emif_spec_1.yaml",
                        "projection_spec_10.yaml", True, False)
    
def test_simulate_layer_n1():
    test_simulate_layer("mlb_spec_3.yaml",
                        "input_spec_2.yaml",
                        "weight_spec_3.yaml",
                        "emif_spec_1.yaml",
                        "projection_spec_11.yaml", True, False)
    
def test_simulate_layer_n2():
    test_simulate_layer("mlb_spec_3.yaml",
                        "input_spec_2.yaml",
                        "weight_spec_3.yaml",
                        "emif_spec_1.yaml",
                        "projection_spec_12.yaml", False, False)
    
def test_simulate_layer_n3():
    test_simulate_layer("mlb_spec_3.yaml",
                        "input_spec_2.yaml",
                        "weight_spec_3.yaml",
                        "emif_spec_1.yaml",
                        "projection_spec_13.yaml", True, False)
    
def test_simulate_layer_n4():
    test_simulate_layer("mlb_spec_3.yaml",
                        "input_spec_2.yaml",
                        "weight_spec_3.yaml",
                        "emif_spec_1.yaml",
                        "projection_spec_14.yaml", True, False)
    
def test_simulate_layer_n4_pp():
    test_simulate_layer("mlb_spec_3.yaml",
                        "input_spec_2.yaml",
                        "weight_spec_3.yaml",
                        "emif_spec_1.yaml",
                        "projection_spec_14.yaml", True, False, True)
    
def test_simulate_layer_maxpool():
    test_simulate_layer("mlb_spec_mp.yaml",
                        "input_spec_2.yaml",
                        "weight_spec_3.yaml",
                        "emif_spec_1.yaml",
                        "projection_spec_mp.yaml", True, False)
    
def test_simulate_layer_urn():
    test_simulate_layer("mlb_spec_3.yaml",
                        "input_spec_2.yaml",
                        "weight_spec_3.yaml",
                        "emif_spec_1.yaml",
                        "projection_spec_9.yaml", True, False)
            
def test_simulate_layer_intel():
    test_simulate_layer("mlb_spec_intel.yaml",
                        "input_spec_intel_8.yaml",
                        "weight_spec_3.yaml",
                        "emif_spec_1.yaml",
                        "projection_spec_cs.yaml", True, False)
    
def test_simulate_layer_soft_e():
    test_simulate_layer("mlb_spec_4.yaml",
                        "input_spec_2.yaml",
                        "weight_spec_3.yaml",
                        "emif_spec_1.yaml",
                        "projection_spec_15.yaml", True, False)
    
def test_simulate_layer_soft_b():
    test_simulate_layer("mlb_spec_4.yaml",
                        "input_spec_2.yaml",
                        "weight_spec_3.yaml",
                        "emif_spec_1.yaml",
                        "projection_spec_16.yaml", True, False)
    
def test_simulate_layer_xilinx_mode1():
    test_simulate_layer("mlb_spec_xilinx_mode1.yaml",
                        "input_spec_2.yaml",
                        "weight_spec_3.yaml",
                        "emif_spec_1.yaml",
                        "projection_spec_x1.yaml", True, False)
    
def test_simulate_layer_xilinx_mode2():
    test_simulate_layer("mlb_spec_xilinx_mode2.yaml",
                        "input_spec_2.yaml",
                        "weight_spec_3.yaml",
                        "emif_spec_1.yaml",
                        "projection_spec_x2.yaml", True, False)
    
def test_simulate_multiple_layer_ws_bc():
    test_simulate_multiple_layers("mlb_spec_3.yaml",
                               "input_spec_1.yaml",
                               "weight_spec_3.yaml",
                               "emif_spec_1.yaml",
                               "projection_spec_multiple.yaml", True, False)
    #assert(0)

    
def test_simulate_emif_statemachine_unit_os_bc():
    test_simulate_emif_statemachine("mlb_spec_3.yaml",
                               "input_spec_1.yaml",
                               "weight_spec_3.yaml",
                               "emif_spec_1.yaml",
                               "projection_spec_7.yaml", False, False)

    
def test_simulate_random_emif_statemachine():

    # Make sure that output gets through odin.
    mlb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mlb_spec_0.yaml")
    ab_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "input_spec_0.yaml")
    wb_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "weight_spec_0.yaml")
    emif_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "emif_spec_0.yaml")
    proj_spec = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "projection_spec_0.yaml")
    
    with open(mlb_spec) as stream:
        mlb_yaml = yaml.safe_load(stream)
    with open(ab_spec) as stream:
        ab_yaml = yaml.safe_load(stream)
    with open(wb_spec) as stream:
        wb_yaml = yaml.safe_load(stream)
    with open(proj_spec) as stream:
        proj_yaml = yaml.safe_load(stream)
    with open(emif_spec) as stream:
        emif_yaml = yaml.safe_load(stream)
  
    # Calculate buffer dimensions info
    wvalues_per_buf, wbuf_len, wbuf_count = utils.get_iw_buffer_dimensions(
        wb_yaml, proj_yaml, 'W')
    ivalues_per_buf, ibuf_len, ibuf_count = utils.get_iw_buffer_dimensions(
        ab_yaml, proj_yaml, 'I')
    #print(ibuf_count)
    #assert(ibuf_count == 80)      
    ovalues_per_buf, obuf_len, obuf_count = utils.get_obuffer_dimensions(
        ab_yaml, proj_yaml)

    # Create random input data arrays to load into EMIF
    wbuf = [[[random.randint(0,(2**proj_yaml["data_widths"]["W"])-1)
            for k in range(wvalues_per_buf)]    # values per word
            for i in range(wbuf_len)]           # words per buffer
            for j in range(wbuf_count)]         # buffer count
    wbuf_flat = [sum((lambda i: inner[i] * \
                      (2**(i*proj_yaml["data_widths"]["W"])))(i) \
                     for i in range(len(inner))) \
                         for outer in wbuf for inner in outer]
    iaddr = len(wbuf_flat)
    ibuf = [[[random.randint(0,(2**proj_yaml["data_widths"]["I"])-1)
             for k in range(ivalues_per_buf)]            # values per word
             for i in range(ibuf_len)]                   # words per buffer
             for j in range (ibuf_count)]                # buffers
    ibuf_flat = [sum((lambda i: inner[i] * \
                (2**(i*proj_yaml["data_widths"]["W"])))(i) \
                     for i in range(len(inner))) \
                          for outer in ibuf for inner in outer]
    emif_data = wbuf_flat + ibuf_flat
    oaddr = len(emif_data)
    
    emif_yaml["fill"] = emif_data
    outvals, testinst = generate_modules.simulate_accelerator_with_random_input(
        module_name="test_odin_emif_sm", 
                                                    mlb_spec=mlb_yaml,
                                                    wb_spec=wb_yaml,
                                                    ab_spec=ab_yaml,
                                                    emif_spec=emif_yaml,
                                                    projection=proj_yaml,
                                                    write_to_file=False,
                                                    waddr=0,
                                                    iaddr=iaddr,
                                                    oaddr=oaddr)

