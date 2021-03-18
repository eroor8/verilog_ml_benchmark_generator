""""
PYM-TL Component Classes Implementing different parts of the dataflow
Test ASSUMPTIONS:
- Weights are always preloaded
- Weight stationary flow
- Inputs don't require muxing

"""
from pymtl3 import Component, OutPort, InPort, update, update_ff, Wire, Bits5
import math
import module_helper_classes
import module_classes
import utils
from utils import printi
il = 1


class SM_BufferWen(Component):
    def construct(s, buf_count_width, curr_buffer_count):
        utils.AddInPort(s, 1, "we_in")
        if (buf_count_width > 0):
            assert(curr_buffer_count < 2 ** buf_count_width)
            utils.AddInPort(s, buf_count_width, "buffer_count")

            @update
            def upblk_set_wen0():
                if (s.buffer_count == curr_buffer_count):
                    s.we @= s.we_in
                else:
                    s.we @= 0
        else:

            @update
            def upblk_set_wen1():
                s.we @= s.we_in
        utils.AddOutPort(s, 1, "we")
        utils.tie_off_clk_reset(s)


class SM_PreloadMLB(Component):
    # Load values from a buffer into the ML Blocks
    # Inputs:
    #  - start
    #  - start address
    # Outputs:
    #  - address
    #  - rdy
    #  - wen
    # On start, assert wen and set address to start address
    # for outer_tile_i in num_outer_tiles:
    #     for outer_repeat_i in outer_tile_repeat_x:
    #         for inner_tile_i in outer_tile_size/inner_tile_size:
    #             for inner_repeat_i  in inner_tile_repeat_x
    def construct(s, addr_width,
                  num_outer_tiles, outer_tile_size,
                  inner_tile_size,
                  outer_tile_repeat_x, inner_tile_repeat_x):
        """ SM_PreloadMLB constructor
            :param addr_width: Width of address port
            :type addr_width: int
            :param num_outer_tiles: Number of outer tiles to iterate through
            :type  num_outer_tiles: bool
            :param outer_tile_size: Size of outer tiles
            :type outer_tile_size: int
            :param inner_tile_size: Size of inner tile
            :type inner_tile_size: int
            :param outer_tile_repeat_x: Number of times to repeat outer tile
            :type outer_tile_repeat_x: int
            :param inner_tile_repeat_x: Number of times to repeat inner tile
            :type inner_tile_repeat_x: int
        """
        w_addr_width = max(int(math.log(outer_tile_size * num_outer_tiles * 2,
                                        2)), addr_width)
        s.start = InPort(1)
        s.outer_tile_repeat_count = Wire(max(int(math.log(outer_tile_repeat_x,
                                                          2)) + 1, 1))
        s.inner_tile_repeat_count = Wire(max(int(math.log(inner_tile_repeat_x,
                                                          2)) + 1, 1))
        s.index_within_inner_tile = Wire(w_addr_width)
        s.inner_tile_index = Wire(w_addr_width)
        max_inner_tile_idx = int(outer_tile_size / inner_tile_size)
        s.outer_tile_index = Wire(w_addr_width)
        s.address = OutPort(addr_width)
        s.addr_w = Wire(w_addr_width)
        s.rdy = OutPort(1)
        s.state = Wire(2)
        s.wen = OutPort(1)
        s.start_address = InPort(addr_width)
        s.w_start_address = Wire(w_addr_width)

        INIT, LOAD = 0, 1

        @update
        def upblk_preload_comb():
            s.w_start_address[0:addr_width] @= s.start_address
            s.addr_w @= s.index_within_inner_tile + s.inner_tile_index * \
                inner_tile_size + s.outer_tile_index * outer_tile_size + \
                s.w_start_address
            s.address @= s.addr_w[0:addr_width]

        @update_ff
        def upblk_preload_ff():
            if s.reset:
                s.wen <<= 0
                s.state <<= INIT
                s.rdy <<= 1
                s.outer_tile_repeat_count <<= 0
                s.inner_tile_repeat_count <<= 0
                s.index_within_inner_tile <<= 0
                s.inner_tile_index <<= 0
                s.outer_tile_index <<= 0
            else:
                if (s.state == INIT):
                    if (s.start):
                        s.state <<= LOAD
                        s.wen <<= 1
                        s.rdy <<= 0
                elif (s.state == LOAD):
                    if (s.index_within_inner_tile < (inner_tile_size - 1)):
                        s.index_within_inner_tile <<= \
                            s.index_within_inner_tile + 1
                    else:
                        s.index_within_inner_tile <<= 0
                        if (s.inner_tile_repeat_count <
                                (inner_tile_repeat_x - 1)):
                            s.inner_tile_repeat_count <<= \
                                s.inner_tile_repeat_count + 1
                        else:
                            s.inner_tile_repeat_count <<= 0
                            if (s.inner_tile_index < (max_inner_tile_idx - 1)):
                                s.inner_tile_index <<= s.inner_tile_index + 1
                            else:
                                s.inner_tile_index <<= 0
                                if (s.outer_tile_repeat_count <
                                        (outer_tile_repeat_x - 1)):
                                    s.outer_tile_repeat_count <<= \
                                        s.outer_tile_repeat_count + 1
                                else:
                                    s.outer_tile_repeat_count <<= 0
                                    if (s.outer_tile_index <
                                            (num_outer_tiles - 1)):
                                        s.outer_tile_index <<= \
                                            s.outer_tile_index + 1
                                    else:
                                        s.outer_tile_index <<= 0
                                        s.state <<= INIT
                                        s.wen <<= 0
                                        s.rdy <<= 1
        utils.tie_off_clk_reset(s)


class SM_IterateThruAddresses(Component):
    # Iterate through addresses
    # Inputs:
    #   - start
    #   - start address
    # Outputs:
    #   - incr
    #   - addr
    #   - rdy
    #   - wen
    # On start, do nothing for start_wait cycles.
    # assert wen, and increment address from start_address write_count times.
    # If skip_n > 0, then periodically deassert wen for skip_n cycles,
    # assert for one.
    def construct(s, write_count, addr_width, stride=1, skip_n=0,
                  skip_after=0, start_wait=0, debug_name='', repeat_x=1,
                  repeat_len=1, addr_b_offset=1, sel_count=1):
        """ SM_IterateThruAddresses constructor
            :param addr_width: Width of address port
            :type addr_width: int
            :param write_count: Number of addresses to increment through
            :type  write_count: int
            :param skip_n: Size of inner tile
            :type  skip_n: int
            :param start_wait: At the very beginning, wait for start_wait
                               cycles.
            :type  start_wait: int
        """
        assert(write_count > 0)
        if (write_count < 2):
            w_addr_width = addr_width + 4
        else:
            w_addr_width = max(int(math.ceil(math.log(write_count, 2))),
                               addr_width) + 4
        write_count = int(write_count)
        s.start = InPort(1)
        s.inner_incr = Wire(w_addr_width)
        s.incr = OutPort(w_addr_width)
        s.section_incr = Wire(w_addr_width)
        s.total_incr = Wire(w_addr_width)
        s.address = OutPort(addr_width)
        s.address_b = OutPort(addr_width)
        s.w_address = Wire(w_addr_width)
        s.address_offset = Wire(addr_width)
        s.start_address = InPort(addr_width)
        s.start_address_w = Wire(w_addr_width)
        s.start_address_w[0:addr_width] //= s.start_address
        if (w_addr_width > addr_width):
            s.start_address_w[addr_width:w_addr_width] //= 0
        s.rdy = OutPort(1)
        s.wen = OutPort(1)
        s.state = Wire(4)
        s.skip_cnt = Wire(w_addr_width)
        s.repeat_count = Wire(repeat_x + 1)
        sel_width = math.ceil(math.log(max(sel_count, 2), 2))
        s.w_urn_sel = Wire(sel_width * 2)
        s.urn_sel = OutPort(sel_width)
        swm = 0
        if (start_wait > 0):
            swm = start_wait - 1

        @update
        def upblk_set_wenc():
            if (s.wen):
                s.w_address @= s.incr + s.inner_incr + s.section_incr + \
                    s.start_address_w
            else:
                s.w_address @= 0
            s.address @= s.w_address[0:addr_width]
            s.address_b @= s.w_address[0:addr_width] + addr_b_offset
            s.total_incr @= s.incr + s.inner_incr + s.section_incr
            s.urn_sel @= s.w_urn_sel[0:sel_width]

        INIT, LOAD, START_WAIT = 0, 1, 2

        @update_ff
        def upblk_set_wen():
            if s.reset:
                s.state <<= INIT
                s.section_incr <<= 0
                s.incr <<= 0
                s.inner_incr <<= 0
                s.rdy <<= 1
                s.wen <<= 0
                s.skip_cnt <<= 0
                s.repeat_count <<= 0
                s.w_urn_sel <<= 0
            else:
                if (s.state == INIT):
                    if (s.start):
                        if (start_wait > 1):
                            s.state <<= START_WAIT
                        else:
                            s.state <<= LOAD
                        s.rdy <<= 0
                        if (((skip_n == 0) | (skip_after == 1)) &
                                (start_wait == 0)):
                            s.wen <<= 1
                        else:
                            s.wen <<= 0
                    else:
                        s.wen <<= 0
                        s.rdy <<= 1
                    s.incr <<= 0
                    s.inner_incr <<= 0
                    s.section_incr <<= 0
                    s.skip_cnt <<= 1
                elif (s.state == START_WAIT):
                    if (s.skip_cnt == swm):
                        s.state <<= LOAD
                        s.skip_cnt <<= 0
                        s.wen <<= 1
                    else:
                        s.skip_cnt <<= s.skip_cnt + 1
                elif (s.state == LOAD):
                    if (s.inner_incr + 1 >= addr_b_offset):
                        if (s.w_urn_sel >= sel_count - stride):
                            s.w_urn_sel <<= s.w_urn_sel - sel_count + stride
                        else:
                            s.w_urn_sel <<= s.w_urn_sel + stride
                    if ((s.total_incr + 1) >= write_count):
                        s.state <<= INIT
                        s.rdy <<= 1
                        s.wen <<= 0
                        s.incr <<= 0
                        s.inner_incr <<= 0
                        s.section_incr <<= 0
                    else:
                        if s.skip_cnt >= skip_n:
                            s.wen <<= 1
                            if (s.incr >= repeat_len-1):
                                if (s.repeat_count >= (repeat_x - 1)):
                                    s.section_incr <<= s.section_incr + \
                                        repeat_len
                                    s.repeat_count <<= 0
                                else:
                                    s.repeat_count <<= s.repeat_count + 1
                                s.incr <<= 0
                                s.inner_incr <<= 0
                            else:
                                if (s.inner_incr + 1 >= addr_b_offset):
                                    s.inner_incr <<= 0
                                    if (s.w_urn_sel >= sel_count - stride):
                                        s.incr <<= s.incr + addr_b_offset
                                else:
                                    s.inner_incr <<= s.inner_incr + 1
                        else:
                            s.wen <<= 0
                    if (s.skip_cnt >= skip_n):
                        s.skip_cnt <<= 0
                    else:
                        s.skip_cnt <<= s.skip_cnt + 1
        utils.tie_off_clk_reset(s)


class SM_LoadBufsEMIF(Component):
    # Load on-chip buffers from the EMIF
    # Inputs:
    # -  start
    # -  emif_readdatavalid
    # -  emif_waitrequest
    # -  emif_readdata
    # Outputs:
    # -  buf_address
    # -  buf_writedata
    # -  emif_address
    # -  emif_writedata
    # -  emif_write
    # -  emif_read
    # -  rdy
    # -  wen_<n>
    # Starting at startaddr, write write_count values from the EMIF into each
    # buffer_count buffers
    def construct(s, total_buffer_count, write_count, addr_width, datawidth,
                  emif_addr_width, emif_data_width, startaddr=0,
                  ibuffer_count=-1):
        """ Constructor for LoadBufsEMIF

         :param ibuffer_count: Number of buffers to write into
         :type  ibuffer_count: int
         :param write_count: Number of values per buffer
         :type  write_count: int
         :param addr_width: Width of address of on-chip buffers
         :type  addr_width: int
         :param datawidth: Width of on-chip buffer
         :type  datawidth: int
         :param emif_addr_width: Width of address of EMIF interface
         :type  emif_addr_width: int
         :param emif_data_width: Width of EMIF data
         :type  emif_data_width: int
         :param startaddr: Start address of EMIF data
         :type  startaddr: int
        """
        if ibuffer_count < 0:
            ibuffer_count = total_buffer_count
        assert(emif_addr_width > 0)
        assert(emif_data_width > 0)
        s.start = InPort(1)
        s.buf_count = Wire(max(math.ceil(math.log(ibuffer_count, 2)), 1))
        s.buf_address = OutPort(addr_width)
        s.buf_writedata = OutPort(datawidth)
        s.buf_wen = Wire(1)
        utils.add_n_outputs(s, total_buffer_count, 1, "wen_")

        s.emif_address = OutPort(emif_addr_width)
        s.emif_address_w = Wire(max(math.ceil(math.log(startaddr +
                                                       write_count *
                                                       ibuffer_count, 2)),
                                    emif_addr_width))
        s.emif_address //= s.emif_address_w[0:emif_addr_width]
        s.emif_write = OutPort(1)
        s.emif_write //= 0
        s.emif_writedata = OutPort(emif_data_width)
        assert(emif_data_width >= datawidth)
        s.emif_writedata[0:datawidth] //= 0
        s.emif_read = OutPort(1)
        s.emif_readdatavalid = InPort(1)
        s.emif_waitrequest = InPort(1)
        s.emif_readdata = InPort(emif_data_width)

        s.rdy = OutPort(1)
        s.state = Wire(2)

        for wb in range(total_buffer_count):
            if (wb >= ibuffer_count):
                out_wen = getattr(s, "wen_{}".format(wb))
                out_wen //= 0
            else:
                assert(wb < 2 ** math.ceil(math.log(ibuffer_count, 2)))
                new_wen = SM_BufferWen(math.ceil(math.log(ibuffer_count, 2)),
                                       wb)
                setattr(s, "buf_wen{}".format(wb), new_wen)
                if (ibuffer_count > 1):
                    new_wen.buffer_count //= s.buf_count
                new_wen.we_in //= s.buf_wen
                out_wen = getattr(s, "wen_{}".format(wb))
                out_wen //= new_wen.we

        INIT, LOAD = 0, 1
        max_wc = write_count-1
        max_ib = ibuffer_count-1

        @update_ff
        def upblk_set_wen_ff():
            if s.reset:
                s.buf_count <<= 0
                s.state <<= INIT
                s.buf_address <<= 0
                s.emif_address_w <<= startaddr
                s.emif_read <<= 0
                s.rdy <<= 1
                s.buf_wen <<= 0
            else:
                if (s.state == INIT):
                    s.buf_wen <<= 0
                    if (s.start):
                        s.state <<= LOAD
                        s.rdy <<= 0
                        s.emif_address_w <<= startaddr
                        s.emif_read <<= 1
                        s.buf_address <<= 0
                        s.buf_count <<= 0
                    else:
                        s.rdy <<= 1
                elif (s.state == LOAD):
                    if (s.emif_waitrequest == 0):
                        if (s.emif_address_w < (startaddr + write_count *
                                                ibuffer_count-1)):
                            s.emif_address_w <<= s.emif_address_w + 1
                        else:
                            s.emif_address_w <<= startaddr
                            s.emif_read <<= 0
                    if (s.emif_readdatavalid == 1):
                        s.buf_writedata <<= s.emif_readdata[0:datawidth]
                        s.buf_wen <<= 1
                    else:
                        s.buf_wen <<= 0

                    if (s.buf_wen):
                        if (s.buf_address == max_wc) & \
                           (s.buf_count == max_ib):
                            s.state <<= INIT
                        elif (s.buf_address == max_wc) & \
                             (s.buf_count < max_ib):
                            s.buf_count <<= s.buf_count + 1
                            s.buf_address <<= 0
                        elif (s.buf_address < max_wc):
                            s.buf_address <<= s.buf_address + 1

        utils.tie_off_clk_reset(s)


class SM_WriteOffChipEMIF(Component):
    # Load on-chip buffers from the EMIF
    # Inputs:
    # -  start
    # -  datain_<n>
    # -  emif_waitrequest
    # Outputs:
    # -  address
    # -  bufdata
    # -  emif_address_w
    # -  emif_writedata
    # -  emif_write
    # -  emif_read
    # -  rdy
    # -  wen_<n>
    # Starting at startaddr, write write_count values from the EMIF into each
    # buffer_count buffers
    def construct(s, obuffer_count, write_count, addr_width, datawidth,
                  emif_addr_width, emif_data_width, startaddr=0,
                  total_buffer_count=-1, start_buffer=-1):
        """ Constructor for WriteOffChipEMIF
         :param obuffer_count: Number of buffers to write from
         :type  obuffer_count: int
         :param write_count: Number of values per buffer
         :type  write_count: int
         :param addr_width: Width of address of on-chip buffers
         :type  addr_width: int
         :param datawidth: Width of on-chip buffer
         :type  datawidth: int
         :param emif_addr_width: Width of address of EMIF interface
         :type  emif_addr_width: int
         :param emif_data_width: Width of EMIF data
         :type  emif_data_width: int
         :param startaddr: Start address of EMIF data
         :type  startaddr: int
        """
        if (total_buffer_count < 0):
            total_buffer_count = obuffer_count
        assert (addr_width > 0)
        assert (emif_addr_width >= addr_width)
        assert (datawidth > 0)
        assert (emif_data_width >= datawidth)
        s.start = InPort(1)
        s.buf_count = Wire(max(math.ceil(math.log(total_buffer_count, 2)), 1))
        s.buf_idx = OutPort(max(math.ceil(math.log(total_buffer_count, 2)), 1))
        s.address = OutPort(addr_width)
        s.bufdata = Wire(datawidth)

        s.emif_address = OutPort(emif_addr_width)
        s.emif_address_w = Wire(max(math.ceil(math.log(startaddr +
                                                       write_count *
                                                       obuffer_count, 2)),
                                    emif_addr_width))
        s.emif_address //= s.emif_address_w[0:emif_addr_width]
        s.emif_write = OutPort(1)
        s.emif_writedata = OutPort(emif_data_width)
        s.emif_writedata[0:datawidth] //= s.bufdata
        s.emif_read = OutPort(1)
        s.emif_read //= 0
        s.emif_waitrequest = InPort(1)
        s.rdy = OutPort(1)
        s.state = Wire(2)
        utils.add_n_inputs(s, total_buffer_count, datawidth, "datain_")

        newmux = module_helper_classes.MUXN(datawidth, total_buffer_count)
        setattr(s, "data_mux", newmux)
        for wb in range(total_buffer_count):
            currin = getattr(newmux, "in{}".format(wb))
            currin //= getattr(s, "datain_{}".format(wb))
        newmux.sel //= s.buf_idx
        s.bufdata //= newmux.out

        INIT, LOAD = 0, 1

        if (start_buffer < 0):
            start_buffer = total_buffer_count - obuffer_count

        @update_ff
        def upblk_set_wen_ff():
            if s.reset:
                s.buf_count <<= 0
                s.buf_idx <<= start_buffer
                s.state <<= INIT
                s.address <<= 0
                s.emif_address_w <<= startaddr
                s.emif_write <<= 0
                s.rdy <<= 1
            else:
                if (s.state == INIT):
                    if (s.start):
                        s.state <<= LOAD
                        s.rdy <<= 0
                        s.emif_write <<= 1
                    else:
                        s.rdy <<= 1
                elif (s.state == LOAD):
                    if (s.emif_waitrequest == 0):
                        if (s.address == (write_count-1)):
                            s.address <<= 0
                            if (s.buf_count == obuffer_count - 1):
                                s.state <<= INIT
                                s.buf_count <<= 0
                                s.buf_idx <<= start_buffer
                                s.emif_write <<= 0
                            else:
                                if (s.buf_idx == total_buffer_count - 1):
                                    s.buf_idx <<= 0
                                else:
                                    s.buf_idx <<= s.buf_idx + 1
                                s.buf_count <<= s.buf_count + 1
                                s.emif_address_w <<= s.emif_address_w + 1
                        else:
                            s.address <<= s.address + 1
                            s.emif_address_w <<= s.emif_address_w + 1
        utils.tie_off_clk_reset(s)


class StateMachineEMIFSeparate(Component):
    def construct(s, mlb_spec={}, wb_spec={}, ib_spec={}, ob_spec={},
                  emif_spec={}, proj_spec={}, w_address=0, i_address=0,
                  o_address=0, ws=True, ibuf_start=0, load_inputs=True,
                  load_outputs=True, num_layers=1, curr_layer=0,
                  total_num_buffers=-1, pingpong_w=False):
        """ Constructor for Datapath

         :param mlb_spec: Contains information about ML block used
         :type mlb_spec: dict
         :param wb_spec: Contains information about weight buffers used
         :type wb_spec: dict
         :param ib_spec: Contains information about input buffers used
         :type ib_spec: dict
         :param ob_spec: Contains information about output buffers used
         :type ob_spec: dict
         :param emif_spec: Contains information about the emif
         :type emif_spec: dict
         :param proj_spec: Contains projection vectors
         :type proj_spec: dict
         :param w_address: Address of weights in EMIF
         :type w_address: int
         :param i_address: Address of inputs in EMIF
         :type i_address: int
         :param o_address: Address of outputs in EMIF
         :type o_address: int
        """
        printi(il, "{:=^60}".format("> Constructing Statemachine with MLB " +
                                    "block " +
                                    str(mlb_spec.get('block_name', "unnamed") +
                                        " <")))
        buffer_specs = {'W': wb_spec, 'I': ib_spec, 'O': ob_spec}
        inner_proj = proj_spec['inner_projection']
        outer_proj = proj_spec['outer_projection']

        # Make sure that projection makes sense
        if not ws:
            assert ("PRELOAD" not in proj_spec["inner_projection"])
            assert ("PRELOAD" not in proj_spec["outer_projection"])

        # Get port names
        addrw_ports = list(utils.get_ports_of_type(buffer_specs['W'],
                                                   'ADDRESS', ["in"]))
        dataw_ports = list(utils.get_ports_of_type(buffer_specs['W'], 'DATA',
                                                   ["out"]))
        addri_ports = list(utils.get_ports_of_type(buffer_specs['I'],
                                                   'ADDRESS', ["in"]))
        datai_ports = list(utils.get_ports_of_type(buffer_specs['I'], 'DATA',
                                                   ["out"]))
        addro_ports = list(utils.get_ports_of_type(buffer_specs['O'],
                                                   'ADDRESS', ["in"]))

        # Do some calculations...
        temp_proj = proj_spec.get("temporal_projection", {})
        outer_proj = proj_spec.get("outer_projection", {})
        inner_proj = proj_spec.get("inner_projection", {})
        buffer_counts = utils.get_buffer_counts([proj_spec], ib_spec, ob_spec,
                                                wb_spec)
        bank_count = 2 if pingpong_w else 1

        # SM to stream inputs into MLB and read outputs into buffer
        obuf_len = 2 ** addro_ports[0]["width"]
        unt = temp_proj.get("RY", 1) * temp_proj.get("C", 1)
        urw = inner_proj.get("RX", 1) * outer_proj.get("RX", 1)
        uet = temp_proj.get("E", 1)
        ugt = temp_proj.get("G", 1)
        ubt = temp_proj.get("B", obuf_len) * temp_proj.get("PX", 1) * \
            temp_proj.get("PY", 1)
        ubx = temp_proj.get("PX", {})
        s.ugt_cnt = Wire(max(int(math.log(ubt * ubt + 1, 2)),
                             addrw_ports[0]["width"]) +
                         addri_ports[0]["width"] + 1)
        s.uet_cnt = Wire(max(int(math.log(ubt * ubt + 1, 2)),
                             addrw_ports[0]["width"]) +
                         addri_ports[0]["width"] + 1)
        stridex = proj_spec.get("stride", {}).get("x", 1)
        stridey = proj_spec.get("stride", {}).get("y", 1)
        mux_size = 1
        if (outer_proj.get('RY', 1) *
                inner_proj.get('RY', 1)) > 1:
            mux_size = outer_proj.get('PY', 1) * \
                   inner_proj.get('PY', 1) * \
                   outer_proj.get('RY', 1) * \
                   inner_proj.get('RY', 1)
        if (ws):
            if (unt > 1):
                input_count = 1
                output_count = 1
                weight_count = 1
                # If we accumulate inside the MLBs,
                # then there is no weight reuse since they
                # need to be reloaded. So effectively
                # ubt => 1
                uet = uet * unt * ubt
                ubt = 1
                unt = 1
            else:
                input_count = ubt * unt
                output_count = ubt * unt
                weight_count = ubt * unt
            repeat_xi = 1
            repeat_xo = 1
            repeat_xw = 1
            s.stream_outputs = SM_IterateThruAddresses((output_count -
                                                        (urw - 1)) / (stridex)
                                                       + (stridex-1),
                                                       addri_ports[0]["width"],
                                                       start_wait=urw + 1,
                                                       debug_name="out",
                                                       skip_n=stridex-1,
                                                       skip_after=1)
        else:
            input_count = ubt * ugt * (unt)
            output_count = uet * ubt * ugt
            weight_count = uet * ugt * (unt)
            repeat_xi = uet
            repeat_xo = 1
            repeat_xw = ubt
            s.stream_outputs = SM_IterateThruAddresses(output_count + 1,
                                                       addri_ports[0]["width"],
                                                       skip_n=(unt-1),
                                                       start_wait=1,
                                                       repeat_x=repeat_xo,
                                                       debug_name="out")
        outer_tile_repeat_x = 1
        num_outer_tiles = 1
        inner_tile_size = 1
        outer_tile_size = 1
        inner_tile_repeat_x = 1
        if ("PRELOAD" in proj_spec["inner_projection"]):
            inner_tile_repeat_x = inner_proj["B"] * inner_proj["PY"] * \
                inner_proj["PX"]
            inner_tile_size = inner_proj["RY"] * inner_proj["C"] * \
                inner_proj["RX"] * inner_proj["E"]
            outer_tile_size = inner_proj["G"] * \
                inner_proj["RY"] * inner_proj["C"] * inner_proj["RX"] * \
                inner_proj["E"]
        if ("PRELOAD" in proj_spec["outer_projection"]):
            outer_tile_repeat_x = outer_proj["B"] * outer_proj["PY"] * \
                outer_proj["PX"]
            outer_tile_size = outer_proj["RY"] * outer_proj["C"] *\
                outer_proj["RX"] * \
                outer_proj["E"] * outer_tile_size
            num_outer_tiles = outer_proj["G"]
        total_weight_count = utils.get_weight_buffer_len(proj_spec)

        # Add top level ports
        if (bank_count > 1):
            bank_sel_w = utils.AddOutPort(
                s, math.ceil(math.log(bank_count, 2)), "bank_sel")
            bank_sel_w //= 0
        s.sel = InPort(math.ceil(math.log(max(num_layers, 2), 2)))
        s.sm_start = InPort(1)

        s.load_wbufs_emif = SM_LoadBufsEMIF(
            buffer_counts['W'][0] * bank_count,
            min(2 ** addrw_ports[0]["width"], total_weight_count),
            addrw_ports[0]["width"], dataw_ports[0]["width"],
            utils.get_sum_datatype_width(emif_spec, "AVALON_ADDRESS", "in"),
            utils.get_sum_datatype_width(emif_spec, "AVALON_WRITEDATA", "in"),
            w_address)
        utils.connect_out_to_top(s, s.load_wbufs_emif.buf_writedata,
                                 "wbuf_writedata")
        connected_ins = utils.connect_inst_ports_by_name(s, r"weight_wen",
                                                         s.load_wbufs_emif,
                                                         r"wen", parent_in=0)

        total_input_count = utils.get_input_buffer_len(proj_spec)
        s.load_ibufs_emif = SM_LoadBufsEMIF(
            buffer_counts['I'][0] + buffer_counts['O'][0],
            min(2 ** addri_ports[0]["width"], total_input_count),
            addri_ports[0]["width"], datai_ports[0]["width"],
            utils.get_sum_datatype_width(emif_spec, "AVALON_ADDRESS", "in"),
            utils.get_sum_datatype_width(emif_spec, "AVALON_WRITEDATA", "in"),
            i_address, buffer_counts['I'][0])
        utils.connect_out_to_top(s, s.load_ibufs_emif.buf_writedata,
                                 "ibuf_writedata")
        connected_ins += utils.connect_inst_ports_by_name(s, r"input_wen",
                                                          s.load_ibufs_emif,
                                                          r"wen", parent_in=0)
        # Instantiate SM to preload weights into MLBs
        s.preload_weights0 = SM_PreloadMLB(
            addr_width=addrw_ports[0]["width"],
            num_outer_tiles=num_outer_tiles,
            outer_tile_size=outer_tile_size,
            inner_tile_size=inner_tile_size,
            outer_tile_repeat_x=outer_tile_repeat_x,
            inner_tile_repeat_x=inner_tile_repeat_x)
        s.start_addr = Wire(addrw_ports[0]["width"])
        s.plw_start = Wire(1)
        s.preload_weights0.start_address //= s.start_addr
        s.preload_weights0.start //= s.plw_start
        if (bank_count > 1):
            s.preload_weights1 = SM_PreloadMLB(
                addr_width=addrw_ports[0]["width"],
                num_outer_tiles=num_outer_tiles,
                outer_tile_size=outer_tile_size,
                inner_tile_size=inner_tile_size,
                outer_tile_repeat_x=outer_tile_repeat_x,
                inner_tile_repeat_x=inner_tile_repeat_x)
            s.preload_weights1.start_address //= s.start_addr
            s.preload_weights1.start //= s.plw_start

        if (ws and (mux_size > 1)):
            os = ubx
        else:
            os = 1
        if ws:
            ms = mux_size
        else:
            ms = 1
        s.stream_inputs = SM_IterateThruAddresses(input_count + 1,
                                                  addri_ports[0]["width"],
                                                  repeat_x=repeat_xi,
                                                  repeat_len=unt * ubt,
                                                  debug_name="in",
                                                  addr_b_offset=os,
                                                  sel_count=ms, stride=stridey)
        s.istart_address_wide = Wire(max(int(math.log(ubt * ubt + 1, 2)),
                                         addrw_ports[0]["width"]) +
                                     addri_ports[0]["width"] + 1)
        s.stream_inputs.start_address //= \
            s.istart_address_wide[0:addri_ports[0]["width"]]
        s.ostart_address_wide = Wire(max(int(math.log(ubt * ubt + 1, 2)),
                                         addrw_ports[0]["width"]) +
                                     addri_ports[0]["width"] + 1)
        s.stream_outputs.start_address //= \
            s.ostart_address_wide[0:addri_ports[0]["width"]]
        s.stream_weights = SM_IterateThruAddresses(weight_count + 1,
                                                   addrw_ports[0]["width"],
                                                   repeat_x=repeat_xw,
                                                   repeat_len=unt,
                                                   debug_name="weight")
        s.stream_weights.start_address //= 0
        utils.connect_out_to_top(s, s.stream_inputs.wen, "stream_inputs_wen")
        utils.connect_out_to_top(s, s.stream_outputs.wen, "stream_outputs_wen")
        if (ws):
            utils.connect_out_to_top(s, s.preload_weights0.wen,
                                     "stream_weights_wen")
        else:
            utils.connect_out_to_top(s, s.stream_weights.wen,
                                     "stream_weights_wen")
        s.pstart_address_wide = Wire(max(int(math.log(ubt * ubt + 1, 2)),
                                         addrw_ports[0]["width"]) +
                                     addri_ports[0]["width"] + 1)
        s.start_addr //= s.pstart_address_wide[0:addrw_ports[0]["width"]]

        # Now read the outputs out to off-chip memory
        datao_ports = list(utils.get_ports_of_type(buffer_specs['O'], 'DATA',
                                                   ["out"]))

        if (total_num_buffers < 1):
            total_num_buffers = buffer_counts['O'][0] + buffer_counts['I'][0]
        total_out_count = utils.get_output_buffer_len(proj_spec)
        s.write_off_emif = SM_WriteOffChipEMIF(
            buffer_counts['O'][0], min(2 ** addro_ports[0]["width"],
                                       total_out_count),
            addro_ports[0]["width"], datao_ports[0]["width"],
            utils.get_sum_datatype_width(emif_spec, "AVALON_ADDRESS", "in"),
            utils.get_sum_datatype_width(emif_spec, "AVALON_WRITEDATA", "in"),
            o_address, total_num_buffers,
            start_buffer=(buffer_counts['I'][0] + ibuf_start) %
            total_num_buffers)

        connected_ins += utils.connect_inst_ports_by_name(s, r"emif_datain",
                                                          s.write_off_emif,
                                                          r"datain")
        s.state = Wire(5)
        s.done = OutPort(1)
        INIT, LOADING_W_BUFFERS, LOADING_I_BUFFERS, LOADING_MLBS, \
            STREAMING_MLBS, WRITE_OFFCHIP, DONE = Bits5(1), Bits5(2), \
            Bits5(3), Bits5(4), Bits5(5), Bits5(6), Bits5(7)
        initial_val = 2 ** addri_ports[0]["width"] - 1
        if (ws):
            initial_val = initial_val + 1

        s.acc_en_top = OutPort(1)
        s.weight_addr_top = OutPort(addrw_ports[0]["width"])
        s.input_addr_top_a = OutPort(addri_ports[0]["width"])
        s.input_addr_top_b = OutPort(addri_ports[0]["width"])
        s.output_addr_top = OutPort(addri_ports[0]["width"])
        s.emif_address = OutPort(utils.get_sum_datatype_width(emif_spec,
                                                              "AVALON_ADDRESS",
                                                              "in"))
        s.emif_read = OutPort(1)
        s.emif_write = OutPort(1)
        s.emif_waitrequest = InPort(1)
        s.emif_readdatavalid = InPort(1)
        s.emif_writedata = OutPort(utils.get_sum_datatype_width(
            emif_spec, "AVALON_WRITEDATA", "in"))
        s.emif_readdata = InPort(utils.get_sum_datatype_width(
            emif_spec, "AVALON_READDATA", "out"))
        s.urn_sel = OutPort(math.ceil(math.log(max(mux_size, 2), 2)))
        s.urn_sel //= s.stream_inputs.urn_sel

        @update_ff
        def connect_weight_address_ff():
            if s.reset:
                s.state <<= INIT
                s.done <<= 0
                s.ugt_cnt <<= 0
                s.uet_cnt <<= 0
                s.istart_address_wide <<= 0
                s.ostart_address_wide <<= initial_val
            else:
                if (s.state == INIT):
                    if (num_layers < 2) | (s.sel == curr_layer):
                        if s.sm_start:
                            s.state <<= LOADING_W_BUFFERS
                elif (s.state == LOADING_W_BUFFERS):
                    # Write weights on chip
                    if (load_inputs):
                        if s.load_wbufs_emif.rdy:
                            s.state <<= LOADING_I_BUFFERS
                    else:
                        if s.load_wbufs_emif.rdy:
                            s.state <<= LOADING_MLBS
                elif (s.state == LOADING_I_BUFFERS):
                    # Write inputs on chip
                    if s.load_ibufs_emif.rdy:
                        s.state <<= LOADING_MLBS
                elif (s.state == LOADING_MLBS):
                    if (ws):
                        # Pre-load weights into the MLBs
                        if s.preload_weights0.rdy:
                            s.state <<= STREAMING_MLBS
                            if (s.uet_cnt < (uet-1)):
                                s.uet_cnt <<= s.uet_cnt + 1
                            else:
                                s.uet_cnt <<= 0
                                s.ugt_cnt <<= s.ugt_cnt + 1
                    else:
                        s.state <<= STREAMING_MLBS
                elif (s.state == STREAMING_MLBS):
                    if s.stream_inputs.rdy:
                        if ws & (s.ugt_cnt < ugt):
                            s.state <<= LOADING_MLBS
                            if (s.uet_cnt == 0):
                                s.istart_address_wide <<= \
                                    s.istart_address_wide + (input_count)
                            s.ostart_address_wide <<= s.ostart_address_wide + \
                                (output_count)
                        else:
                            if (load_outputs):
                                s.state <<= WRITE_OFFCHIP
                                s.ostart_address_wide <<= 0
                            else:
                                s.state <<= DONE
                elif (s.state == WRITE_OFFCHIP):
                    # Write outputs out to EMIF
                    if s.write_off_emif.rdy:
                        s.state <<= DONE
                        s.done <<= 1
                elif s.state == DONE:
                    s.done <<= 1
                if (ws):
                    s.acc_en_top <<= 0
                else:
                    s.acc_en_top <<= (s.state == STREAMING_MLBS) & \
                        ~s.stream_outputs.wen

        @update
        def connect_weight_address():
            if (s.state == LOADING_MLBS):
                s.weight_addr_top @= s.preload_weights0.address
            elif (s.state == STREAMING_MLBS):
                s.weight_addr_top @= s.stream_weights.address
            else:
                s.weight_addr_top @= s.load_wbufs_emif.buf_address
            if (s.state == STREAMING_MLBS):
                s.input_addr_top_a @= s.stream_inputs.address
                s.input_addr_top_b @= s.stream_inputs.address_b
            else:
                s.input_addr_top_a @= s.load_ibufs_emif.buf_address
                s.input_addr_top_b @= s.load_ibufs_emif.buf_address
            if (s.state == STREAMING_MLBS):
                s.output_addr_top @= s.stream_outputs.address
            else:
                s.output_addr_top @= s.write_off_emif.address
            if (s.state == WRITE_OFFCHIP):
                s.emif_address @= s.write_off_emif.emif_address
                s.emif_read @= s.write_off_emif.emif_read
                s.emif_write @= s.write_off_emif.emif_write
                s.emif_writedata @= s.write_off_emif.emif_writedata
            elif (s.state == LOADING_W_BUFFERS):
                s.emif_address @= s.load_wbufs_emif.emif_address
                s.emif_read @= s.load_wbufs_emif.emif_read
                s.emif_write @= s.load_wbufs_emif.emif_write
                s.emif_writedata @= s.load_wbufs_emif.emif_writedata
            else:
                s.emif_address @= s.load_ibufs_emif.emif_address
                s.emif_read @= s.load_ibufs_emif.emif_read
                s.emif_write @= s.load_ibufs_emif.emif_write
                s.emif_writedata @= s.load_ibufs_emif.emif_writedata
            s.load_wbufs_emif.start @= (s.state == INIT) & s.sm_start
            s.load_wbufs_emif.emif_readdatavalid @= s.emif_readdatavalid
            s.load_wbufs_emif.emif_waitrequest @= s.emif_waitrequest
            s.load_wbufs_emif.emif_readdata @= s.emif_readdata
            if (load_inputs):
                s.load_ibufs_emif.start @= (s.state == LOADING_W_BUFFERS) & \
                    s.load_wbufs_emif.rdy
            else:
                s.load_ibufs_emif.start @= 0
            s.load_ibufs_emif.emif_readdatavalid @= s.emif_readdatavalid
            s.load_ibufs_emif.emif_waitrequest @= s.emif_waitrequest
            s.load_ibufs_emif.emif_readdata @= s.emif_readdata
            if (ws):
                if (load_inputs):
                    s.plw_start @= \
                        ((s.state == LOADING_I_BUFFERS) &
                         s.load_ibufs_emif.rdy) | \
                        ((s.state == STREAMING_MLBS) &
                         s.stream_inputs.rdy & (s.ugt_cnt < ugt))
                else:
                    s.plw_start @= \
                        ((s.state == LOADING_W_BUFFERS) &
                         s.load_wbufs_emif.rdy) | \
                        ((s.state == STREAMING_MLBS) &
                         s.stream_inputs.rdy & (s.ugt_cnt < ugt))
                s.pstart_address_wide @= (s.ugt_cnt * uet + s.uet_cnt) * \
                    outer_tile_size
                s.stream_inputs.start @= (s.state == LOADING_MLBS) & \
                    s.preload_weights0.rdy
                s.stream_outputs.start @= (s.state == LOADING_MLBS) & \
                    s.preload_weights0.rdy
            else:
                s.plw_start @= 0
                s.stream_inputs.start @= (s.state == LOADING_MLBS) & \
                    s.load_ibufs_emif.rdy
                s.stream_weights.start @= (s.state == LOADING_MLBS) & \
                    s.load_ibufs_emif.rdy
                s.stream_outputs.start @= (s.state == LOADING_MLBS) & \
                    s.load_ibufs_emif.rdy
            s.write_off_emif.emif_waitrequest @= s.emif_waitrequest
            if (load_outputs):
                s.write_off_emif.start @= (s.state == STREAMING_MLBS) & \
                    s.stream_inputs.rdy & ((ws == 0) | (s.ugt_cnt == ugt))
            else:
                s.write_off_emif.start @= 0


class MultipleLayerSystem(Component):
    def construct(s, mlb_spec={}, wb_spec={}, ib_spec={}, ob_spec={},
                  emif_spec={}, proj_specs=[], w_address=[0], i_address=[0],
                  o_address=[0], ws=True, fast_gen=False,
                  write_between_layers=False, pingpong_w=False):
        """ Constructor
        """
        printi(il, "{:=^60}".format("> Constructing Accelerator, MLB block " +
                                    str(mlb_spec.get('block_name', "unnamed") +
                                        " <")))
        if "inner_projection" in proj_specs:
            proj_specs = [proj_specs]
            w_address = [w_address]
            i_address = [i_address]
            o_address = [o_address]

        # Do some calculations...
        buffer_counts = utils.get_buffer_counts(proj_specs, ib_spec, ob_spec,
                                                wb_spec)
        total_num_buffers = max(sum(x) for x in zip(buffer_counts['I'],
                                                    buffer_counts['O']))
        buffer_start_idxs = [sum(buffer_counts['I'][0:i+1]) %
                             total_num_buffers
                             for i in range(len(buffer_counts['I']) - 1)]
        ibuffer_start_idxs = [0] + buffer_start_idxs

        # Add top level ports
        s.synth_keep = OutPort(1)
        s.sel = InPort(math.ceil(math.log(max(len(proj_specs), 2), 2)))
        s.sm_start = InPort(1)

        s.datapath = module_classes.Datapath(mlb_spec, wb_spec, ib_spec,
                                             ob_spec, proj_specs,
                                             fast_gen=fast_gen,
                                             pingpong_w=pingpong_w)

        if isinstance(fast_gen, bool):
            emif_fastgen = fast_gen
            inner_emif_fastgen = fast_gen
        else:
            emif_fastgen = (emif_spec['block_name'] not in fast_gen)
            inner_emif_fastgen = ('emif_inner' not in fast_gen)

        s.emif_inst = module_classes.HWB_Sim(emif_spec, {}, sim=True,
                                             fast_gen=emif_fastgen,
                                             inner_fast_gen=inner_emif_fastgen)

        statemachines = []
        for i in range(len(proj_specs)):
            # Instantiate and connect a statemachine
            if (i > 0):
                newname = proj_specs[i].get("name", i)
            else:
                newname = ""
            statemachine = StateMachineEMIFSeparate(
                mlb_spec, wb_spec, ib_spec, ob_spec, emif_spec, proj_specs[i],
                w_address[i], i_address[i], o_address[i], ws,
                ibuf_start=ibuffer_start_idxs[i],
                load_inputs=((i == 0) or write_between_layers),
                load_outputs=((i == len(proj_specs) - 1) or
                              write_between_layers),
                num_layers=len(proj_specs), curr_layer=i,
                total_num_buffers=total_num_buffers,
                pingpong_w=pingpong_w)
            setattr(s, "statemachine" + newname, statemachine)
            statemachines += [statemachine]

        # Connect sub modules
        connected_ins = [s.datapath.weight_modules_portaaddr_top,
                         s.datapath.weight_datain,
                         s.datapath.input_datain, s.emif_inst.address,
                         s.emif_inst.read,
                         s.emif_inst.write,
                         s.emif_inst.writedata,
                         s.emif_inst.readdatavalid,
                         s.emif_inst.readdata,
                         s.emif_inst.waitrequest,
                         s.datapath.sel]

        # Connect statemachines to EMIF and top level
        for statemachine in statemachines:
            statemachine.emif_waitrequest //= s.emif_inst.waitrequest
            statemachine.emif_readdatavalid //= s.emif_inst.readdatavalid
            statemachine.emif_readdata //= s.emif_inst.readdata
            statemachine.sm_start //= s.sm_start
            statemachine.sel //= s.sel
            connected_ins += utils.connect_ports_by_name(
                s.datapath, r"portadataout_out_(\d+)", statemachine,
                r"emif_datain_(\d+)")
            connected_ins += [statemachine.emif_waitrequest,
                              statemachine.emif_readdatavalid,
                              statemachine.emif_readdata,
                              statemachine.done,
                              statemachine.sm_start]

        s.synth_keep //= s.emif_inst.readdata[0]
        s.datapath.sel //= s.sel

        acc_en_port_name = list(utils.get_ports_of_type(mlb_spec, 'ACC_EN',
                                                        ["in"]))
        acc_en_port = getattr(s.datapath, "mlb_modules_" +
                              acc_en_port_name[0]["name"] + "_top")
        w_en_port_name = list(utils.get_ports_of_type(mlb_spec, 'W_EN',
                                                      ["in"]))
        if (len(w_en_port_name) > 0):
            w_en_port = getattr(s.datapath, "mlb_modules_" +
                                w_en_port_name[0]["name"] + "_top")
            connected_ins += [w_en_port]
        i_en_port_name = list(utils.get_ports_of_type(mlb_spec, 'I_EN',
                                                      ["in"]))
        i_en_port = getattr(s.datapath, "mlb_modules_" +
                            i_en_port_name[0]["name"] + "_top")
        connected_ins += [i_en_port, acc_en_port]

        connected_ins += utils.mux_ports_by_name(
            s, statemachines, "urn_sel", s.datapath,
            "input_act_modules_addr_sel_top", insel=s.sel)
        connected_ins += utils.mux_ports_by_name(
            s, statemachines, "urn_sel", s.datapath,
            "input_interconnect_urn_sel_top", insel=s.sel)
        connected_ins += utils.mux_ports_by_name(
            s, statemachines, "input_addr_top_a", s.datapath,
            "input_act_modules_portaaddr_top", insel=s.sel)
        connected_ins += utils.mux_ports_by_name(
            s, statemachines, "output_addr_top", s.datapath,
            "input_act_modules_portaaddr_o_top", insel=s.sel)
        connected_ins += utils.mux_ports_by_name(
            s, statemachines, "input_addr_top_b", s.datapath,
            "input_act_modules_portaaddr_b_top", insel=s.sel)
        connected_ins += utils.mux_ports_by_name(
            s, statemachines, "weight_addr_top", s.datapath,
            "weight_modules_portaaddr_top", insel=s.sel)
        connected_ins += utils.mux_ports_by_name(
            s, statemachines, "acc_en_top", s.datapath,
            "mlb_modules_" + acc_en_port_name[0]["name"] + "_top", insel=s.sel)
        if (len(w_en_port_name) > 0):
            connected_ins += utils.mux_ports_by_name(
                s, statemachines, "stream_weights_wen", s.datapath,
                "mlb_modules_" + w_en_port_name[0]["name"] + "_top",
                insel=s.sel)
        connected_ins += utils.mux_ports_by_name(
            s, statemachines, "stream_inputs_wen", s.datapath,
            "mlb_modules_" + i_en_port_name[0]["name"] + "_top", insel=s.sel)
        connected_ins += utils.mux_ports_by_name(
            s, statemachines, "wbuf_writedata", s.datapath,
            "weight_datain", insel=s.sel)
        connected_ins += utils.mux_ports_by_name(
            s, statemachines, "ibuf_writedata", s.datapath,
            "input_datain", insel=s.sel)

        connected_ins += utils.mux_ports_by_name(s, statemachines,
                                                 "emif_address", s.emif_inst,
                                                 "address", insel=s.sel)
        connected_ins += utils.mux_ports_by_name(s, statemachines,
                                                 "emif_writedata", s.emif_inst,
                                                 "writedata", insel=s.sel)
        connected_ins += utils.mux_ports_by_name(s, statemachines, "emif_read",
                                                 s.emif_inst, "read",
                                                 insel=s.sel)
        connected_ins += utils.mux_ports_by_name(s, statemachines,
                                                 "emif_write", s.emif_inst,
                                                 "write", insel=s.sel)

        connected_ins += utils.mux_ports_by_name(
            s, statemachines, r"weight_wen_(\d+)", s.datapath,
            r"weight_modules_portawe_(\d+)_top", insel=s.sel)

        connected_ins += utils.mux_ports_by_name(
            s, statemachines, r"input_wen_(\d+)", s.datapath,
            r"input_act_modules_portawe_(\d+)_top", insel=s.sel)
        connected_ins += utils.mux_ports_by_name(
            s, statemachines, r"stream_outputs_wen", s.datapath,
            r"input_act_modules_portawe_(\d+)_out_top", insel=s.sel)
        if (pingpong_w):
            connected_ins += utils.mux_ports_by_name(
                s, statemachines, r"bank_sel", s.datapath,
                r"bank_sel", insel=s.sel)
        s.done = OutPort(1)
        utils.mux_inst_ports_by_name(s, "done", statemachines,
                                     "done", insel=s.sel, sim=False)

        # Connect all inputs not otherwise connected to top
        for inst in statemachines + [s.datapath, s.emif_inst]:
            for port in (inst.get_input_value_ports()):
                if (port._dsl.my_name not in s.__dict__.keys()) and \
                   (port not in connected_ins):
                    utils.connect_in_to_top(s, port, port._dsl.my_name)
            for port in (inst.get_output_value_ports()):
                if (port._dsl.my_name not in s.__dict__.keys()) and \
                   (port not in connected_ins):
                    utils.connect_out_to_top(s, port, port._dsl.my_name)
