// zeta_engine: the descriptor-driven overlay engine (M8).
//
// Consumes a kernel program as a stream of 64-bit words (4-word descriptors
// + payloads, see host/zetafpga/kernel/isa.py for the ISA contract) and
// executes it against the euler_maclaurin_top zeta core:
//
//   WRITE_TABLE  -> lnn/bern RAMs (the host-generated ln(n) and Bernoulli
//                   tables; runtime data, never bitstream ROMs)
//   COMPUTE_EM   -> one zeta(s) evaluation appended to the result buffer
//   READBACK     -> header + results streamed out, buffer and flags cleared
//   BARRIER/NOP/SET_FORMAT -> in-order fence / no-ops
//   unknown      -> sticky err flag, descriptor skipped (no payload allowed)
//
// v0 transport is a plain word stream (the cocotb/DMA boundary); PCIe DMA
// framing lands in Phase 2 behind the same two streams.
module zeta_engine #(
    parameter int unsigned LIMBS = 1,
    parameter int unsigned EXPW = 20,
    parameter int unsigned LIMBW = 64,
    parameter int unsigned TW = 32,
    parameter int unsigned TILE_LATENCY = 2,
    parameter int unsigned PHW = 96,
    parameter int unsigned FG = 88,
    parameter int unsigned CONSTW = 96,
    parameter int unsigned TERMS = 22,
    parameter int unsigned CTERMS = 11,
    parameter int unsigned SEGW = 10,
    parameter string EXP_ROM = "expln_w64_exp.mem",
    parameter string CEXP_ROM = "cexp_w64.mem",
    // theta / RS-correction constants (COMPUTE_Z; theta_w*.json, rsck_w*.json)
    parameter int unsigned KTERMS = 12,
    parameter int unsigned LOG_TERMS = 32,
    parameter string THETA_FX_ROM = "theta_w64_fx.mem",
    parameter string THETA_MPF_ROM = "theta_w64_mpf.mem",
    parameter string EXP_ROM2 = "expln_w128_exp.mem",
    parameter string LN_ROM2 = "expln_w128_ln.mem",
    parameter int unsigned NC = 37,
    parameter int unsigned KMAX = 4,
    parameter string RSCK_ROM = "rsck_w64.mem",
    parameter int unsigned LNN_DEPTH = 1024,
    parameter int unsigned BERN_DEPTH = 256,
    parameter int unsigned RES_DEPTH = 128,
    parameter int unsigned RS_DEPTH = 1024,
    parameter int unsigned RS_LANES = 1,  // power of 2; RS table is banked by entry index
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned MPW = WIDTH + EXPW + 3,
    localparam int unsigned BW = PHW + 32,
    localparam int unsigned LNW = FG + 8,
    localparam int unsigned ENTRYW = LNW + BW + 8,
    localparam int unsigned RSW = BW + 8 + MPW,         // pipelined-RS entry
    localparam int unsigned K = (MPW + 63) / 64,        // words per MPF
    localparam int unsigned EWRD = (ENTRYW + 63) / 64,  // words per lnn entry
    localparam int unsigned RS_EWRD = (RSW + 63) / 64,  // words per RS entry
    localparam int unsigned CFGW = 1 + 5 * K            // COMPUTE_EM payload words
) (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        in_valid,
    output logic        in_ready,
    input  logic [63:0] in_data,

    output logic        out_valid,
    input  logic        out_ready,
    output logic [63:0] out_data
);

  // ---- opcodes (mirror isa.py) ----------------------------------------------
  localparam logic [7:0] OP_NOP = 8'd0;
  localparam logic [7:0] OP_SET_FORMAT = 8'd1;
  localparam logic [7:0] OP_WRITE_TABLE = 8'd2;
  localparam logic [7:0] OP_COMPUTE_EM = 8'd3;
  localparam logic [7:0] OP_READBACK = 8'd4;
  localparam logic [7:0] OP_BARRIER = 8'd5;
  localparam logic [7:0] OP_COMPUTE_PS = 8'd6;  // power sum only (RS main sum)
  localparam logic [7:0] OP_COMPUTE_RS = 8'd7;  // pipelined RS main sum
  localparam logic [7:0] OP_COMPUTE_Z = 8'd8;      // fully on-chip Z(t), count ignored
  localparam logic [7:0] OP_COMPUTE_ZGRID = 8'd9;  // J = count points from (t0, dt)

  // packed MPF zero (is_zero flag set) for the Z result's imaginary word
  localparam logic [MPW-1:0] ZERO_MPF = MPW'(1) << (WIDTH + EXPW + 1);

  // ---- table / result storage -------------------------------------------------
  logic [ENTRYW-1:0] lnn_ram [0:LNN_DEPTH-1];
  logic [MPW-1:0] bern_ram [0:BERN_DEPTH-1];
  // RS table banked by entry index low bits: entry i -> bank i % RS_LANES
  logic [RSW-1:0] rs_ram [0:RS_LANES-1][0:RS_DEPTH-1];
  logic [2*MPW+1:0] res_ram [0:RES_DEPTH-1];

  // ---- zeta core ---------------------------------------------------------------
  logic em_start, em_ready, em_done, em_ovf, em_unf;
  logic [MPW-1:0] em_zre, em_zim;
  logic em_entry_valid, em_entry_ready, em_bern_valid, em_bern_ready;

  // Payload staging: the padding bits above MPW/ENTRYW inside each 64-bit
  // word group are structurally unused.
  // verilator lint_off UNUSEDSIGNAL
  logic [CFGW*64-1:0] cfg;
  // verilator lint_on UNUSEDSIGNAL
  logic [23:0] n_q;
  logic [11:0] m_q;

  localparam int unsigned OFF_SIGMA = 64;
  localparam int unsigned OFF_TMPF = 64 + K * 64;
  localparam int unsigned OFF_IRE = 64 + 2 * K * 64;
  localparam int unsigned OFF_IIM = 64 + 3 * K * 64;
  localparam int unsigned OFF_IN2 = 64 + 4 * K * 64;

  logic [23:0] ectr;
  logic [11:0] bctr;
  logic [ENTRYW-1:0] entry_cur;

  assign entry_cur = lnn_ram[32'(ectr) <= 32'(n_q) ? 32'(ectr) : 32'(n_q)];

  logic ps_flag;

  euler_maclaurin_top #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW), .TW(TW),
      .TILE_LATENCY(TILE_LATENCY), .PHW(PHW), .FG(FG), .CONSTW(CONSTW),
      .TERMS(TERMS), .CTERMS(CTERMS), .SEGW(SEGW),
      .EXP_ROM(EXP_ROM), .CEXP_ROM(CEXP_ROM)
  ) u_em (
      .clk(clk), .rst_n(rst_n),
      .start_valid(em_start), .start_ready(em_ready), .ps_only(ps_flag),
      .sigma(cfg[OFF_SIGMA+:MPW]), .t_mpf(cfg[OFF_TMPF+:MPW]),
      .t_fx(cfg[63:0]), .n_in(n_q), .m_in(m_q),
      .inv_sm1_re(cfg[OFF_IRE+:MPW]), .inv_sm1_im(cfg[OFF_IIM+:MPW]),
      .inv_np2(cfg[OFF_IN2+:MPW]),
      .entry_valid(em_entry_valid), .entry_ready(em_entry_ready),
      .entry_lnn_fx(entry_cur[LNW-1:0]), .entry_lnn2pi(entry_cur[LNW+:BW+8]),
      .bern_valid(em_bern_valid), .bern_ready(em_bern_ready),
      .bern_data(bern_ram[32'(bctr)]),
      .out_valid(em_done), .z_re(em_zre), .z_im(em_zim),
      .ovf(em_ovf), .unf(em_unf)
  );

  // ---- pipelined RS engine (RS_LANES parallel lanes) ------------------------------
  logic rs_start, rs_ready, rs_done, rs_feed;
  logic [MPW-1:0] rs_sre, rs_sim;
  logic [RS_LANES-1:0] rs_entry_ready;
  logic [23:0] rs_ctr [RS_LANES];
  logic [RS_LANES*(BW+8)-1:0] rs_lnn_flat;
  logic [RS_LANES*MPW-1:0] rs_amp_flat;

  always_comb begin
    for (int unsigned l = 0; l < RS_LANES; l++) begin
      rs_lnn_flat[l*(BW+8)+:BW+8] = rs_ram[l][32'(rs_ctr[l])][BW+7:0];
      rs_amp_flat[l*MPW+:MPW]     = rs_ram[l][32'(rs_ctr[l])][BW+8+:MPW];
    end
  end

  rs_power_sum_tiled #(
      .LANES(RS_LANES),
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW), .TW(TW),
      .TILE_LATENCY(TILE_LATENCY), .PHW(PHW), .FG(FG), .CONSTW(CONSTW),
      .CTERMS(CTERMS), .SEGW(SEGW), .EXP_TERMS(TERMS),
      .CEXP_ROM(CEXP_ROM), .EXP_ROM(EXP_ROM)
  ) u_rs (
      .clk(clk), .rst_n(rst_n),
      .start_valid(rs_start), .start_ready(rs_ready),
      .t_fx(t_cur), .n_in(n_q),
      .entry_valid({RS_LANES{rs_feed}}), .entry_ready(rs_entry_ready),
      .lnn2pi(rs_lnn_flat), .amp(rs_amp_flat),
      .out_valid(rs_done), .s_re(rs_sre), .s_im(rs_sim)
  );

  // ---- on-chip Z unit (COMPUTE_Z / COMPUTE_ZGRID) -----------------------------------
  logic [63:0] t_cur, dt_q;
  logic [23:0] jcnt, jdone;
  logic zp_go, zp_ready, zp_prep, zsum_go, zp_done;
  logic [23:0] zp_n;
  logic [MPW-1:0] zp_z;

  rs_z_unit #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW), .TW(TW),
      .TILE_LATENCY(TILE_LATENCY), .PHW(PHW), .FG(FG), .CONSTW(CONSTW),
      .SEGW(SEGW), .CTERMS(CTERMS), .EXP_TERMS(TERMS),
      .CEXP_ROM(CEXP_ROM), .EXP_ROM(EXP_ROM),
      .KTERMS(KTERMS), .LOG_TERMS(LOG_TERMS),
      .THETA_FX_ROM(THETA_FX_ROM), .THETA_MPF_ROM(THETA_MPF_ROM),
      .EXP_ROM2(EXP_ROM2), .LN_ROM2(LN_ROM2),
      .NC(NC), .KMAX(KMAX), .RSCK_ROM(RSCK_ROM)
  ) u_z (
      .clk(clk), .rst_n(rst_n),
      .in_valid(zp_go), .in_ready(zp_ready), .t_fx(t_cur),
      .prep_valid(zp_prep), .n_out(zp_n),
      .sum_valid(zsum_go), .s_re(rs_sre), .s_im(rs_sim),
      .out_valid(zp_done), .z_o(zp_z)
  );

  // ---- engine FSM ----------------------------------------------------------------
  typedef enum logic [3:0] {
    FETCH, DISPATCH, WT_PAY, CEM_PAY, CEM_RUN, CRS_PAY, CZG_PAY, CZ_PREP,
    CRS_RUN, CZ_POST, RB_HDR, RB_STREAM
  } state_e;
  state_e state;

  logic z_flag;  // current CRS run belongs to a COMPUTE_Z/ZGRID descriptor

  logic [1:0] dcnt;
  // dw0[63:52] is reserved by the ISA.
  // verilator lint_off UNUSEDSIGNAL
  logic [63:0] dw0;
  // verilator lint_on UNUSEDSIGNAL
  logic [7:0] op;
  logic [7:0] table_id;
  logic [23:0] count_f;

  logic [7:0] wcnt;         // payload word index within one item
  logic [23:0] wt_left;     // table entries remaining
  logic [23:0] widx;        // table write index
  // verilator lint_off UNUSEDSIGNAL
  logic [EWRD*64-1:0] asm_q;
  // verilator lint_on UNUSEDSIGNAL

  logic [23:0] rcnt;        // results held
  logic [23:0] rb_eval;
  logic [7:0] rb_word;
  logic ovf_s, unf_s, err_s;

  assign op       = dw0[7:0];
  assign table_id = dw0[15:8];
  assign count_f  = dw0[39:16];

  localparam logic [7:0] TBL_BERN = 8'd1;

  // words per table item, and the item with the in-flight final word merged
  // (bits above ENTRYW are padding).
  logic [31:0] item_words;
  // verilator lint_off UNUSEDSIGNAL
  logic [EWRD*64-1:0] item_full;
  // verilator lint_on UNUSEDSIGNAL

  localparam logic [7:0] TBL_RS = 8'd2;

  assign item_words = (table_id == 8'(TBL_BERN)) ? 32'(K)
                    : (table_id == 8'(TBL_RS)) ? 32'(RS_EWRD) : 32'(EWRD);
  assign item_full  = asm_q | ((EWRD * 64)'(in_data) << (64 * 32'(wcnt)));

  assign in_ready = (state == FETCH) || (state == WT_PAY) || (state == CEM_PAY)
                 || (state == CRS_PAY) || (state == CZG_PAY);
  assign rs_feed  = (state == CRS_RUN);
  assign out_valid = (state == RB_HDR) || (state == RB_STREAM);

  // readback data mux
  logic [2*MPW+1:0] res_cur;
  logic [K*64-1:0] re_pad, im_pad;

  assign res_cur = res_ram[32'(rb_eval)];
  assign re_pad  = (K * 64)'(res_cur[MPW-1:0]);
  assign im_pad  = (K * 64)'(res_cur[2*MPW-1:MPW]);

  always_comb begin
    if (state == RB_HDR) begin
      out_data = {37'b0, err_s, unf_s, ovf_s, rcnt};
    end else if (32'(rb_word) < 32'(K)) begin
      out_data = re_pad[64*rb_word+:64];
    end else if (32'(rb_word) < 2 * 32'(K)) begin
      out_data = im_pad[64*(32'(rb_word)-32'(K))+:64];
    end else begin
      out_data = {62'b0, res_cur[2*MPW+1], res_cur[2*MPW]};  // {unf, ovf}
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= FETCH;
      dcnt  <= 2'd0;
      rcnt  <= 24'd0;
      ovf_s <= 1'b0;
      unf_s <= 1'b0;
      err_s <= 1'b0;
    end else begin
      em_start       <= 1'b0;
      rs_start       <= 1'b0;
      zp_go          <= 1'b0;
      zsum_go        <= 1'b0;
      em_entry_valid <= (state == CEM_RUN);
      em_bern_valid  <= (state == CEM_RUN);

      unique case (state)
        FETCH: begin
          if (in_valid) begin
            if (dcnt == 2'd0) begin
              dw0 <= in_data;
            end
            if (dcnt == 2'd3) begin
              dcnt  <= 2'd0;
              state <= DISPATCH;
            end else begin
              dcnt <= dcnt + 2'd1;
            end
          end
        end

        DISPATCH: begin
          unique case (op)
            OP_NOP, OP_SET_FORMAT, OP_BARRIER: state <= FETCH;
            OP_WRITE_TABLE: begin
              if (count_f == 0) begin
                state <= FETCH;
              end else begin
                wt_left <= count_f;
                widx    <= 24'd0;
                wcnt    <= 8'd0;
                asm_q   <= '0;
                state   <= WT_PAY;
              end
            end
            OP_COMPUTE_EM, OP_COMPUTE_PS: begin
              n_q     <= count_f;
              m_q     <= dw0[51:40];
              ps_flag <= (op == OP_COMPUTE_PS);
              wcnt    <= 8'd0;
              state   <= CEM_PAY;
            end
            OP_COMPUTE_RS, OP_COMPUTE_Z: begin
              n_q    <= count_f;  // ignored for COMPUTE_Z (prep derives N)
              z_flag <= (op == OP_COMPUTE_Z);
              jcnt   <= 24'd1;
              state  <= CRS_PAY;
            end
            OP_COMPUTE_ZGRID: begin
              z_flag <= 1'b1;
              jcnt   <= count_f;
              wcnt   <= 8'd0;
              state  <= CZG_PAY;  // always consumes both payload words
            end
            OP_READBACK: state <= RB_HDR;
            default: begin
              err_s <= 1'b1;  // malformed opcode: skip descriptor
              state <= FETCH;
            end
          endcase
        end

        WT_PAY: begin
          if (in_valid) begin
            if (32'(wcnt) == item_words - 1) begin
              // asm_q holds words 0..wcnt-1; in_data is the final word.
              if (table_id == 8'(TBL_BERN)) begin
                bern_ram[32'(widx)] <= item_full[MPW-1:0];
              end else if (table_id == TBL_RS) begin
                rs_ram[32'(widx)%RS_LANES][32'(widx)/RS_LANES] <= item_full[RSW-1:0];
              end else begin
                lnn_ram[32'(widx)] <= item_full[ENTRYW-1:0];
              end
              widx  <= widx + 24'd1;
              asm_q <= '0;
              wcnt  <= 8'd0;
              if (wt_left == 24'd1) begin
                state <= FETCH;
              end else begin
                wt_left <= wt_left - 24'd1;
              end
            end else begin
              asm_q[64*wcnt+:64] <= in_data;
              wcnt <= wcnt + 8'd1;
            end
          end
        end

        CEM_PAY: begin
          if (in_valid) begin
            cfg[64*wcnt+:64] <= in_data;
            if (32'(wcnt) == 32'(CFGW) - 1) begin
              wcnt     <= 8'd0;
              em_start <= 1'b1;
              ectr     <= 24'd0;
              bctr     <= 12'd0;
              state    <= CEM_RUN;
            end else begin
              wcnt <= wcnt + 8'd1;
            end
          end
        end

        CRS_PAY: begin
          if (in_valid) begin
            t_cur <= in_data;  // t_fx
            if (z_flag) begin
              zp_go <= 1'b1;
              jdone <= 24'd0;
              state <= CZ_PREP;
            end else begin
              rs_start <= 1'b1;
              for (int unsigned l = 0; l < RS_LANES; l++) begin
                rs_ctr[l] <= 24'd0;
              end
              state <= CRS_RUN;
            end
          end
        end

        CZG_PAY: begin
          if (in_valid) begin
            if (wcnt == 8'd0) begin
              t_cur <= in_data;  // t0
              wcnt  <= 8'd1;
            end else begin
              dt_q  <= in_data;
              wcnt  <= 8'd0;
              jdone <= 24'd0;
              if (jcnt == 0) begin
                state <= FETCH;
              end else begin
                zp_go <= 1'b1;
                state <= CZ_PREP;
              end
            end
          end
        end

        CZ_PREP: begin
          if (zp_prep) begin
            n_q      <= zp_n;  // N derived on chip
            rs_start <= 1'b1;
            for (int unsigned l = 0; l < RS_LANES; l++) begin
              rs_ctr[l] <= 24'd0;
            end
            state <= CRS_RUN;
          end
        end

        CRS_RUN: begin
          for (int unsigned l = 0; l < RS_LANES; l++) begin
            if (rs_feed && rs_entry_ready[l]) begin
              rs_ctr[l] <= rs_ctr[l] + 24'd1;
            end
          end
          if (rs_done) begin
            if (z_flag) begin
              zsum_go <= 1'b1;
              state   <= CZ_POST;
            end else begin
              res_ram[32'(rcnt)] <= {2'b00, rs_sim, rs_sre};
              rcnt  <= rcnt + 24'd1;
              state <= FETCH;
            end
          end
        end

        CZ_POST: begin
          if (zp_done) begin
            res_ram[32'(rcnt)] <= {2'b00, ZERO_MPF, zp_z};
            rcnt <= rcnt + 24'd1;
            if (jdone + 24'd1 < jcnt) begin
              jdone <= jdone + 24'd1;
              t_cur <= t_cur + dt_q;
              zp_go <= 1'b1;
              state <= CZ_PREP;
            end else begin
              state <= FETCH;
            end
          end
        end

        CEM_RUN: begin
          if (em_entry_valid && em_entry_ready) begin
            ectr <= ectr + 24'd1;
          end
          if (em_bern_valid && em_bern_ready) begin
            bctr <= bctr + 12'd1;
          end
          if (em_done) begin
            res_ram[32'(rcnt)] <= {em_unf, em_ovf, em_zim, em_zre};
            rcnt  <= rcnt + 24'd1;
            ovf_s <= ovf_s | em_ovf;
            unf_s <= unf_s | em_unf;
            state <= FETCH;
          end
        end

        RB_HDR: begin
          if (out_ready) begin
            if (rcnt == 0) begin
              ovf_s <= 1'b0;
              unf_s <= 1'b0;
              err_s <= 1'b0;
              state <= FETCH;
            end else begin
              rb_eval <= 24'd0;
              rb_word <= 8'd0;
              state   <= RB_STREAM;
            end
          end
        end

        RB_STREAM: begin
          if (out_ready) begin
            if (32'(rb_word) == 2 * 32'(K)) begin
              rb_word <= 8'd0;
              if (rb_eval == rcnt - 24'd1) begin
                rcnt  <= 24'd0;
                ovf_s <= 1'b0;
                unf_s <= 1'b0;
                err_s <= 1'b0;
                state <= FETCH;
              end else begin
                rb_eval <= rb_eval + 24'd1;
              end
            end else begin
              rb_word <= rb_word + 8'd1;
            end
          end
        end

        default: state <= FETCH;
      endcase
    end
  end

  // em_ready/rs_ready are guaranteed by the in-order engine (one eval in flight).
  // verilator lint_off UNUSEDSIGNAL
  logic unused_em_ready;
  assign unused_em_ready = em_ready | rs_ready | zp_ready;
  // verilator lint_on UNUSEDSIGNAL

endmodule
