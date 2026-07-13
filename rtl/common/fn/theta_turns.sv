// theta_turns: Riemann-Siegel theta(t)/2pi mod 1, on chip (M14).
//
//   theta/2pi = (u/2)(ln u - 1) - 1/16 + sum_k (c_k/2pi) t^(1-2k),  u = t/2pi
//
// Internally one limb wider than the target format (W2 = WIDTH+64): the
// main term's magnitude ~t*ln t costs ~log2(t) <= 32 bits of the mantissa,
// so ln u is evaluated by a log_mpf instance at W2 and the term is formed in
// Q(38).(FG2) fixed point where the mod-1 wrap is exact, then truncated to
// the target PHW turns. The tail Horner runs in MPF@W2 with ROM'd c_k/2pi.
//
// inv_t (1/t as MPF@W2) is host-supplied for now (the engine's Newton
// reciprocal is a stretch item). Validity: t >= t_min (theta_w*.json);
// below it the host computes theta itself.
//
// Latency ~ FG2 + K*(mul+add latency) + 10 cycles, one operand in flight.
// Golden model: host/zetafpga/golden/theta.py::theta_turns (bit-exact).
module theta_turns #(
    parameter int unsigned LIMBS = 1,          // target format
    parameter int unsigned EXPW = 20,
    parameter int unsigned LIMBW = 64,
    parameter int unsigned TW = 32,
    parameter int unsigned TILE_LATENCY = 2,
    parameter int unsigned PHW = 96,
    parameter int unsigned KTERMS = 12,        // tail terms (theta_w*.json)
    parameter int unsigned LOG_TERMS = 32,     // exp/log terms at W2 (expln_w*.json)
    parameter string THETA_FX_ROM = "theta_w64_fx.mem",
    parameter string THETA_MPF_ROM = "theta_w64_mpf.mem",
    parameter string EXP_ROM2 = "expln_w128_exp.mem",  // constants at W2
    parameter string LN_ROM2 = "expln_w128_ln.mem",
    localparam int unsigned LIMBS2 = LIMBS + 1,
    localparam int unsigned W2 = LIMBS2 * LIMBW,
    localparam int unsigned MPW2 = W2 + EXPW + 3,
    localparam int unsigned FG2 = W2 + 24,
    localparam int unsigned FXW = ((FG2 + 36 + 3) / 4) * 4,  // inv2pi ROM width
    localparam int unsigned UW = FG2 + 33                    // u = t/2pi, Q33.FG2
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            in_valid,
    output logic            in_ready,
    input  logic [63:0]     t_fx,
    input  logic [MPW2-1:0] inv_t,
    output logic            out_valid,
    output logic [PHW-1:0]  theta_o,
    output logic [MPW2-1:0] lnu_o     // ln(t/2pi) as MPF@W2 (RS Z-epilogue reuse)
);

  localparam int unsigned SIGN_B2 = W2 + EXPW;
  localparam int unsigned ZERO_B2 = W2 + EXPW + 1;
  localparam int unsigned EXP_LO2 = W2;
  localparam int unsigned THW = FG2 + 2;  // theta accumulator, mod 4

  logic [FXW-1:0] inv2pi_rom [0:0];
  logic [MPW2-1:0] coef_rom [0:KTERMS-1];

  initial begin
    $readmemh(THETA_FX_ROM, inv2pi_rom);
    $readmemh(THETA_MPF_ROM, coef_rom);
  end

  localparam logic [MPW2-1:0] NEG_ONE2 =
      {1'b0, 1'b0, 1'b1, EXPW'(1), {1'b1, {(W2 - 1) {1'b0}}}};

  // ---- W2-format datapath units ----------------------------------------------
  logic log_go, log_ready, log_done;
  logic [MPW2-1:0] log_in, log_out;

  /* verilator lint_off PINCONNECTEMPTY */
  log_mpf #(
      .LIMBS(LIMBS2), .EXPW(EXPW), .LIMBW(LIMBW), .FG(FG2), .CONSTW(FG2 + 8),
      .CONST_LINES(LOG_TERMS + 2), .CONSTS_ROM(EXP_ROM2), .LN_ROM(LN_ROM2)
  ) u_log (
      .clk(clk), .rst_n(rst_n),
      .in_valid(log_go), .in_ready(log_ready), .x(log_in),
      .out_valid(log_done), .result(log_out), .ovf(), .unf()
  );

  logic mul_go, mul_done;
  logic [MPW2-1:0] mul_x, mul_y, mul_out;

  mpf_mul #(
      .LIMBS(LIMBS2), .EXPW(EXPW), .LIMBW(LIMBW), .TW(TW), .TILE_LATENCY(TILE_LATENCY)
  ) u_mul (
      .clk(clk), .rst_n(rst_n), .in_valid(mul_go),
      .x(mul_x), .y(mul_y),
      .out_valid(mul_done), .result(mul_out), .ovf(), .unf()
  );

  logic add_go, add_done;
  logic [MPW2-1:0] add_x, add_y, add_out;

  mpf_add #(
      .LIMBS(LIMBS2), .EXPW(EXPW), .LIMBW(LIMBW)
  ) u_add (
      .clk(clk), .rst_n(rst_n), .in_valid(add_go),
      .x(add_x), .y(add_y),
      .out_valid(add_done), .result(add_out), .ovf(), .unf()
  );
  /* verilator lint_on PINCONNECTEMPTY */

  // ---- fixed<->MPF converters ---------------------------------------------------
  // u (positive, scale 2^FG2) -> MPF@W2, RNE
  logic [UW-1:0] u_q;
  logic [$clog2(UW+1)-1:0] u_lz;
  logic u_zero;

  lzc #(
      .WIDTH(UW)
  ) u_lzc (
      .data    (u_q),
      .count   (u_lz),
      .all_zero(u_zero)
  );

  logic [MPW2-1:0] u_mpf;
  logic [W2+1:0] ut_norm;
  logic [W2:0] um_rnd;
  // verilator lint_off UNUSEDSIGNAL
  logic signed [23:0] ue_norm, ue_f;
  // verilator lint_on UNUSEDSIGNAL

  always_comb begin
    automatic int p = int'(UW) - 1 - int'(32'(u_lz));
    ue_norm = 24'(p - int'(FG2) + 1);
    if (p >= int'(W2)) begin
      ut_norm = (W2+2)'(u_q >> unsigned'(p - int'(W2)));
    end else begin
      ut_norm = (W2+2)'(u_q) << unsigned'(int'(W2) - p);
    end
    um_rnd = (W2+1)'((ut_norm + (W2+2)'(1)) >> 1);
    if (um_rnd[W2]) begin
      u_mpf = {1'b0, 1'b0, 1'b0, (ue_norm[EXPW-1:0] + EXPW'(1)), um_rnd[W2:1]};
    end else begin
      u_mpf = {1'b0, 1'b0, 1'b0, ue_norm[EXPW-1:0], um_rnd[W2-1:0]};
    end
  end

  // MPF@W2 -> signed fixed at scale 2^FG2 (truncate toward zero)
  function automatic logic signed [THW+8:0] mpf_to_fx(input logic [MPW2-1:0] v);
    logic signed [EXPW-1:0] e;
    logic [THW+8:0] mag;
    int sh;
    if (v[ZERO_B2]) return '0;
    e  = signed'(v[EXP_LO2+:EXPW]);
    sh = int'(FG2) + int'(e) - int'(W2);
    if (sh >= 0) begin
      mag = (THW + 9)'(v[W2-1:0]) << unsigned'(sh);
    end else if (-sh < int'(W2)) begin
      mag = (THW + 9)'(v[W2-1:0]) >> unsigned'(-sh);
    end else begin
      mag = '0;
    end
    return v[SIGN_B2] ? -signed'(mag) : signed'(mag);
  endfunction

  // ---- FSM -------------------------------------------------------------------------
  typedef enum logic [3:0] {
    IDLE, UMUL, LOG_GO, LOG_W, LM1, LM1_W, MAIN, V2, V2_W,
    H_MUL, H_MUL_W, H_ADD_W, TAILM, TAILM_W, DONE
  } state_e;
  state_e state;

  logic [MPW2-1:0] invt_q, lnu_q, v2_q, s_q;
  logic [63:0] tq;
  logic [THW-1:0] theta_q;
  logic [$clog2(KTERMS+1)-1:0] j_ctr;

  // u = (t_fx * inv2pi) >> 64
  logic [64+FXW-1:0] p_u;
  assign p_u = tq * inv2pi_rom[0];

  // main term product: u * (ln u - 1)_fx
  logic signed [THW+8:0] l_fx;
  logic signed [UW+THW+9:0] p_main;
  assign p_main = signed'({1'b0, u_q}) * l_fx;

  assign in_ready = (state == IDLE);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= IDLE;
      out_valid <= 1'b0;
    end else begin
      out_valid <= 1'b0;
      log_go    <= 1'b0;
      mul_go    <= 1'b0;
      add_go    <= 1'b0;
      unique case (state)
        IDLE: begin
          if (in_valid) begin
            tq     <= t_fx;
            invt_q <= inv_t;
            state  <= UMUL;
          end
        end
        UMUL: begin
          u_q   <= UW'(p_u >> 64);
          state <= LOG_GO;
        end
        LOG_GO: begin
          log_in <= u_mpf;
          log_go <= 1'b1;
          state  <= LOG_W;
        end
        LOG_W: if (log_done) begin
          lnu_q <= log_out;
          state <= LM1;
        end
        LM1: begin
          add_x  <= lnu_q;
          add_y  <= NEG_ONE2;
          add_go <= 1'b1;
          state  <= LM1_W;
        end
        LM1_W: if (add_done) begin
          l_fx  <= mpf_to_fx(add_out);
          state <= MAIN;
        end
        MAIN: begin
          // (u/2)(ln u - 1) - 1/16, mod 4, at scale 2^FG2
          theta_q <= THW'(p_main >>> (FG2 + 1)) - (THW'(1) << (FG2 - 4));
          state   <= V2;
        end
        V2: begin
          mul_x  <= invt_q;
          mul_y  <= invt_q;
          mul_go <= 1'b1;
          state  <= V2_W;
        end
        V2_W: if (mul_done) begin
          v2_q  <= mul_out;
          s_q   <= coef_rom[KTERMS-1];
          j_ctr <= ($clog2(KTERMS+1))'(KTERMS - 1);
          state <= (KTERMS > 1) ? H_MUL : TAILM;
        end
        H_MUL: begin
          mul_x  <= s_q;
          mul_y  <= v2_q;
          mul_go <= 1'b1;
          state  <= H_MUL_W;
        end
        H_MUL_W: if (mul_done) begin
          add_x  <= coef_rom[32'(j_ctr) - 1];
          add_y  <= mul_out;
          add_go <= 1'b1;
          state  <= H_ADD_W;
        end
        H_ADD_W: if (add_done) begin
          s_q <= add_out;
          if (j_ctr == 1) begin
            state <= TAILM;
          end else begin
            j_ctr <= j_ctr - 1'b1;
            state <= H_MUL;
          end
        end
        TAILM: begin
          mul_x  <= s_q;
          mul_y  <= invt_q;
          mul_go <= 1'b1;
          state  <= TAILM_W;
        end
        TAILM_W: if (mul_done) begin
          theta_q <= theta_q + THW'(mpf_to_fx(mul_out));
          state   <= DONE;
        end
        DONE: begin
          theta_o   <= theta_q[FG2-1-:PHW];
          lnu_o     <= lnu_q;
          out_valid <= 1'b1;
          state     <= IDLE;
        end
        default: state <= IDLE;
      endcase
    end
  end

  // Structurally unused (u_zero: u >= 1 always; log_ready: single in flight).
  // verilator lint_off UNUSEDSIGNAL
  logic unused_bits;
  assign unused_bits = u_zero | log_ready;
  // verilator lint_on UNUSEDSIGNAL

endmodule
