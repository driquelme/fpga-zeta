// rs_z_unit: fully on-chip Z(t) in two phases (M15/M16/M18).
//
//   Z(t) = 2 Re(e^(i theta) S)
//        + (-1)^(N-1) a^(-1/4) sum_{k<=KMAX} C_k(p) a^(-k/2),  a = t/2pi
//
// Phase 1 (in_valid -> prep_valid): everything that does not need the power
// sum — 1/t via mpf_recip@W2 (Newton, no divider), theta via theta_turns
// (which exports ln a at W2), m = sqrt(a) = exp(ln a / 2) and
// r = a^(-1/4) = exp(-ln a / 4) on one exp_mpf@W2, N = floor(m) (so the
// engine derives the main-sum length itself — the key to grid batching),
// p = m - N at W2, z = 2p - 1 narrowed to W, and the whole C_k correction
// by Clenshaw over Chebyshev ROMs (rsck_w*.mem). n_out is presented with
// prep_valid; the parent then runs the power sum for N terms.
//
// Phase 2 (sum_valid -> out_valid): Z = 2(cos theta * S_re - sin theta *
// S_im) + corr.
//
// Validity: t >= t_min (theta_w*.json contract); the host must have written
// at least floor(sqrt(t/2pi)) + 1 RS table entries.
// Golden model: host/zetafpga/golden/rs_z.py::z_prep/z_post (bit-exact).
module rs_z_unit #(
    parameter int unsigned LIMBS = 1,
    parameter int unsigned EXPW = 20,
    parameter int unsigned LIMBW = 64,
    parameter int unsigned TW = 32,
    parameter int unsigned TILE_LATENCY = 2,
    parameter int unsigned PHW = 96,
    // W-level cexp/exp constants (cexp_w*.json / expln_w*.json)
    parameter int unsigned FG = 88,
    parameter int unsigned CONSTW = 96,
    parameter int unsigned SEGW = 10,
    parameter int unsigned CTERMS = 11,
    parameter int unsigned EXP_TERMS = 22,
    parameter string CEXP_ROM = "cexp_w64.mem",
    parameter string EXP_ROM = "expln_w64_exp.mem",
    // theta unit (theta_w*.json) and W2-level constants (expln_w*.json)
    parameter int unsigned KTERMS = 12,
    parameter int unsigned LOG_TERMS = 32,
    parameter string THETA_FX_ROM = "theta_w64_fx.mem",
    parameter string THETA_MPF_ROM = "theta_w64_mpf.mem",
    parameter string EXP_ROM2 = "expln_w128_exp.mem",
    parameter string LN_ROM2 = "expln_w128_ln.mem",
    // Chebyshev correction ROM (rsck_w*.json)
    parameter int unsigned NC = 37,
    parameter int unsigned KMAX = 4,
    parameter string RSCK_ROM = "rsck_w64.mem",
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned MPW = WIDTH + EXPW + 3,
    localparam int unsigned LIMBS2 = LIMBS + 1,
    localparam int unsigned W2 = LIMBS2 * LIMBW,
    localparam int unsigned MPW2 = W2 + EXPW + 3,
    localparam int unsigned FG2 = W2 + 24
) (
    input  logic           clk,
    input  logic           rst_n,

    input  logic           in_valid,
    output logic           in_ready,
    input  logic [63:0]    t_fx,

    output logic           prep_valid,  // pulse; n_out then held to out_valid
    output logic [23:0]    n_out,

    input  logic           sum_valid,   // pulse with the power-sum result
    input  logic [MPW-1:0] s_re,
    input  logic [MPW-1:0] s_im,

    output logic           out_valid,
    output logic [MPW-1:0] z_o
);

  localparam int unsigned SIGN_B = WIDTH + EXPW;
  localparam int unsigned ZERO_B = WIDTH + EXPW + 1;
  localparam int unsigned EXP_LO2 = W2;
  localparam int unsigned SIGN_B2 = W2 + EXPW;
  localparam int unsigned ZERO_B2 = W2 + EXPW + 1;
  localparam int unsigned SPEC_B2 = W2 + EXPW + 2;

  localparam logic [MPW-1:0] ZERO_W = MPW'(1) << ZERO_B;
  localparam logic [MPW2-1:0] NEG_ONE2 =
      {1'b0, 1'b0, 1'b1, EXPW'(1), {1'b1, {(W2 - 1) {1'b0}}}};

  logic [MPW-1:0] coef_rom [0:(KMAX+1)*NC-1];

  initial begin
    $readmemh(RSCK_ROM, coef_rom);
  end

  // ---- helper functions -------------------------------------------------------
  function automatic logic [MPW-1:0] negw(input logic [MPW-1:0] v);
    return v ^ (MPW'(1) << SIGN_B);
  endfunction

  function automatic logic [MPW2-1:0] neg2(input logic [MPW2-1:0] v);
    return v ^ (MPW2'(1) << SIGN_B2);
  endfunction

  // exact /2 of a nonzero finite MPF@W2 (exponent decrement)
  function automatic logic [MPW2-1:0] half2(input logic [MPW2-1:0] v);
    return {v[MPW2-1:EXP_LO2+EXPW], v[EXP_LO2+:EXPW] - EXPW'(1), v[EXP_LO2-1:0]};
  endfunction

  // round MPF@W2 -> MPF@W, RNE on the dropped 64 bits
  function automatic logic [MPW-1:0] narrow(input logic [MPW2-1:0] v);
    logic [WIDTH:0] mplus;
    logic [EXPW-1:0] e;
    logic rnd;
    logic [63:0] low;
    if (v[ZERO_B2] || v[SPEC_B2]) begin
      return {v[SPEC_B2], v[ZERO_B2], v[SIGN_B2], {EXPW{1'b0}}, {WIDTH{1'b0}}};
    end
    low = v[63:0];
    rnd = (low > 64'h8000_0000_0000_0000)
       || (low == 64'h8000_0000_0000_0000 && v[64]);
    mplus = {1'b0, v[W2-1:64]} + (WIDTH + 1)'(rnd);
    e = v[EXP_LO2+:EXPW];
    if (mplus[WIDTH]) begin
      mplus = mplus >> 1;
      e = e + EXPW'(1);
    end
    return {1'b0, 1'b0, v[SIGN_B2], e, mplus[WIDTH-1:0]};
  endfunction

  // exact Q32.32 -> MPF@W2 conversion (t_fx < 2^64 <= W2 bits)
  function automatic logic [MPW2-1:0] tfx2mpf2(input logic [63:0] t);
    int p;
    logic [W2-1:0] mant;
    p = 0;
    for (int i = 0; i < 64; i++) begin
      if (t[i]) p = i;
    end
    mant = W2'(t) << (int'(W2) - 1 - p);
    return {1'b0, 1'b0, 1'b0, EXPW'(p - 31), mant};
  endfunction

  // exact small-integer conversion (1 <= n < 2^24) into MPF@W2
  function automatic logic [MPW2-1:0] int2mpf2(input logic [23:0] n);
    int p;
    logic [W2-1:0] mant;
    p = 0;
    for (int i = 0; i < 24; i++) begin
      if (n[i]) p = i;
    end
    mant = W2'(n) << (int'(W2) - 1 - p);
    return {1'b0, 1'b0, 1'b0, EXPW'(p + 1), mant};
  endfunction

  // ---- datapath units --------------------------------------------------------
  logic [63:0] tq;

  logic rc_go, rc_ready, rc_done;
  logic [MPW2-1:0] rc_out;
  logic [MPW2-1:0] invt_q;

  /* verilator lint_off PINCONNECTEMPTY */
  mpf_recip #(
      .LIMBS(LIMBS2), .EXPW(EXPW), .LIMBW(LIMBW)
  ) u_recip (
      .clk(clk), .rst_n(rst_n),
      .in_valid(rc_go), .in_ready(rc_ready), .x(tfx2mpf2(tq)),
      .out_valid(rc_done), .result(rc_out), .ovf(), .unf()
  );
  /* verilator lint_on PINCONNECTEMPTY */

  logic th_go, th_ready, th_done;
  logic [PHW-1:0] th_phi;
  logic [MPW2-1:0] th_lnu;

  theta_turns #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW), .TW(TW),
      .TILE_LATENCY(TILE_LATENCY), .PHW(PHW), .KTERMS(KTERMS),
      .LOG_TERMS(LOG_TERMS), .THETA_FX_ROM(THETA_FX_ROM),
      .THETA_MPF_ROM(THETA_MPF_ROM), .EXP_ROM2(EXP_ROM2), .LN_ROM2(LN_ROM2)
  ) u_theta (
      .clk(clk), .rst_n(rst_n),
      .in_valid(th_go), .in_ready(th_ready),
      .t_fx(tq), .inv_t(invt_q),
      .out_valid(th_done), .theta_o(th_phi), .lnu_o(th_lnu)
  );

  logic ce_go, ce_ready, ce_done;
  logic [MPW-1:0] ce_cos, ce_sin;

  cexp_turns #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW), .PHW(PHW), .FG(FG),
      .CONSTW(CONSTW), .SEGW(SEGW), .TERMS(CTERMS),
      .CONST_LINES(EXP_TERMS + 2), .CEXP_ROM(CEXP_ROM), .EXP_ROM(EXP_ROM)
  ) u_cexp (
      .clk(clk), .rst_n(rst_n),
      .in_valid(ce_go), .in_ready(ce_ready), .phi(th_phi),
      .out_valid(ce_done), .cos_o(ce_cos), .sin_o(ce_sin)
  );

  logic exp2_go, exp2_ready, exp2_done;
  logic [MPW2-1:0] exp2_in, exp2_out;

  /* verilator lint_off PINCONNECTEMPTY */
  exp_mpf #(
      .LIMBS(LIMBS2), .EXPW(EXPW), .LIMBW(LIMBW), .FG(FG2), .CONSTW(FG2 + 8),
      .TERMS(LOG_TERMS), .CONSTS_ROM(EXP_ROM2)
  ) u_exp2 (
      .clk(clk), .rst_n(rst_n),
      .in_valid(exp2_go), .in_ready(exp2_ready), .x(exp2_in),
      .fx_mode(1'b0), .yfx_in('0),
      .out_valid(exp2_done), .result(exp2_out), .ovf(), .unf()
  );

  logic mul_go, mul_done;
  logic [MPW-1:0] mul_x, mul_y, mul_out;

  mpf_mul #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW), .TW(TW), .TILE_LATENCY(TILE_LATENCY)
  ) u_mul (
      .clk(clk), .rst_n(rst_n), .in_valid(mul_go),
      .x(mul_x), .y(mul_y),
      .out_valid(mul_done), .result(mul_out), .ovf(), .unf()
  );

  logic add_go, add_done;
  logic [MPW-1:0] add_x, add_y, add_out;

  mpf_add #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW)
  ) u_add (
      .clk(clk), .rst_n(rst_n), .in_valid(add_go),
      .x(add_x), .y(add_y),
      .out_valid(add_done), .result(add_out), .ovf(), .unf()
  );

  logic add2_go, add2_done;
  logic [MPW2-1:0] add2_x, add2_y, add2_out;

  mpf_add #(
      .LIMBS(LIMBS2), .EXPW(EXPW), .LIMBW(LIMBW)
  ) u_add2 (
      .clk(clk), .rst_n(rst_n), .in_valid(add2_go),
      .x(add2_x), .y(add2_y),
      .out_valid(add2_done), .result(add2_out), .ovf(), .unf()
  );
  /* verilator lint_on PINCONNECTEMPTY */

  // ---- FSM -------------------------------------------------------------------
  typedef enum logic [5:0] {
    IDLE, RC, RCW, THG, THW, EM, EMW, ER, ERW, QM, QMW, NF, PA, PAW,
    TA, TAW, ZA, ZAW, ZD, ZDW,
    CLI, CLM, CLMW, CA1, CA1W, CA2, CA2W, CFM, CFMW, CF1, CF1W, CF2, CF2W,
    HM, HMW, HA, HAW, RM, RMW, PREP, WSUM,
    CEG, CEW, M1, M1W, M2, M2W, AM, AMW, AD, ADW, FZ, FZW, DONE
  } state_e;
  state_e state;

  logic [MPW-1:0] sre_q, sim_q, c_q, s_q, m1_q, main_q;
  logic [MPW-1:0] r_q, q_q, zc_q, zcd_q, b1_q, b2_q, ck_q, corr_q, tmp_q;
  logic [MPW2-1:0] lnu_q, m2_q, p2_q;
  logic [2:0] k_ctr;
  logic [$clog2(NC)-1:0] j_ctr;

  assign in_ready = (state == IDLE);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state      <= IDLE;
      out_valid  <= 1'b0;
      prep_valid <= 1'b0;
    end else begin
      out_valid  <= 1'b0;
      prep_valid <= 1'b0;
      rc_go      <= 1'b0;
      th_go      <= 1'b0;
      ce_go      <= 1'b0;
      exp2_go    <= 1'b0;
      mul_go     <= 1'b0;
      add_go     <= 1'b0;
      add2_go    <= 1'b0;
      unique case (state)
        IDLE: begin
          if (in_valid) begin
            tq    <= t_fx;
            state <= RC;
          end
        end
        // phase 1: 1/t, theta, power chain, N, correction ------------------------
        RC: begin
          rc_go <= 1'b1;
          state <= RCW;
        end
        RCW: if (rc_done) begin
          invt_q <= rc_out;
          state  <= THG;
        end
        THG: begin
          th_go <= 1'b1;
          state <= THW;
        end
        THW: if (th_done) begin
          lnu_q <= th_lnu;
          state <= EM;
        end
        EM: begin
          exp2_in <= half2(lnu_q);  // m = exp(ln a / 2)
          exp2_go <= 1'b1;
          state   <= EMW;
        end
        EMW: if (exp2_done) begin
          m2_q  <= exp2_out;
          state <= ER;
        end
        ER: begin
          exp2_in <= neg2(half2(half2(lnu_q)));  // r = exp(-ln a / 4)
          exp2_go <= 1'b1;
          state   <= ERW;
        end
        ERW: if (exp2_done) begin
          r_q   <= narrow(exp2_out);
          state <= QM;
        end
        QM: begin
          mul_x  <= r_q;
          mul_y  <= r_q;
          mul_go <= 1'b1;
          state  <= QMW;
        end
        QMW: if (mul_done) begin
          q_q   <= mul_out;  // q = a^(-1/2)
          state <= NF;
        end
        NF: begin
          // N = floor(m): m >= 1 by the t >= t_min contract
          n_out <= 24'(m2_q[W2-1:0] >> (W2 - 32'(m2_q[EXP_LO2+:EXPW])));
          state <= PA;
        end
        PA: begin
          add2_x  <= m2_q;
          add2_y  <= neg2(int2mpf2(n_out));
          add2_go <= 1'b1;
          state   <= PAW;
        end
        PAW: if (add2_done) begin
          p2_q  <= add2_out;  // p = frac(sqrt(a))
          state <= TA;
        end
        TA: begin
          add2_x  <= p2_q;
          add2_y  <= p2_q;
          add2_go <= 1'b1;
          state   <= TAW;
        end
        TAW: if (add2_done) begin
          p2_q  <= add2_out;  // 2p
          state <= ZA;
        end
        ZA: begin
          add2_x  <= p2_q;
          add2_y  <= NEG_ONE2;
          add2_go <= 1'b1;
          state   <= ZAW;
        end
        ZAW: if (add2_done) begin
          zc_q  <= narrow(add2_out);  // z = 2p - 1
          state <= ZD;
        end
        ZD: begin
          add_x  <= zc_q;
          add_y  <= zc_q;
          add_go <= 1'b1;
          state  <= ZDW;
        end
        ZDW: if (add_done) begin
          zcd_q  <= add_out;  // 2z for the Clenshaw recurrence
          corr_q <= ZERO_W;
          k_ctr  <= 3'(KMAX);
          state  <= CLI;
        end
        CLI: begin
          b1_q  <= ZERO_W;
          b2_q  <= ZERO_W;
          j_ctr <= ($clog2(NC))'(NC - 1);
          state <= CLM;
        end
        CLM: begin
          mul_x  <= zcd_q;
          mul_y  <= b1_q;
          mul_go <= 1'b1;
          state  <= CLMW;
        end
        CLMW: if (mul_done) begin
          tmp_q <= mul_out;
          state <= CA1;
        end
        CA1: begin
          add_x  <= tmp_q;
          add_y  <= negw(b2_q);
          add_go <= 1'b1;
          state  <= CA1W;
        end
        CA1W: if (add_done) begin
          tmp_q <= add_out;
          state <= CA2;
        end
        CA2: begin
          add_x  <= coef_rom[32'(k_ctr)*NC+32'(j_ctr)];
          add_y  <= tmp_q;
          add_go <= 1'b1;
          state  <= CA2W;
        end
        CA2W: if (add_done) begin
          b2_q <= b1_q;
          b1_q <= add_out;
          if (j_ctr == 1) begin
            state <= CFM;
          end else begin
            j_ctr <= j_ctr - 1'b1;
            state <= CLM;
          end
        end
        CFM: begin
          mul_x  <= zc_q;
          mul_y  <= b1_q;
          mul_go <= 1'b1;
          state  <= CFMW;
        end
        CFMW: if (mul_done) begin
          tmp_q <= mul_out;
          state <= CF1;
        end
        CF1: begin
          add_x  <= tmp_q;
          add_y  <= negw(b2_q);
          add_go <= 1'b1;
          state  <= CF1W;
        end
        CF1W: if (add_done) begin
          tmp_q <= add_out;
          state <= CF2;
        end
        CF2: begin
          add_x  <= coef_rom[32'(k_ctr)*NC];
          add_y  <= tmp_q;
          add_go <= 1'b1;
          state  <= CF2W;
        end
        CF2W: if (add_done) begin
          ck_q  <= add_out;  // C_k(z)
          state <= HM;
        end
        HM: begin
          mul_x  <= q_q;
          mul_y  <= corr_q;
          mul_go <= 1'b1;
          state  <= HMW;
        end
        HMW: if (mul_done) begin
          tmp_q <= mul_out;
          state <= HA;
        end
        HA: begin
          add_x  <= ck_q;
          add_y  <= tmp_q;
          add_go <= 1'b1;
          state  <= HAW;
        end
        HAW: if (add_done) begin
          corr_q <= add_out;
          if (k_ctr == 0) begin
            state <= RM;
          end else begin
            k_ctr <= k_ctr - 1'b1;
            state <= CLI;
          end
        end
        RM: begin
          mul_x  <= r_q;
          mul_y  <= corr_q;
          mul_go <= 1'b1;
          state  <= RMW;
        end
        RMW: if (mul_done) begin
          corr_q <= n_out[0] ? mul_out : negw(mul_out);  // (-1)^(N-1)
          state  <= PREP;
        end
        PREP: begin
          prep_valid <= 1'b1;
          state      <= WSUM;
        end
        // phase 2: main term + assembly (after the parent's power sum) ----------
        WSUM: begin
          if (sum_valid) begin
            sre_q <= s_re;
            sim_q <= s_im;
            state <= CEG;
          end
        end
        CEG: begin
          ce_go <= 1'b1;  // phi = th_phi (held by theta until its next start)
          state <= CEW;
        end
        CEW: if (ce_done) begin
          c_q   <= ce_cos;
          s_q   <= ce_sin;
          state <= M1;
        end
        M1: begin
          mul_x  <= c_q;
          mul_y  <= sre_q;
          mul_go <= 1'b1;
          state  <= M1W;
        end
        M1W: if (mul_done) begin
          m1_q  <= mul_out;
          state <= M2;
        end
        M2: begin
          mul_x  <= s_q;
          mul_y  <= sim_q;
          mul_go <= 1'b1;
          state  <= M2W;
        end
        M2W: if (mul_done) begin
          tmp_q <= mul_out;
          state <= AM;
        end
        AM: begin
          add_x  <= m1_q;
          add_y  <= negw(tmp_q);
          add_go <= 1'b1;
          state  <= AMW;
        end
        AMW: if (add_done) begin
          main_q <= add_out;
          state  <= AD;
        end
        AD: begin
          add_x  <= main_q;
          add_y  <= main_q;
          add_go <= 1'b1;
          state  <= ADW;
        end
        ADW: if (add_done) begin
          main_q <= add_out;
          state  <= FZ;
        end
        FZ: begin
          add_x  <= main_q;
          add_y  <= corr_q;
          add_go <= 1'b1;
          state  <= FZW;
        end
        FZW: if (add_done) begin
          z_o   <= add_out;
          state <= DONE;
        end
        DONE: begin
          out_valid <= 1'b1;
          state     <= IDLE;
        end
        default: state <= IDLE;
      endcase
    end
  end

  // Structurally unused (single operand in flight per unit).
  // verilator lint_off UNUSEDSIGNAL
  logic unused_bits;
  assign unused_bits = th_ready | ce_ready | exp2_ready | rc_ready;
  // verilator lint_on UNUSEDSIGNAL

endmodule
