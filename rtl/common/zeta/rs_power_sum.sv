// rs_power_sum: pipelined Riemann-Siegel main-sum engine, II = 1 (M12).
//
//   S = sum_{n=1..N} amp_n * e^(-i * t * ln n)
//
// One table entry (host-precomputed amplitude n^-sigma, |amp| <= 1, plus the
// full-value ln n/2pi fraction) is consumed per cycle:
//
//   entry -> fx_mul_mod1 (comb phase) -> cexp_pipe (II=1) ->
//   2x mpf_mul (amp*cos, amp*sin) -> MPF->fixed conversion ->
//   single-cycle Q27.FRAC complex accumulate -> final normalize to MPF.
//
// The fixed-point accumulator is what makes II=1 possible (mpf_add has a
// 3-cycle dependency); it is exact to 2^-FRAC per term, valid because every
// term is bounded by 1 on the critical line (host contract).
//
// Throughput: N + ~(cexp latency + mul latency + 6) cycles per sum, vs
// ~130*N for the sequential COMPUTE_PS path.
//
// Golden model: host/zetafpga/golden/rs_pipe.py::rs_power_sum (bit-exact).
module rs_power_sum #(
    parameter int unsigned LIMBS = 1,
    parameter int unsigned EXPW = 20,
    parameter int unsigned LIMBW = 64,
    parameter int unsigned TW = 32,
    parameter int unsigned TILE_LATENCY = 2,
    parameter int unsigned PHW = 96,
    parameter int unsigned FG = 88,
    parameter int unsigned CONSTW = 96,
    parameter int unsigned CTERMS = 11,
    parameter int unsigned SEGW = 10,
    parameter int unsigned EXP_TERMS = 22,  // lines-2 of EXP_ROM (invfact count)
    parameter string CEXP_ROM = "cexp_w64.mem",
    parameter string EXP_ROM = "expln_w64_exp.mem",
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned MPW = WIDTH + EXPW + 3,
    localparam int unsigned BW = PHW + 32,
    localparam int unsigned FRAC = WIDTH + 16,
    localparam int unsigned ACCW = FRAC + 27,           // + 26 int bits + sign
    localparam int unsigned CEXP_LAT = CTERMS + 2
) (
    input  logic            clk,
    input  logic            rst_n,

    input  logic            start_valid,
    output logic            start_ready,
    input  logic [63:0]     t_fx,
    input  logic [23:0]     n_in,

    input  logic            entry_valid,
    output logic            entry_ready,
    input  logic [BW+7:0]   lnn2pi,
    input  logic [MPW-1:0]  amp,

    output logic            out_valid,
    output logic [MPW-1:0]  s_re,
    output logic [MPW-1:0]  s_im,

    // Raw accumulators (scale 2^FRAC), stable when out_valid pulses. The
    // multi-lane wrapper (rs_power_sum_tiled, M17) merges these exactly —
    // fixed-point partial sums are order-independent.
    output logic signed [ACCW-1:0] acc_re_o,
    output logic signed [ACCW-1:0] acc_im_o
);

  localparam int unsigned SIGN_B = WIDTH + EXPW;
  localparam int unsigned ZERO_B = WIDTH + EXPW + 1;
  localparam int unsigned EXP_LO = WIDTH;

  typedef enum logic [1:0] { IDLE, RUN, NORM_R, NORM_I } state_e;
  state_e state;

  logic [63:0] t_q;
  logic [23:0] n_q, issued, done_cnt;
  logic signed [ACCW-1:0] acc_r, acc_i;

  assign acc_re_o = acc_r;
  assign acc_im_o = acc_i;

  assign start_ready = (state == IDLE);
  assign entry_ready = (state == RUN) && (issued < n_q);

  // ---- phase (combinational) + pipelined cexp --------------------------------
  logic [PHW-1:0] phi;

  fx_mul_mod1 #(
      .AW(64), .AF(32), .BW(BW), .BI(8), .PHW(PHW)
  ) u_phase (
      .a   (t_q),
      .b   (lnn2pi),
      .frac(phi)
  );

  logic entry_beat;
  assign entry_beat = entry_valid && entry_ready;

  logic cexp_v;
  logic [MPW-1:0] cos_w, sin_w;

  cexp_pipe #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW), .PHW(PHW), .FG(FG),
      .CONSTW(CONSTW), .SEGW(SEGW), .TERMS(CTERMS),
      .CONST_LINES(EXP_TERMS + 2), .CEXP_ROM(CEXP_ROM), .EXP_ROM(EXP_ROM)
  ) u_cexp (
      .clk(clk), .rst_n(rst_n),
      .in_valid(entry_beat), .phi(phi),
      .out_valid(cexp_v), .cos_o(cos_w), .sin_o(sin_w)
  );

  // Amplitude sideband: free-running shift register aligned with cexp_pipe.
  logic [MPW-1:0] amp_d [0:CEXP_LAT-1];

  always_ff @(posedge clk) begin
    amp_d[0] <= amp;
    for (int unsigned i = 1; i < CEXP_LAT; i++) begin
      amp_d[i] <= amp_d[i-1];
    end
  end

  // ---- amplitude scaling (two II=1 multipliers) --------------------------------
  logic mul_v, mul_v_i;
  logic [MPW-1:0] p_re, p_im;

  // Flags unused: |amp| <= 1 and |cos/sin| <= 1 cannot overflow; underflow
  // to zero is exact for the accumulator.
  /* verilator lint_off PINCONNECTEMPTY */
  mpf_mul #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW), .TW(TW), .TILE_LATENCY(TILE_LATENCY)
  ) u_mul_re (
      .clk(clk), .rst_n(rst_n), .in_valid(cexp_v),
      .x(amp_d[CEXP_LAT-1]), .y(cos_w),
      .out_valid(mul_v), .result(p_re), .ovf(), .unf()
  );

  mpf_mul #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW), .TW(TW), .TILE_LATENCY(TILE_LATENCY)
  ) u_mul_im (
      .clk(clk), .rst_n(rst_n), .in_valid(cexp_v),
      .x(amp_d[CEXP_LAT-1]), .y(sin_w),
      .out_valid(mul_v_i), .result(p_im), .ovf(), .unf()
  );
  /* verilator lint_on PINCONNECTEMPTY */

  // ---- MPF -> fixed conversion (|v| <= 1 by host contract) ----------------------
  function automatic logic signed [ACCW-1:0] to_fxa(input logic [MPW-1:0] v);
    logic signed [EXPW-1:0] e;
    logic [ACCW-1:0] mag;
    int sh;
    if (v[ZERO_B]) return '0;
    e  = signed'(v[EXP_LO+:EXPW]);
    sh = int'(FRAC) + int'(e) - int'(WIDTH);
    if (sh >= 0) begin
      mag = ACCW'(v[WIDTH-1:0]) << unsigned'(sh);
    end else if (-sh < int'(WIDTH)) begin
      mag = ACCW'(v[WIDTH-1:0]) >> unsigned'(-sh);
    end else begin
      mag = '0;
    end
    return v[SIGN_B] ? -signed'(mag) : signed'(mag);
  endfunction

  // ---- accumulate + drain --------------------------------------------------------
  logic signed [ACCW-1:0] norm_in;
  assign norm_in = (state == NORM_R) ? acc_r : acc_i;

  logic [MPW-1:0] norm_word;

  rs_acc_norm #(
      .EXPW(EXPW), .WIDTH(WIDTH), .FRAC(FRAC), .ACCW(ACCW)
  ) u_norm (
      .val (norm_in),
      .word(norm_word)
  );

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= IDLE;
      out_valid <= 1'b0;
    end else begin
      out_valid <= 1'b0;
      if (mul_v) begin
        acc_r    <= acc_r + to_fxa(p_re);
        acc_i    <= acc_i - to_fxa(p_im);  // conjugate: e^(-i t ln n)
        done_cnt <= done_cnt + 24'd1;
      end
      unique case (state)
        IDLE: begin
          if (start_valid) begin
            t_q      <= t_fx;
            n_q      <= n_in;
            issued   <= 24'd0;
            done_cnt <= 24'd0;
            acc_r    <= '0;
            acc_i    <= '0;
            state    <= RUN;
          end
        end
        RUN: begin
          if (entry_beat) begin
            issued <= issued + 24'd1;
          end
          if (done_cnt == n_q && n_q != 0) begin
            state <= NORM_R;
          end
        end
        NORM_R: begin
          s_re  <= norm_word;
          state <= NORM_I;
        end
        NORM_I: begin
          s_im      <= norm_word;
          out_valid <= 1'b1;
          state     <= IDLE;
        end
        default: state <= IDLE;
      endcase
    end
  end

  // mul_v_i mirrors mul_v (identical unit latency).
  // verilator lint_off UNUSEDSIGNAL
  logic unused_mvi;
  assign unused_mvi = mul_v_i;
  // verilator lint_on UNUSEDSIGNAL

endmodule

// Combinational ACCW-wide fixed point (scale 2^FRAC) -> MPF normalizer (RNE).
// Local helper module.
// verilator lint_off DECLFILENAME
module rs_acc_norm #(
    parameter int unsigned EXPW = 20,
    parameter int unsigned WIDTH = 64,
    parameter int unsigned FRAC = 80,
    parameter int unsigned ACCW = 107,
    localparam int unsigned MPW = WIDTH + EXPW + 3
) (
    input  logic signed [ACCW-1:0] val,
    output logic [MPW-1:0]         word
);

  logic sign;
  logic [ACCW-1:0] mag;
  logic [$clog2(ACCW+1)-1:0] lz;
  logic zero;

  assign sign = val < 0;
  assign mag  = sign ? unsigned'(-val) : unsigned'(val);

  lzc #(
      .WIDTH(ACCW)
  ) u_lzc (
      .data    (mag),
      .count   (lz),
      .all_zero(zero)
  );

  logic [WIDTH+1:0] t_norm;
  logic [WIDTH:0] mant_rnd;
  // |sum| <= N: exponent far from saturation, top bits structurally unused.
  // verilator lint_off UNUSEDSIGNAL
  logic signed [23:0] e_norm, e_f;
  // verilator lint_on UNUSEDSIGNAL
  logic [WIDTH-1:0] mant_f;

  always_comb begin
    automatic int p = int'(ACCW) - 1 - int'(32'(lz));
    e_norm = 24'(p - int'(FRAC) + 1);
    if (p >= int'(WIDTH)) begin
      t_norm = (WIDTH+2)'(((ACCW + WIDTH + 2)'(mag)) >> unsigned'(p - int'(WIDTH)));
    end else begin
      t_norm = (WIDTH+2)'(mag) << unsigned'(int'(WIDTH) - p);
    end
    mant_rnd = (WIDTH+1)'((t_norm + (WIDTH+2)'(1)) >> 1);
    if (mant_rnd[WIDTH]) begin
      mant_f = mant_rnd[WIDTH:1];
      e_f    = e_norm + 24'sd1;
    end else begin
      mant_f = mant_rnd[WIDTH-1:0];
      e_f    = e_norm;
    end
    if (zero) begin
      word = {1'b0, 1'b1, 1'b0, {EXPW{1'b0}}, {WIDTH{1'b0}}};
    end else begin
      word = {1'b0, 1'b0, sign, e_f[EXPW-1:0], mant_f};
    end
  end

endmodule
// verilator lint_on DECLFILENAME
