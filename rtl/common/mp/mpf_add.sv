// mpf_add: MPF floating-point adder (signed add/sub), 3-stage pipeline (II = 1).
//
//   result = x + y, correctly rounded (RNE).
//
// Stage 1: unpack, order by magnitude (biased-exponent:mantissa key), classify
//          pass-through cases (zero/special operands) and build their result
//          word up front.
// Stage 2: exact alignment into a EW = 2*WIDTH+3-bit window (differences up to
//          WIDTH+2 lose no bits; beyond that the small operand collapses to a
//          sticky 1 in the LSB, which rounds identically), then one wide
//          add/sub. Ordering guarantees no borrow.
// Stage 3: leading-zero normalize, single RNE rounding of the exact value,
//          exponent saturation, pack.
//
// Format layout: see rtl/common/pkg/mp_pkg.sv.
// Golden model: host/zetafpga/golden/mpfloat.py::mpf_add
module mpf_add #(
    parameter int unsigned LIMBS = 2,
    parameter int unsigned EXPW = 20,
    parameter int unsigned LIMBW = 64,
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned MPW = WIDTH + EXPW + 3,
    localparam int unsigned EW = 2 * WIDTH + 3,
    localparam int unsigned AMTW = $clog2(EW + 1)  // pipeline depth (LATENCY) is 3
) (
    input  logic           clk,
    input  logic           rst_n,
    input  logic           in_valid,
    input  logic [MPW-1:0] x,
    input  logic [MPW-1:0] y,
    output logic           out_valid,
    output logic [MPW-1:0] result,
    output logic           ovf,
    output logic           unf
);

  localparam int unsigned EXP_LO = WIDTH;
  localparam int unsigned SIGN_B = WIDTH + EXPW;
  localparam int unsigned ZERO_B = WIDTH + EXPW + 1;
  localparam int unsigned SPEC_B = WIDTH + EXPW + 2;

  localparam int EMAX = (1 << (EXPW - 1)) - 1;
  localparam int EMIN = -(1 << (EXPW - 1));

  typedef logic signed [EXPW+1:0] exp2_t;

  // ---- stage 1: unpack, order, classify -------------------------------------
  logic [EXPW-1:0] ex_b, ey_b;  // biased exponents (unsigned order)
  logic x_ge;
  logic [EXPW:0] d_raw;

  assign ex_b = {~x[MPW-4], x[EXP_LO+:EXPW-1]};
  assign ey_b = {~y[MPW-4], y[EXP_LO+:EXPW-1]};
  assign x_ge = {ex_b, x[WIDTH-1:0]} >= {ey_b, y[WIDTH-1:0]};
  assign d_raw = x_ge ? ({1'b0, ex_b} - {1'b0, ey_b}) : ({1'b0, ey_b} - {1'b0, ex_b});

  logic any_pass;
  logic [MPW-1:0] pass_word;

  always_comb begin
    any_pass = 1'b1;
    if (x[SPEC_B]) begin
      pass_word = {1'b1, 1'b0, x[SIGN_B], {EXPW{1'b0}}, {WIDTH{1'b0}}};
    end else if (y[SPEC_B]) begin
      pass_word = {1'b1, 1'b0, y[SIGN_B], {EXPW{1'b0}}, {WIDTH{1'b0}}};
    end else if (x[ZERO_B] && y[ZERO_B]) begin
      pass_word = {1'b0, 1'b1, x[SIGN_B] & y[SIGN_B], {EXPW{1'b0}}, {WIDTH{1'b0}}};
    end else if (x[ZERO_B]) begin
      pass_word = y;
    end else if (y[ZERO_B]) begin
      pass_word = x;
    end else begin
      any_pass  = 1'b0;
      pass_word = '0;
    end
  end

  logic s1_valid, s1_pass, s1_eff_sub, s1_sign;
  logic [MPW-1:0] s1_pass_word;
  logic [WIDTH-1:0] s1_mant_b, s1_mant_s;
  exp2_t s1_eb;
  logic [EXPW:0] s1_d;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s1_valid <= 1'b0;
    end else begin
      s1_valid <= in_valid;
    end
    s1_pass      <= any_pass;
    s1_pass_word <= pass_word;
    s1_eff_sub   <= x[SIGN_B] ^ y[SIGN_B];
    // Common sign for add; sign of the larger magnitude for subtract.
    s1_sign      <= x_ge ? x[SIGN_B] : y[SIGN_B];
    s1_mant_b    <= x_ge ? x[WIDTH-1:0] : y[WIDTH-1:0];
    s1_mant_s    <= x_ge ? y[WIDTH-1:0] : x[WIDTH-1:0];
    s1_eb        <= exp2_t'(signed'(x_ge ? x[EXP_LO+:EXPW] : y[EXP_LO+:EXPW]));
    s1_d         <= d_raw;
  end

  // ---- stage 2: exact align + add/sub ---------------------------------------
  logic [EW-1:0] mb_ext, ms_ext, ms_shifted;
  logic align_lost;

  assign mb_ext = {1'b0, s1_mant_b, {(WIDTH + 2) {1'b0}}};

  limb_shift #(
      .LIMBS(1), .LIMBW(EW)
  ) u_align (
      .left  (1'b1),
      .amount(AMTW'(WIDTH + 2) - s1_d[AMTW-1:0]),
      .a     ({{(EW - WIDTH) {1'b0}}, s1_mant_s}),
      .result(ms_shifted),
      .lost  (align_lost)
  );

  assign ms_ext = (s1_d > (EXPW + 1)'(WIDTH + 2)) ? {{(EW - 1) {1'b0}}, 1'b1} : ms_shifted;

  logic [EW-1:0] r_sum;
  logic sum_carry;

  limb_addsub #(
      .LIMBS(1), .LIMBW(EW)
  ) u_sum (
      .sub   (s1_eff_sub),
      .cin   (1'b0),
      .a     (mb_ext),
      .b     (ms_ext),
      .result(r_sum),
      .carry (sum_carry)
  );

  logic s2_valid, s2_pass, s2_sign;
  logic [MPW-1:0] s2_pass_word;
  logic [EW-1:0] s2_r;
  exp2_t s2_eb;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s2_valid <= 1'b0;
    end else begin
      s2_valid <= s1_valid;
    end
    s2_pass      <= s1_pass;
    s2_pass_word <= s1_pass_word;
    s2_sign      <= s1_sign;
    s2_r         <= r_sum;
    s2_eb        <= s1_eb;
  end

  // ---- stage 3: normalize, round (RNE), saturate, pack ----------------------
  logic [AMTW-1:0] lz;
  logic r_is_zero;

  lzc #(
      .WIDTH(EW)
  ) u_lzc (
      .data    (s2_r),
      .count   (lz),
      .all_zero(r_is_zero)
  );

  logic [EW-1:0] rn;
  logic norm_lost;

  limb_shift #(
      .LIMBS(1), .LIMBW(EW)
  ) u_norm (
      .left  (1'b1),
      .amount(lz),
      .a     (s2_r),
      .result(rn),
      .lost  (norm_lost)
  );

  logic [WIDTH-1:0] mant;
  logic g, s, rup;
  logic [WIDTH:0] mant_r;
  logic [WIDTH-1:0] mant_f;
  exp2_t e_f;

  always_comb begin
    mant   = rn[EW-1-:WIDTH];
    g      = rn[WIDTH+2];
    s      = |rn[WIDTH+1:0];
    rup    = g && (s || mant[0]);
    mant_r = {1'b0, mant} + {{WIDTH{1'b0}}, rup};
    mant_f = mant_r[WIDTH] ? {1'b1, {(WIDTH - 1) {1'b0}}} : mant_r[WIDTH-1:0];
    e_f    = s2_eb + exp2_t'(1) - exp2_t'({{(EXPW + 2 - AMTW) {1'b0}}, lz})
           + (mant_r[WIDTH] ? exp2_t'(1) : exp2_t'(0));
  end

  always_ff @(posedge clk) begin
    ovf <= 1'b0;
    unf <= 1'b0;
    if (s2_pass) begin
      result <= s2_pass_word;
    end else if (r_is_zero) begin
      result <= {1'b0, 1'b1, 1'b0, {EXPW{1'b0}}, {WIDTH{1'b0}}};  // exact cancel -> +0
    end else if (e_f > exp2_t'(EMAX)) begin
      result <= {1'b1, 1'b0, s2_sign, {EXPW{1'b0}}, {WIDTH{1'b0}}};
      ovf    <= 1'b1;
    end else if (e_f < exp2_t'(EMIN)) begin
      result <= {1'b0, 1'b1, s2_sign, {EXPW{1'b0}}, {WIDTH{1'b0}}};
      unf    <= 1'b1;
    end else begin
      result <= {1'b0, 1'b0, s2_sign, e_f[EXPW-1:0], mant_f};
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      out_valid <= 1'b0;
    end else begin
      out_valid <= s2_valid;
    end
  end

  // Structurally unused outputs of the reused primitives.
  // verilator lint_off UNUSEDSIGNAL
  logic unused_bits;
  assign unused_bits = align_lost | sum_carry | norm_lost;
  // verilator lint_on UNUSEDSIGNAL

endmodule
