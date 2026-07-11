// mpf_mul: MPF floating-point multiplier, fully pipelined (II = 1).
//
//   result = x * y, correctly rounded (RNE).
//
// Structure: limb_mul computes the exact 2*WIDTH-bit mantissa product; the
// sign/exponent/flag sideband travels in shift registers of the same depth;
// one final registered stage normalizes (0/1-bit shift), rounds, saturates,
// and packs. ovf/unf pulse with out_valid on exponent saturation.
//
//   LATENCY = limb_mul latency + 1 = TILE_LATENCY + $clog2(NT*NT) + 1
//
// Format layout: see rtl/common/pkg/mp_pkg.sv.
// Golden model: host/zetafpga/golden/mpfloat.py::mpf_mul
module mpf_mul #(
    parameter int unsigned LIMBS = 2,
    parameter int unsigned EXPW = 20,
    parameter int unsigned LIMBW = 64,
    parameter int unsigned TW = 32,
    parameter int unsigned TILE_LATENCY = 2,
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned MPW = WIDTH + EXPW + 3,
    // Mirrors limb_mul's derived latency (same parameters, same formula).
    localparam int unsigned MUL_LATENCY = TILE_LATENCY + $clog2((WIDTH/TW) * (WIDTH/TW))
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

  // ---- classify and launch the mantissa product ----------------------------
  logic sgn_in, spec_in, zero_in;
  exp2_t esum_in;

  assign sgn_in  = x[SIGN_B] ^ y[SIGN_B];
  assign spec_in = x[SPEC_B] | y[SPEC_B];
  assign zero_in = (x[ZERO_B] | y[ZERO_B]) & ~spec_in;
  assign esum_in = exp2_t'(signed'(x[EXP_LO+:EXPW])) + exp2_t'(signed'(y[EXP_LO+:EXPW]));

  logic mul_valid;
  logic [2*WIDTH-1:0] p;

  limb_mul #(
      .LIMBS(LIMBS), .LIMBW(LIMBW), .TW(TW), .TILE_LATENCY(TILE_LATENCY)
  ) u_mul (
      .clk(clk), .rst_n(rst_n), .in_valid(in_valid),
      .a(x[WIDTH-1:0]), .b(y[WIDTH-1:0]),
      .out_valid(mul_valid), .product(p)
  );

  // ---- sideband delay matching the multiplier latency -----------------------
  logic [MUL_LATENCY-1:0] sgn_d, spec_d, zero_d;
  exp2_t esum_d [0:MUL_LATENCY-1];

  always_ff @(posedge clk) begin
    sgn_d  <= {sgn_d[MUL_LATENCY-2:0], sgn_in};
    spec_d <= {spec_d[MUL_LATENCY-2:0], spec_in};
    zero_d <= {zero_d[MUL_LATENCY-2:0], zero_in};
    esum_d[0] <= esum_in;
    for (int unsigned i = 1; i < MUL_LATENCY; i++) begin
      esum_d[i] <= esum_d[i-1];
    end
  end

  // ---- normalize, round (RNE), saturate, pack ------------------------------
  logic sgn_l, spec_l, zero_l;
  exp2_t esum_l;
  assign sgn_l  = sgn_d[MUL_LATENCY-1];
  assign spec_l = spec_d[MUL_LATENCY-1];
  assign zero_l = zero_d[MUL_LATENCY-1];
  assign esum_l = esum_d[MUL_LATENCY-1];

  logic [2*WIDTH-1:0] norm;
  exp2_t e_norm;
  logic [WIDTH-1:0] mant;
  logic g, s, rup;
  logic [WIDTH:0] mant_r;
  logic [WIDTH-1:0] mant_f;
  exp2_t e_f;

  always_comb begin
    if (p[2*WIDTH-1]) begin
      norm   = p;
      e_norm = esum_l;
    end else begin
      norm   = p << 1;
      e_norm = esum_l - exp2_t'(1);
    end
    mant   = norm[2*WIDTH-1:WIDTH];
    g      = norm[WIDTH-1];
    s      = |norm[WIDTH-2:0];
    rup    = g && (s || mant[0]);
    mant_r = {1'b0, mant} + {{WIDTH{1'b0}}, rup};
    mant_f = mant_r[WIDTH] ? {1'b1, {(WIDTH - 1) {1'b0}}} : mant_r[WIDTH-1:0];
    e_f    = e_norm + (mant_r[WIDTH] ? exp2_t'(1) : exp2_t'(0));
  end

  always_ff @(posedge clk) begin
    ovf <= 1'b0;
    unf <= 1'b0;
    if (spec_l) begin
      result <= {1'b1, 1'b0, sgn_l, {EXPW{1'b0}}, {WIDTH{1'b0}}};
    end else if (zero_l) begin
      result <= {1'b0, 1'b1, sgn_l, {EXPW{1'b0}}, {WIDTH{1'b0}}};
    end else if (e_f > exp2_t'(EMAX)) begin
      result <= {1'b1, 1'b0, sgn_l, {EXPW{1'b0}}, {WIDTH{1'b0}}};
      ovf    <= 1'b1;
    end else if (e_f < exp2_t'(EMIN)) begin
      result <= {1'b0, 1'b1, sgn_l, {EXPW{1'b0}}, {WIDTH{1'b0}}};
      unf    <= 1'b1;
    end else begin
      result <= {1'b0, 1'b0, sgn_l, e_f[EXPW-1:0], mant_f};
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      out_valid <= 1'b0;
    end else begin
      out_valid <= mul_valid;
    end
  end

endmodule
