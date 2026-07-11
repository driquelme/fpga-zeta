// limb_mul_ks: one Karatsuba level over three half-width schoolbook limb_muls.
//
// For a = a1*2^H + a0, b = b1*2^H + b0 (H = WIDTH/2):
//   z0 = a0*b0,  z2 = a1*b1,  z1 = (a0+a1)*(b0+b1) - z0 - z2
//   product = z2*2^(2H) + z1*2^H + z0
//
// The half-operand sums carry one extra bit each (sa, sb); the cross terms
// they induce are added back explicitly:
//   (sa*2^H + as)*(sb*2^H + bs) = as*bs + sb*as*2^H + sa*bs*2^H + sa*sb*2^(2H)
//
// Uses 3 half-width multipliers instead of 4 — the win compounds at deeper
// recursion / on real DSP tiles; here it exists to prove the decomposition
// (acceptance: bit-equivalence with schoolbook limb_mul). II = 1.
//
//   LATENCY = INNER_LATENCY + 3
//
// Golden model: host/zetafpga/golden/limb.py::mul (same contract as limb_mul).
module limb_mul_ks #(
    parameter int unsigned LIMBS = 4,          // must be even (halves feed limb_mul)
    parameter int unsigned LIMBW = 64,
    parameter int unsigned TW = 32,
    parameter int unsigned TILE_LATENCY = 2,
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned H = WIDTH / 2,
    localparam int unsigned HNT = H / TW,
    localparam int unsigned INNER_LATENCY = TILE_LATENCY + $clog2(HNT * HNT),
    localparam int unsigned LATENCY = INNER_LATENCY + 3
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               in_valid,
    input  logic [WIDTH-1:0]   a,
    input  logic [WIDTH-1:0]   b,
    output logic               out_valid,
    output logic [2*WIDTH-1:0] product
);

  // Half-operand splits and one-extra-bit presums (combinational; the inner
  // multipliers register internally).
  logic [H-1:0] a0, a1, b0, b1;
  logic [H:0]   asum, bsum;

  assign a0 = a[H-1:0];
  assign a1 = a[WIDTH-1:H];
  assign b0 = b[H-1:0];
  assign b1 = b[WIDTH-1:H];
  assign asum = {1'b0, a0} + {1'b0, a1};
  assign bsum = {1'b0, b0} + {1'b0, b1};

  logic [2*H-1:0] p00, p11, pss;

  // Inner out_valid ports are intentionally unconnected: validity is tracked
  // once for the whole datapath by the vpipe shift register below.
  /* verilator lint_off PINCONNECTEMPTY */
  limb_mul #(
      .LIMBS(LIMBS / 2), .LIMBW(LIMBW), .TW(TW), .TILE_LATENCY(TILE_LATENCY)
  ) u_mul00 (
      .clk(clk), .rst_n(rst_n), .in_valid(1'b0),
      .a(a0), .b(b0), .out_valid(), .product(p00)
  );

  limb_mul #(
      .LIMBS(LIMBS / 2), .LIMBW(LIMBW), .TW(TW), .TILE_LATENCY(TILE_LATENCY)
  ) u_mul11 (
      .clk(clk), .rst_n(rst_n), .in_valid(1'b0),
      .a(a1), .b(b1), .out_valid(), .product(p11)
  );

  limb_mul #(
      .LIMBS(LIMBS / 2), .LIMBW(LIMBW), .TW(TW), .TILE_LATENCY(TILE_LATENCY)
  ) u_mulss (
      .clk(clk), .rst_n(rst_n), .in_valid(1'b0),
      .a(asum[H-1:0]), .b(bsum[H-1:0]), .out_valid(), .product(pss)
  );
  /* verilator lint_on PINCONNECTEMPTY */

  // Delay the presum carries and low halves to meet the inner-product outputs.
  logic [INNER_LATENCY-1:0] sa_d, sb_d;
  logic [H-1:0] as_d [0:INNER_LATENCY-1];
  logic [H-1:0] bs_d [0:INNER_LATENCY-1];

  always_ff @(posedge clk) begin
    sa_d <= {sa_d[INNER_LATENCY-2:0], asum[H]};
    sb_d <= {sb_d[INNER_LATENCY-2:0], bsum[H]};
    as_d[0] <= asum[H-1:0];
    bs_d[0] <= bsum[H-1:0];
    for (int unsigned i = 1; i < INNER_LATENCY; i++) begin
      as_d[i] <= as_d[i-1];
      bs_d[i] <= bs_d[i-1];
    end
  end

  // Stage m1: full (a0+a1)*(b0+b1) with carry cross terms restored.
  // Stage m2: subtract z0 and z2 to get z1.
  // Stage f : recombine.
  logic [2*WIDTH-1:0] mid_raw, mid;
  logic [2*H-1:0] p00_d1, p11_d1, p00_d2, p11_d2;

  logic sa, sb;
  logic [H-1:0] as_last, bs_last;
  assign sa = sa_d[INNER_LATENCY-1];
  assign sb = sb_d[INNER_LATENCY-1];
  assign as_last = as_d[INNER_LATENCY-1];
  assign bs_last = bs_d[INNER_LATENCY-1];

  always_ff @(posedge clk) begin
    mid_raw <= {{(2*WIDTH - 2*H){1'b0}}, pss}
             + (sb ? {{(2*WIDTH - 2*H){1'b0}}, as_last, {H{1'b0}}} : '0)
             + (sa ? {{(2*WIDTH - 2*H){1'b0}}, bs_last, {H{1'b0}}} : '0)
             + ((sa && sb) ? {{(2*WIDTH - 2*H - 1){1'b0}}, 1'b1, {(2*H){1'b0}}} : '0);
    p00_d1 <= p00;
    p11_d1 <= p11;

    mid    <= mid_raw - {{(2*WIDTH - 2*H){1'b0}}, p00_d1}
                      - {{(2*WIDTH - 2*H){1'b0}}, p11_d1};
    p00_d2 <= p00_d1;
    p11_d2 <= p11_d1;

    product <= {{(2*WIDTH - 2*H){1'b0}}, p00_d2}
             + (mid << H)
             + ({{(2*WIDTH - 2*H){1'b0}}, p11_d2} << (2 * H));
  end

  // Validity pipeline.
  logic [LATENCY-1:0] vpipe;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      vpipe <= '0;
    end else begin
      vpipe <= {vpipe[LATENCY-2:0], in_valid};
    end
  end

  assign out_valid = vpipe[LATENCY-1];

endmodule
