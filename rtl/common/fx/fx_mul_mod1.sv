// fx_mul_mod1: fixed-point multiply keeping only the fractional part, mod 1.
//
//   frac = top PHW fractional bits of frac(a * b), truncated.
//   a: unsigned fixed point, AW bits with AF fractional bits (the "t" operand).
//   b: unsigned fixed point, BI integer + BW fractional bits (the phase
//      constant ln(n)/2pi, FULL value: since a has fractional bits, the
//      integer part of b contributes to the fractional product and cannot
//      be dropped ahead of time).
//
// This is the NCO-style phase multiply at the heart of the n^(-s) kernel:
// the integer part of the product is discarded by the mod, so partial
// products lying entirely above the fractional window need not be computed.
// This generic implementation forms the full product and slices (bit-exact
// by construction); the DSP-tiled arch implementations realize the saving
// by not instantiating the high tiles. Combinational, like all mp/fx leaves;
// pipelining is applied on composition.
//
// Golden model: host/zetafpga/golden/fixedpt.py::fx_mul_mod1
module fx_mul_mod1 #(
    parameter int unsigned AW = 64,
    parameter int unsigned AF = 16,
    parameter int unsigned BW = 128,  // fractional bits of b
    parameter int unsigned BI = 0,    // integer bits of b
    parameter int unsigned PHW = 96,
    localparam int unsigned FB = AF + BW  // fractional bits of the raw product
) (
    input  logic [BI+BW-1:0] b,
    input  logic [AW-1:0]    a,
    output logic [PHW-1:0]   frac
);

  initial begin
    assert (FB >= PHW)
    else $fatal(1, "fx_mul_mod1: AF + BW (%0d) must be >= PHW (%0d)", FB, PHW);
  end

  // The unused high bits (integer part, discarded by mod 1) and low bits
  // (below the PHW window, truncated) are the design intent.
  // verilator lint_off UNUSEDSIGNAL
  logic [AW+BI+BW-1:0] product;
  // verilator lint_on UNUSEDSIGNAL

  assign product = a * b;
  assign frac    = product[FB-1-:PHW];

endmodule
