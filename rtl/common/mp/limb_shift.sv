// limb_shift: wide logical barrel shift with lost-bits detection, combinational.
//
//   left = 0: result = a >> amount     left = 1: result = a << amount
//   lost = OR of every bit shifted out of the WIDTH-bit window.
//   amount saturates at WIDTH: larger values behave as a full shift
//   (result = 0, lost = |a|). Without saturation the double-width trick
//   would drop bits past the 2*WIDTH window and under-report `lost`.
//
// The double-width intermediate keeps result and shifted-out bits in one
// vector: left shift exposes the lost bits above WIDTH, right shift below.
// Golden model: host/zetafpga/golden/limb.py::shift
module limb_shift #(
    parameter int unsigned LIMBS = 2,
    parameter int unsigned LIMBW = 64,
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned AMTW  = $clog2(WIDTH + 1)
) (
    input  logic             left,
    input  logic [AMTW-1:0]  amount,
    input  logic [WIDTH-1:0] a,
    output logic [WIDTH-1:0] result,
    output logic             lost
);

  logic [2*WIDTH-1:0] ext;
  logic [AMTW-1:0] amt_eff;

  assign amt_eff = (amount > AMTW'(WIDTH)) ? AMTW'(WIDTH) : amount;

  always_comb begin
    if (left) begin
      ext    = {{WIDTH{1'b0}}, a} << amt_eff;
      result = ext[WIDTH-1:0];
      lost   = |ext[2*WIDTH-1:WIDTH];
    end else begin
      ext    = {a, {WIDTH{1'b0}}} >> amt_eff;
      result = ext[2*WIDTH-1:WIDTH];
      lost   = |ext[WIDTH-1:0];
    end
  end

endmodule
