// limb_addsub: wide add/subtract with carry/borrow, combinational.
//
//   sub = 0: {carry, result} = a + b + cin
//   sub = 1: result = a - b - cin, carry = borrow-out (1 iff a < b + cin)
//
// Implemented as a single two's-complement addition: for subtraction the
// second operand is inverted and the carry-in complemented, and the borrow
// is the complement of the internal carry-out.
// Pipelining is applied externally when composed (see DESIGN.md, M1 decision).
// Golden model: host/zetafpga/golden/limb.py::addsub
module limb_addsub #(
    parameter int unsigned LIMBS = 2,
    parameter int unsigned LIMBW = 64,
    localparam int unsigned WIDTH = LIMBS * LIMBW
) (
    input  logic             sub,
    input  logic             cin,
    input  logic [WIDTH-1:0] a,
    input  logic [WIDTH-1:0] b,
    output logic [WIDTH-1:0] result,
    output logic             carry
);

  logic [WIDTH:0] full;
  logic [WIDTH-1:0] b_eff;
  logic cin_eff;

  assign b_eff   = sub ? ~b : b;
  assign cin_eff = sub ? ~cin : cin;
  assign full    = {1'b0, a} + {1'b0, b_eff} + {{WIDTH{1'b0}}, cin_eff};
  assign result  = full[WIDTH-1:0];
  assign carry   = sub ? ~full[WIDTH] : full[WIDTH];

endmodule
