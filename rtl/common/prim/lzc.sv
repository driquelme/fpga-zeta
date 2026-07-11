// lzc: leading-zero count, combinational.
//
// count == WIDTH when data == 0 (hence the count port is $clog2(WIDTH+1) wide).
// Golden model: host/zetafpga/golden/limb.py::lzc
module lzc #(
    parameter int unsigned WIDTH = 64
) (
    input  logic [WIDTH-1:0]           data,
    output logic [$clog2(WIDTH+1)-1:0] count,
    output logic                       all_zero
);

  localparam int unsigned COUNTW = $clog2(WIDTH + 1);

  always_comb begin
    count = COUNTW'(WIDTH);
    // Priority scan from LSB up: the last (highest) set bit wins.
    for (int unsigned i = 0; i < WIDTH; i++) begin
      if (data[i]) begin
        count = COUNTW'(WIDTH - 1 - i);
      end
    end
  end

  assign all_zero = (data == '0);

endmodule
