// mul_tile (generic-sim): behavioral multiplier tile.
//
// The portability boundary for multiplication: rtl/common code only ever
// instantiates mul_tile; each rtl/arch/<target> provides an implementation
// (this one is plain `*` for simulation / any-FPGA correctness; the
// xilinx-usplus version will tile onto DSP48E2 primitives).
//
// Fully pipelined, II=1, fixed LATENCY cycles from inputs to p. No
// valid/reset — callers track validity (see limb_mul's valid shift register).
module mul_tile #(
    parameter int unsigned TW = 32,
    parameter int unsigned LATENCY = 2
) (
    input  logic              clk,
    input  logic [TW-1:0]     a,
    input  logic [TW-1:0]     b,
    output logic [2*TW-1:0]   p
);

  logic [2*TW-1:0] pipe [0:LATENCY-1];

  always_ff @(posedge clk) begin
    pipe[0] <= a * b;
    for (int unsigned i = 1; i < LATENCY; i++) begin
      pipe[i] <= pipe[i-1];
    end
  end

  assign p = pipe[LATENCY-1];

endmodule
