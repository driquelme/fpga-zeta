// limb_mul: full-width schoolbook multiplier, fully pipelined (II = 1).
//
//   product = a * b, exact (WIDTH x WIDTH -> 2*WIDTH bits).
//
// Structure: the operands are sliced into NT = WIDTH/TW tiles; NT*NT mul_tile
// instances compute all partial products in parallel (TILE_LATENCY cycles),
// which are then summed by a registered binary reduction tree
// ($clog2(NT*NT) further cycles). Every stage is registered feed-forward
// logic, so a new operand pair can enter every cycle.
//
//   LATENCY = TILE_LATENCY + $clog2(NT*NT)
//
// Golden model: host/zetafpga/golden/limb.py::mul
module limb_mul #(
    parameter int unsigned LIMBS = 2,
    parameter int unsigned LIMBW = 64,
    parameter int unsigned TW = 32,
    parameter int unsigned TILE_LATENCY = 2,
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned NT = WIDTH / TW,
    localparam int unsigned NPP = NT * NT,
    localparam int unsigned TREE_LEVELS = $clog2(NPP),
    localparam int unsigned LATENCY = TILE_LATENCY + TREE_LEVELS
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               in_valid,
    input  logic [WIDTH-1:0]   a,
    input  logic [WIDTH-1:0]   b,
    output logic               out_valid,
    output logic [2*WIDTH-1:0] product
);

  // Partial products over the tile abstraction.
  logic [2*TW-1:0] pp [0:NPP-1];

  for (genvar gi = 0; gi < int'(NT); gi++) begin : gen_row
    for (genvar gj = 0; gj < int'(NT); gj++) begin : gen_col
      mul_tile #(
          .TW     (TW),
          .LATENCY(TILE_LATENCY)
      ) u_tile (
          .clk(clk),
          .a  (a[gi*TW+:TW]),
          .b  (b[gj*TW+:TW]),
          .p  (pp[gi*int'(NT)+gj])
      );
    end
  end

  // Registered binary reduction tree. lvl[0] holds the shifted partial
  // products (combinational from the tile outputs); each subsequent level
  // halves the term count with one register stage.
  logic [2*WIDTH-1:0] lvl [0:TREE_LEVELS][0:NPP-1];

  for (genvar gi = 0; gi < int'(NT); gi++) begin : gen_t_row
    for (genvar gj = 0; gj < int'(NT); gj++) begin : gen_t_col
      assign lvl[0][gi*int'(NT)+gj] =
          {{(2*WIDTH - 2*TW){1'b0}}, pp[gi*int'(NT)+gj]} << (TW * (gi + gj));
    end
  end

  for (genvar gl = 1; gl <= int'(TREE_LEVELS); gl++) begin : gen_level
    localparam int unsigned CNT = NPP >> gl;
    for (genvar gk = 0; gk < int'(CNT); gk++) begin : gen_pair
      always_ff @(posedge clk) begin
        lvl[gl][gk] <= lvl[gl-1][2*gk] + lvl[gl-1][2*gk+1];
      end
    end
    for (genvar gk = int'(CNT); gk < int'(NPP); gk++) begin : gen_tie
      assign lvl[gl][gk] = '0;
    end
  end

  assign product = lvl[TREE_LEVELS][0];

  // Validity travels alongside the data pipeline.
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
