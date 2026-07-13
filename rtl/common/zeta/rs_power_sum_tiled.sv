// rs_power_sum_tiled: LANES parallel rs_power_sum lanes over a striped
// entry stream — L terms per cycle (M17).
//
// Term n (0-indexed entry i) belongs to lane i % LANES; lane l therefore
// consumes N_l = ceil((N - l) / LANES) entries from its own bank. Because
// the per-term accumulator is exact fixed point (Q27.FRAC), the lane
// partial sums merge by plain integer addition into the SAME value the
// single-lane engine produces — the wrapper is bit-exact against
// golden/rs_pipe.py with zero golden-model changes.
//
// Throughput: ceil(N/LANES) + drain cycles per sum. LANES must be a power
// of two (the engine banks its RS table by entry index low bits).
module rs_power_sum_tiled #(
    parameter int unsigned LANES = 2,
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
    parameter int unsigned EXP_TERMS = 22,
    parameter string CEXP_ROM = "cexp_w64.mem",
    parameter string EXP_ROM = "expln_w64_exp.mem",
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned MPW = WIDTH + EXPW + 3,
    localparam int unsigned BW = PHW + 32,
    localparam int unsigned FRAC = WIDTH + 16,
    localparam int unsigned ACCW = FRAC + 27
) (
    input  logic                     clk,
    input  logic                     rst_n,

    input  logic                     start_valid,
    output logic                     start_ready,
    input  logic [63:0]              t_fx,
    input  logic [23:0]              n_in,

    input  logic [LANES-1:0]         entry_valid,
    output logic [LANES-1:0]         entry_ready,
    input  logic [LANES*(BW+8)-1:0]  lnn2pi,
    input  logic [LANES*MPW-1:0]     amp,

    output logic                     out_valid,
    output logic [MPW-1:0]           s_re,
    output logic [MPW-1:0]           s_im
);

  typedef enum logic [1:0] { IDLE, RUN, NORM_R, NORM_I } state_e;
  state_e state;

  // per-lane stripe counts for the current sum
  logic [63:0] t_q;
  logic [23:0] n_q;
  logic [23:0] n_lane [LANES];

  always_comb begin
    for (int unsigned l = 0; l < LANES; l++) begin
      n_lane[l] = (32'(n_q) > l) ? 24'((32'(n_q) - 1 - l) / LANES + 1) : 24'd0;
    end
  end

  logic [LANES-1:0] lane_start, lane_done_v, done_mask;
  logic signed [ACCW-1:0] lane_acc_r [LANES];
  logic signed [ACCW-1:0] lane_acc_i [LANES];
  logic signed [ACCW-1:0] acc_r, acc_i;

  assign start_ready = (state == IDLE);

  // Lane s_re/s_im outputs unused: the wrapper normalizes the merged
  // accumulator once.
  /* verilator lint_off PINCONNECTEMPTY */
  for (genvar gl = 0; gl < int'(LANES); gl++) begin : g_lane
    rs_power_sum #(
        .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW), .TW(TW),
        .TILE_LATENCY(TILE_LATENCY), .PHW(PHW), .FG(FG), .CONSTW(CONSTW),
        .CTERMS(CTERMS), .SEGW(SEGW), .EXP_TERMS(EXP_TERMS),
        .CEXP_ROM(CEXP_ROM), .EXP_ROM(EXP_ROM)
    ) u_lane (
        .clk(clk), .rst_n(rst_n),
        .start_valid(lane_start[gl]), .start_ready(),
        .t_fx(t_q), .n_in(n_lane[gl]),
        .entry_valid(entry_valid[gl]), .entry_ready(entry_ready[gl]),
        .lnn2pi(lnn2pi[gl*(BW+8)+:BW+8]), .amp(amp[gl*MPW+:MPW]),
        .out_valid(lane_done_v[gl]),
        .s_re(), .s_im(),
        .acc_re_o(lane_acc_r[gl]), .acc_im_o(lane_acc_i[gl])
    );
  end
  /* verilator lint_on PINCONNECTEMPTY */

  // exact merge of the lane partial sums (order-independent fixed point)
  logic signed [ACCW-1:0] sum_r, sum_i;

  always_comb begin
    sum_r = '0;
    sum_i = '0;
    for (int unsigned l = 0; l < LANES; l++) begin
      if (n_lane[l] != 0) begin
        sum_r = sum_r + lane_acc_r[l];
        sum_i = sum_i + lane_acc_i[l];
      end
    end
  end

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
      state      <= IDLE;
      out_valid  <= 1'b0;
      lane_start <= '0;
    end else begin
      out_valid  <= 1'b0;
      lane_start <= '0;
      unique case (state)
        IDLE: begin
          if (start_valid) begin
            t_q <= t_fx;
            n_q <= n_in;
            // zero-stripe lanes are never started; mark them done up front
            for (int unsigned l = 0; l < LANES; l++) begin
              done_mask[l]  <= (32'(n_in) <= l);
              lane_start[l] <= (32'(n_in) > l);
            end
            state <= RUN;
          end
        end
        RUN: begin
          done_mask <= done_mask | lane_done_v;
          if (&(done_mask | lane_done_v)) begin
            acc_r <= sum_r;
            acc_i <= sum_i;
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

endmodule
