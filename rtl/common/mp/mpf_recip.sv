// mpf_recip: y = 1/x by Newton iteration — the engine's divider-free
// reciprocal (M16). Sequential FSM unit (not II=1).
//
// Algorithm (mirrored bit-for-bit by golden/recip.py):
//   1. Specials: 1/special = special; 1/0 = special + ovf.
//   2. Seed on the mantissa m in [0.5, 1):  y0 = 48/17 - 32/17 * m
//      (classical minimax line, |rel err| <= 1/17 ~ 2^-4.09), fixed point
//      at F = WIDTH + 8 guard bits.
//   3. NITER = clog2(WIDTH) Newton steps  y <- y * (2 - m*y), floor
//      truncations (error doubles in accuracy per step; over-converged
//      past 2^-(WIDTH+2)).
//   4. One RNE to WIDTH bits; result exponent 1 - x.exp (+carry), saturate.
//
// Accuracy: <= 2 ulp (exact for powers of two). Latency ~ 2*NITER + 4
// cycles, one operand in flight. The wide products are behavioral (`*`)
// like the other fn/ FSM units; DSP tiling arrives with synthesis targets.
module mpf_recip #(
    parameter int unsigned LIMBS = 2,
    parameter int unsigned EXPW = 20,
    parameter int unsigned LIMBW = 64,
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned MPW = WIDTH + EXPW + 3,
    localparam int unsigned F = WIDTH + 8,
    localparam int unsigned YW = F + 2  // y in (1, 2] at scale 2^F
) (
    input  logic           clk,
    input  logic           rst_n,
    input  logic           in_valid,
    output logic           in_ready,
    input  logic [MPW-1:0] x,
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

  localparam int unsigned NITER = $clog2(WIDTH);
  localparam logic [YW-1:0] K48 = YW'(((YW + 8)'(48) << F) / (YW + 8)'(17));
  localparam logic [YW-1:0] K32 = YW'(((YW + 8)'(32) << F) / (YW + 8)'(17));

  typedef enum logic [2:0] { IDLE, SEED, NEWA, NEWB, NORM, EMIT } state_e;
  state_e state;

  logic [WIDTH-1:0] m;
  logic [YW-1:0] y, e;
  logic sign_q, spec_q, zero_q;
  logic signed [EXPW-1:0] exp_q;
  logic [$clog2(NITER+1)-1:0] iter;

  assign in_ready = (state == IDLE);

  // Wide products, combinational per state (behavioral; see header).
  logic [WIDTH+YW-1:0] p_seed, p_my;
  logic [2*YW-1:0] p_ye;

  assign p_seed = K32 * m;
  assign p_my   = m * y;
  assign p_ye   = y * e;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= IDLE;
      out_valid <= 1'b0;
    end else begin
      out_valid <= 1'b0;
      unique case (state)
        IDLE: begin
          if (in_valid) begin
            m      <= x[WIDTH-1:0];
            sign_q <= x[SIGN_B];
            spec_q <= x[SPEC_B];
            zero_q <= x[ZERO_B];
            exp_q  <= signed'(x[EXP_LO+:EXPW]);
            iter   <= '0;
            state  <= SEED;
          end
        end
        SEED: begin
          if (spec_q || zero_q) begin
            result    <= {1'b1, 1'b0, sign_q, {EXPW{1'b0}}, {WIDTH{1'b0}}};
            ovf       <= zero_q;  // 1/0 overflows
            unf       <= 1'b0;
            out_valid <= 1'b1;
            state     <= IDLE;
          end else begin
            y     <= K48 - YW'(p_seed >> WIDTH);
            state <= NEWA;
          end
        end
        NEWA: begin
          e     <= (YW'(2) << F) - YW'(p_my >> WIDTH);
          state <= NEWB;
        end
        NEWB: begin
          y <= YW'(p_ye >> F);
          if (32'(iter) == NITER - 1) begin
            state <= NORM;
          end else begin
            iter  <= iter + 1'b1;
            state <= NEWA;
          end
        end
        NORM: begin
          automatic logic [WIDTH+1:0] t;
          automatic logic [WIDTH:0] mant;
          automatic int er;
          t    = (WIDTH + 2)'(y >> (F - WIDTH));
          mant = (WIDTH + 1)'((t + 1) >> 1);
          er   = 1 - int'(exp_q);
          if (mant[WIDTH]) begin
            mant = mant >> 1;
            er   = er + 1;
          end
          if (er > EMAX) begin
            result <= {1'b1, 1'b0, sign_q, {EXPW{1'b0}}, {WIDTH{1'b0}}};
            ovf    <= 1'b1;
            unf    <= 1'b0;
          end else if (er < EMIN) begin
            result <= {1'b0, 1'b1, sign_q, {EXPW{1'b0}}, {WIDTH{1'b0}}};
            ovf    <= 1'b0;
            unf    <= 1'b1;
          end else begin
            result <= {1'b0, 1'b0, sign_q, EXPW'(er), mant[WIDTH-1:0]};
            ovf    <= 1'b0;
            unf    <= 1'b0;
          end
          state <= EMIT;
        end
        EMIT: begin
          out_valid <= 1'b1;
          state     <= IDLE;
        end
        default: state <= IDLE;
      endcase
    end
  end

endmodule
