// exp_mpf: e^y for MPF operands. Sequential FSM unit (not II=1).
//
// Algorithm (mirrored bit-for-bit by golden/expln.py::exp_mpf):
//   1. Saturate |y| beyond the exponent range (y.exp > 21).
//   2. Convert y to signed fixed point at scale 2^FG (exact for representable y).
//   3. Range-reduce: k = round(y / ln2), r = y - k*ln2, |r| <= ln2/2.
//   4. Taylor-Horner: e^r = sum_{j<TERMS} r^j/j! with ROM'd 1/j! constants.
//   5. Normalize e^r in [0.70, 1.42], round to W+1 bits, exponent k adjust,
//      saturate, pack.
//
// Latency ~ TERMS + 6 cycles, one operand in flight (in_ready handshake).
// Throughput is a documented M5 non-goal: the E-M engine instantiates
// width-reduced copies where rate matters; revisit at M7 (see DESIGN.md).
//
// Constants from tools/coeffgen/gen_expln.py (committed .mem, per width).
module exp_mpf #(
    parameter int unsigned LIMBS = 2,
    parameter int unsigned EXPW = 20,
    parameter int unsigned LIMBW = 64,
    parameter int unsigned FG = 152,     // fractional working bits (width + 24)
    parameter int unsigned CONSTW = 160, // ROM entry width (FG + 8)
    parameter int unsigned TERMS = 32,   // Taylor terms (from expln_w*.json)
    parameter string CONSTS_ROM = "expln_w128_exp.mem",
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned MPW = WIDTH + EXPW + 3,
    localparam int unsigned XW = FG + 32  // signed working width
) (
    input  logic           clk,
    input  logic           rst_n,
    input  logic           in_valid,
    output logic           in_ready,
    input  logic [MPW-1:0] x,
    // Fused fixed-point path (npow_s_kernel): when fx_mode is set with
    // in_valid, y is taken directly from yfx_in (scale 2^FG) and x is
    // ignored — avoids W-bit MPF quantization of intermediate products.
    input  logic           fx_mode,
    input  logic signed [XW-1:0] yfx_in,
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

  logic signed [CONSTW-1:0] consts [0:TERMS+1];  // [0]=ln2 [1]=1/ln2 [2+j]=1/j!

  initial begin
    $readmemh(CONSTS_ROM, consts);
  end

  typedef enum logic [2:0] { IDLE, MULK, REDK, HORNER, NORM, EMIT } state_e;
  state_e state;

  logic signed [XW-1:0] yfx, r, acc;
  logic signed [23:0] kreg;
  logic [$clog2(TERMS+1)-1:0] cnt;
  logic [MPW-1:0] early_word;
  logic early_ovf, early_unf, take_early;

  assign in_ready = (state == IDLE);

  // Wide products, combinational per state.
  logic signed [XW+CONSTW-1:0] p_mulk;
  // k*ln2 fits far below the top of the product; upper bits unused by design.
  // verilator lint_off UNUSEDSIGNAL
  logic signed [XW+CONSTW-1:0] p_kln2;
  // verilator lint_on UNUSEDSIGNAL
  logic signed [2*XW-1:0] p_horner;

  assign p_mulk   = yfx * consts[1];
  assign p_kln2   = kreg * consts[0];
  assign p_horner = acc * r;

  // Rounding for k = round(p / 2^(2FG)): floor((floor(p/2^(2FG-1)) + 1)/2).
  logic signed [XW+CONSTW-1:0] p_half;
  assign p_half = p_mulk >>> (2 * FG - 1);

  // Final normalize/round (combinational from acc/kreg).
  logic acc_ge_one;
  logic [WIDTH+1:0] t_norm;
  logic [WIDTH:0] mant_rnd;
  logic signed [23:0] e_norm;
  logic [WIDTH-1:0] mant_f;
  logic signed [23:0] e_f;

  always_comb begin
    acc_ge_one = acc >= (XW'(1) <<< FG);
    if (acc_ge_one) begin
      t_norm = (WIDTH+2)'(acc >>> (FG - WIDTH));
      e_norm = kreg + 24'sd1;
    end else begin
      t_norm = (WIDTH+2)'(acc >>> (FG - WIDTH - 1));
      e_norm = kreg;
    end
    mant_rnd = (WIDTH+1)'((t_norm + (WIDTH+2)'(1)) >> 1);
    if (mant_rnd[WIDTH]) begin
      mant_f = mant_rnd[WIDTH:1];
      e_f    = e_norm + 24'sd1;
    end else begin
      mant_f = mant_rnd[WIDTH-1:0];
      e_f    = e_norm;
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= IDLE;
      out_valid <= 1'b0;
    end else begin
      out_valid <= 1'b0;
      unique case (state)
        IDLE: begin
          if (in_valid && fx_mode) begin
            take_early <= 1'b0;
            early_ovf  <= 1'b0;
            early_unf  <= 1'b0;
            if (yfx_in == 0) begin
              take_early <= 1'b1;  // e^0 = 1.0
              early_word <= {1'b0, 1'b0, 1'b0, EXPW'(1), {1'b1, {(WIDTH-1){1'b0}}}};
              state      <= EMIT;
            end else if (yfx_in >= (XW'(1) <<< (FG + 21))) begin
              take_early <= 1'b1;
              early_word <= {1'b1, 1'b0, 1'b0, {EXPW{1'b0}}, {WIDTH{1'b0}}};
              early_ovf  <= 1'b1;
              state      <= EMIT;
            end else if (yfx_in <= -(XW'(1) <<< (FG + 21))) begin
              take_early <= 1'b1;
              early_word <= {1'b0, 1'b1, 1'b0, {EXPW{1'b0}}, {WIDTH{1'b0}}};
              early_unf  <= 1'b1;
              state      <= EMIT;
            end else begin
              yfx   <= yfx_in;
              state <= MULK;
            end
          end else if (in_valid) begin
            take_early <= 1'b0;
            early_ovf  <= 1'b0;
            early_unf  <= 1'b0;
            if (x[SPEC_B]) begin
              take_early <= 1'b1;
              early_word <= {1'b1, 1'b0, x[SIGN_B], {EXPW{1'b0}}, {WIDTH{1'b0}}};
              state      <= EMIT;
            end else if (x[ZERO_B]) begin
              take_early <= 1'b1;  // e^0 = 1.0
              early_word <= {1'b0, 1'b0, 1'b0, EXPW'(1), {1'b1, {(WIDTH-1){1'b0}}}};
              state      <= EMIT;
            end else if (signed'(x[EXP_LO+:EXPW]) > EXPW'(21)) begin
              take_early <= 1'b1;
              if (x[SIGN_B]) begin
                early_word <= {1'b0, 1'b1, 1'b0, {EXPW{1'b0}}, {WIDTH{1'b0}}};
                early_unf  <= 1'b1;
              end else begin
                early_word <= {1'b1, 1'b0, 1'b0, {EXPW{1'b0}}, {WIDTH{1'b0}}};
                early_ovf  <= 1'b1;
              end
              state <= EMIT;
            end else begin
              automatic int sh = int'(FG) + int'(signed'(x[EXP_LO+:EXPW])) - int'(WIDTH);
              automatic logic [XW-1:0] mag;
              if (sh >= 0) begin
                mag = XW'(x[WIDTH-1:0]) << unsigned'(sh);
              end else if (-sh < int'(WIDTH)) begin
                mag = XW'(x[WIDTH-1:0]) >> unsigned'(-sh);
              end else begin
                mag = '0;
              end
              yfx   <= x[SIGN_B] ? -signed'(mag) : signed'(mag);
              state <= MULK;
            end
          end
        end
        MULK: begin
          kreg  <= 24'((p_half + 1) >>> 1);
          state <= REDK;
        end
        REDK: begin
          r     <= yfx - XW'(p_kln2);
          acc   <= XW'(consts[TERMS + 1]);  // invfact[TERMS-1]
          cnt   <= ($clog2(TERMS+1))'(TERMS - 2);
          state <= HORNER;
        end
        HORNER: begin
          acc <= XW'(p_horner >>> FG) + XW'(consts[32'(cnt) + 2]);
          if (cnt == 0) begin
            state <= NORM;
          end else begin
            cnt <= cnt - 1'b1;
          end
        end
        NORM: begin
          ovf <= 1'b0;
          unf <= 1'b0;
          if (e_f > 24'(EMAX)) begin
            result <= {1'b1, 1'b0, 1'b0, {EXPW{1'b0}}, {WIDTH{1'b0}}};
            ovf    <= 1'b1;
          end else if (e_f < 24'(EMIN)) begin
            result <= {1'b0, 1'b1, 1'b0, {EXPW{1'b0}}, {WIDTH{1'b0}}};
            unf    <= 1'b1;
          end else begin
            result <= {1'b0, 1'b0, 1'b0, e_f[EXPW-1:0], mant_f};
          end
          out_valid <= 1'b1;
          state     <= IDLE;
        end
        EMIT: begin
          result    <= early_word;
          ovf       <= early_ovf;
          unf       <= early_unf;
          out_valid <= 1'b1;
          state     <= IDLE;
        end
        default: state <= IDLE;
      endcase
    end
  end

  // take_early is a documentation register (EMIT implies it); keep lint quiet.
  // verilator lint_off UNUSEDSIGNAL
  logic unused_take_early;
  assign unused_take_early = take_early;
  // verilator lint_on UNUSEDSIGNAL

endmodule
