// log_mpf: ln(x) for MPF operands. Sequential FSM unit (not II=1).
//
// Algorithm (mirrored bit-for-bit by golden/expln.py::log_mpf):
//   x = mu * 2^e, mu in [0.5, 1):  ln x = e*ln2 + ln(mu).
//   Multiplicative normalization: for i = 1..FG, multiply mu by (1 + 2^-i)
//   (a shift-and-add) whenever the product stays <= 1, subtracting the ROM'd
//   ln(1+2^-i); the residual ln(v_final) is -(1-v) to within 2^-2FG.
//
// Domain: x > 0. Zero, negative, or special inputs return is_special.
// ln(1) = 0 exactly (special-cased). Accuracy: <= 2 ulp for |ln x| >= 2^-8;
// near x = 1 the absolute error is <= 2^-(WIDTH+12) but relative accuracy
// degrades (documented band — see DESIGN.md).
//
// Latency ~ FG + 5 cycles, one operand in flight.
// Constants from tools/coeffgen/gen_expln.py (committed .mem, per width).
module log_mpf #(
    parameter int unsigned LIMBS = 2,
    parameter int unsigned EXPW = 20,
    parameter int unsigned LIMBW = 64,
    parameter int unsigned FG = 152,
    parameter int unsigned CONSTW = 160,
    parameter string CONSTS_ROM = "expln_w128_exp.mem",  // for ln2 (entry 0)
    parameter int unsigned CONST_LINES = 34,  // lines in CONSTS_ROM (terms + 2)
    parameter string LN_ROM = "expln_w128_ln.mem",
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned MPW = WIDTH + EXPW + 3,
    localparam int unsigned XW = FG + 32
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

  // Full constants file is loaded; only entry 0 (ln2) is used here.
  // verilator lint_off UNUSEDSIGNAL
  logic signed [CONSTW-1:0] consts [0:CONST_LINES-1];
  // verilator lint_on UNUSEDSIGNAL
  logic signed [CONSTW-1:0] lntbl [0:FG-1];    // ln(1 + 2^-i), i = 1..FG

  initial begin
    $readmemh(CONSTS_ROM, consts);
    $readmemh(LN_ROM, lntbl);
  end

  typedef enum logic [2:0] { IDLE, LOOP, FIN, NORM, EMIT } state_e;
  state_e state;

  logic [FG:0] v;
  logic signed [XW-1:0] acc;
  logic [$clog2(FG+2)-1:0] i;
  logic [MPW-1:0] early_word;

  localparam logic [FG:0] ONE = {1'b1, {FG{1'b0}}};

  assign in_ready = (state == IDLE);

  logic [FG+1:0] trial;
  logic [FG:0] resid;
  assign trial = {1'b0, v} + {1'b0, (v >> i)};
  assign resid = ONE - v;  // 1 - v_final, the ln residual

  // Normalization datapath (combinational from acc).
  logic sign_res;
  logic [XW-1:0] a_mag;
  logic [$clog2(XW+1)-1:0] lz;
  logic a_zero;

  assign sign_res = acc < 0;
  assign a_mag    = sign_res ? unsigned'(-acc) : unsigned'(acc);

  lzc #(
      .WIDTH(XW)
  ) u_lzc (
      .data    (a_mag),
      .count   (lz),
      .all_zero(a_zero)
  );

  logic [WIDTH+1:0] t_norm;
  logic [WIDTH:0] mant_rnd;
  logic signed [23:0] e_norm, e_f;
  logic [WIDTH-1:0] mant_f;

  always_comb begin
    automatic int p = int'(XW) - 1 - int'(32'(lz));
    e_norm = 24'(p - int'(FG) + 1);
    if (p >= int'(WIDTH)) begin
      t_norm = (WIDTH+2)'(a_mag >> unsigned'(p - int'(WIDTH)));
    end else begin
      t_norm = (WIDTH+2)'(a_mag) << unsigned'(int'(WIDTH) - p);
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
          if (in_valid) begin
            if (x[SPEC_B] || x[ZERO_B] || x[SIGN_B]) begin
              early_word <= {1'b1, 1'b0, x[SIGN_B], {EXPW{1'b0}}, {WIDTH{1'b0}}};
              state      <= EMIT;
            end else if (signed'(x[EXP_LO+:EXPW]) == EXPW'(1)
                         && x[WIDTH-1:0] == {1'b1, {(WIDTH-1){1'b0}}}) begin
              // ln(1) = 0 exactly
              early_word <= {1'b0, 1'b1, 1'b0, {EXPW{1'b0}}, {WIDTH{1'b0}}};
              state      <= EMIT;
            end else begin
              v     <= {1'b0, x[WIDTH-1:0], {(FG - WIDTH) {1'b0}}};
              acc   <= XW'(signed'(x[EXP_LO+:EXPW]) * consts[0]);
              i     <= 'd1;
              state <= LOOP;
            end
          end
        end
        LOOP: begin
          if (trial <= {1'b0, ONE}) begin
            v   <= trial[FG:0];
            acc <= acc - XW'(lntbl[32'(i) - 1]);
          end
          if (32'(i) == FG) begin
            state <= FIN;
          end else begin
            i <= i + 1'b1;
          end
        end
        FIN: begin
          acc   <= acc - XW'(resid);
          state <= NORM;
        end
        NORM: begin
          ovf <= 1'b0;
          unf <= 1'b0;
          if (a_zero) begin
            result <= {1'b0, 1'b1, 1'b0, {EXPW{1'b0}}, {WIDTH{1'b0}}};
          end else if (e_f > 24'(EMAX)) begin
            result <= {1'b1, 1'b0, sign_res, {EXPW{1'b0}}, {WIDTH{1'b0}}};
            ovf    <= 1'b1;
          end else if (e_f < 24'(EMIN)) begin
            result <= {1'b0, 1'b1, sign_res, {EXPW{1'b0}}, {WIDTH{1'b0}}};
            unf    <= 1'b1;
          end else begin
            result <= {1'b0, 1'b0, sign_res, e_f[EXPW-1:0], mant_f};
          end
          out_valid <= 1'b1;
          state     <= IDLE;
        end
        EMIT: begin
          result    <= early_word;
          ovf       <= 1'b0;
          unf       <= 1'b0;
          out_valid <= 1'b1;
          state     <= IDLE;
        end
        default: state <= IDLE;
      endcase
    end
  end

endmodule
