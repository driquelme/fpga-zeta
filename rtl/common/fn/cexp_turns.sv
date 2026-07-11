// cexp_turns: full-precision e^(2*pi*i*phi), phi in turns. Sequential FSM.
//
//   (cos_o, sin_o) = cos/sin(2*pi * phi/2^PHW) as MPF words.
//
// Decomposition (mirrored bit-for-bit by golden/cexp.py):
//   phi = phi_hi (top SEGW bits) + phi_lo
//   e^(2*pi*i*phi) = table[phi_hi] * e^(i*z),  z = 2*pi*phi_lo < 2*pi*2^-SEGW
// with the residual factor from a complex Taylor-Horner over the shared
// 1/k! constants (expln ROM) and one complex multiply, all at FG fractional
// working bits. Unlike sincos_turns (64-bit table+poly fast path), this unit
// delivers full mantissa precision at any LIMBS — it is the phase factor of
// the n^(-s) kernel.
//
// Latency ~ TERMS + 7 cycles, one operand in flight.
module cexp_turns #(
    parameter int unsigned LIMBS = 2,
    parameter int unsigned EXPW = 20,
    parameter int unsigned LIMBW = 64,
    parameter int unsigned PHW = 160,
    parameter int unsigned FG = 152,
    parameter int unsigned CONSTW = 160,
    parameter int unsigned SEGW = 10,
    parameter int unsigned TERMS = 17,        // cexp taylor terms (cexp_w*.json)
    parameter int unsigned CONST_LINES = 34,  // lines in EXP_ROM (exp terms + 2)
    parameter string CEXP_ROM = "cexp_w128.mem",
    parameter string EXP_ROM = "expln_w128_exp.mem",
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned MPW = WIDTH + EXPW + 3,
    localparam int unsigned XW = FG + 32,
    localparam int unsigned NTBL = 2 * (1 << SEGW) + 1  // cos/sin pairs + 2*pi
) (
    input  logic           clk,
    input  logic           rst_n,
    input  logic           in_valid,
    output logic           in_ready,
    input  logic [PHW-1:0] phi,
    output logic           out_valid,
    output logic [MPW-1:0] cos_o,
    output logic [MPW-1:0] sin_o
);

  localparam int unsigned LOW = PHW - SEGW;  // phi_lo width

  logic signed [CONSTW-1:0] tbl [0:NTBL-1];
  // Full expln constants file; only the 1/k! entries (2..) are used here.
  // verilator lint_off UNUSEDSIGNAL
  logic signed [CONSTW-1:0] econsts [0:CONST_LINES-1];
  // verilator lint_on UNUSEDSIGNAL

  initial begin
    $readmemh(CEXP_ROM, tbl);
    $readmemh(EXP_ROM, econsts);
  end

  typedef enum logic [2:0] { IDLE, ZC, HORNER, TMUL, NORMC, NORMS, EMIT } state_e;
  state_e state;

  logic [SEGW-1:0] hi;
  logic [LOW-1:0] lo;
  logic signed [XW-1:0] z, are, aim, ore, oim;
  logic [$clog2(TERMS+1)-1:0] cnt;

  assign in_ready = (state == IDLE);

  // z = lo * 2pi: lo at scale 2^PHW, 2pi at scale 2^(FG-3) -> >> (PHW-3).
  logic [LOW+CONSTW-1:0] p_z;
  assign p_z = lo * unsigned'(tbl[NTBL-1]);

  // Complex Horner step products.
  logic signed [2*XW-1:0] p_zre, p_zim;
  assign p_zre = z * are;
  assign p_zim = z * aim;

  // Final complex multiply by the table factor.
  logic signed [XW+CONSTW-1:0] p_tc_re, p_ts_im, p_tc_im, p_ts_re;
  assign p_tc_re = tbl[32'(hi)*2] * are;
  assign p_ts_im = tbl[32'(hi)*2+1] * aim;
  assign p_tc_im = tbl[32'(hi)*2] * aim;
  assign p_ts_re = tbl[32'(hi)*2+1] * are;

  // Shared MPF normalizer (same structure as log_mpf's NORM).
  logic signed [XW-1:0] norm_in;
  assign norm_in = (state == NORMC) ? ore : oim;

  logic n_sign;
  logic [XW-1:0] n_mag;
  logic [$clog2(XW+1)-1:0] lz;
  logic n_zero;

  assign n_sign = norm_in < 0;
  assign n_mag  = n_sign ? unsigned'(-norm_in) : unsigned'(norm_in);

  lzc #(
      .WIDTH(XW)
  ) u_lzc (
      .data    (n_mag),
      .count   (lz),
      .all_zero(n_zero)
  );

  logic [WIDTH+1:0] t_norm;
  logic [WIDTH:0] mant_rnd;
  // |values| <= 1: the exponent never approaches saturation, top bits unused.
  // verilator lint_off UNUSEDSIGNAL
  logic signed [23:0] e_norm, e_f;
  // verilator lint_on UNUSEDSIGNAL
  logic [WIDTH-1:0] mant_f;
  logic [MPW-1:0] norm_word;

  always_comb begin
    automatic int p = int'(XW) - 1 - int'(32'(lz));
    e_norm = 24'(p - int'(FG) + 1);
    if (p >= int'(WIDTH)) begin
      t_norm = (WIDTH+2)'(n_mag >> unsigned'(p - int'(WIDTH)));
    end else begin
      t_norm = (WIDTH+2)'(n_mag) << unsigned'(int'(WIDTH) - p);
    end
    mant_rnd = (WIDTH+1)'((t_norm + (WIDTH+2)'(1)) >> 1);
    if (mant_rnd[WIDTH]) begin
      mant_f = mant_rnd[WIDTH:1];
      e_f    = e_norm + 24'sd1;
    end else begin
      mant_f = mant_rnd[WIDTH-1:0];
      e_f    = e_norm;
    end
    if (n_zero) begin
      norm_word = {1'b0, 1'b1, 1'b0, {EXPW{1'b0}}, {WIDTH{1'b0}}};
    end else begin
      norm_word = {1'b0, 1'b0, n_sign, e_f[EXPW-1:0], mant_f};
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
            hi    <= phi[PHW-1-:SEGW];
            lo    <= phi[LOW-1:0];
            state <= ZC;
          end
        end
        ZC: begin
          z     <= XW'(p_z >> (PHW - 3));
          are   <= XW'(econsts[TERMS + 1]);  // invfact[TERMS-1]
          aim   <= '0;
          cnt   <= ($clog2(TERMS+1))'(TERMS - 2);
          state <= HORNER;
        end
        HORNER: begin
          are <= XW'(econsts[32'(cnt) + 2]) - XW'(p_zim >>> FG);
          aim <= XW'(p_zre >>> FG);
          if (cnt == 0) begin
            state <= TMUL;
          end else begin
            cnt <= cnt - 1'b1;
          end
        end
        TMUL: begin
          ore   <= XW'((p_tc_re - p_ts_im) >>> FG);
          oim   <= XW'((p_tc_im + p_ts_re) >>> FG);
          state <= NORMC;
        end
        NORMC: begin
          cos_o <= norm_word;
          state <= NORMS;
        end
        NORMS: begin
          sin_o <= norm_word;
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
