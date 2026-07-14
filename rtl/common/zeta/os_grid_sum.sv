// os_grid_sum: on-chip Odlyzko-Schonhage grid main sum (M21, COMPUTE_OS v1).
//
//   S(t_j) = sum_{n<=N} amp_n e^(-i t_j ln n),  t_j = t0 + j*dt,  j < J
//
// One sweep over the RS table bins each tone (anchor c_n at the grid center
// from fx_mul_mod1 + cexp_turns + amp muls, Q9.54; rate offset eps-hat =
// eps*M in bin units, Q1.62) into P+1 = 15 bin arrays; fft_radix2 runs 15
// times in place; a per-point complex Horner in u = j' 2pi/M (|u| <= pi/4
// by the J <= M/4 contract) with ROM'd 1/p! combines the spectra; results
// normalize to MPF via rs_acc_norm at FRAC = 54.
//
// Contracts (host side): constant N across the batch (slice at N
// boundaries), J <= M/4, table l1 < 2^9. Hunting-grade (double-quality)
// output — candidates are polished by COMPUTE_Z.
// Golden model: host/zetafpga/golden/os_pipe.py (bit-exact).
module os_grid_sum #(
    parameter int unsigned LIMBS = 1,
    parameter int unsigned EXPW = 20,
    parameter int unsigned LIMBW = 64,
    parameter int unsigned PHW = 96,
    parameter int unsigned FG = 88,
    parameter int unsigned CONSTW = 96,
    parameter int unsigned SEGW = 10,
    parameter int unsigned CTERMS = 11,
    parameter int unsigned EXP_TERMS = 22,
    parameter int unsigned TW = 32,
    parameter int unsigned TILE_LATENCY = 2,
    parameter string CEXP_ROM = "cexp_w64.mem",
    parameter string EXP_ROM = "expln_w64_exp.mem",
    parameter int unsigned M = 128,
    parameter string FFT_ROM = "fft_m128.mem",
    parameter string OS_ROM = "fft_os.mem",
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned MPW = WIDTH + EXPW + 3,
    localparam int unsigned BW = PHW + 32,
    localparam int unsigned LOG2M = $clog2(M),
    localparam int unsigned P = 14,
    localparam int unsigned DW = 64,
    localparam int unsigned SHIFT = PHW - LOG2M  // eps split point (> 62)
) (
    input  logic            clk,
    input  logic            rst_n,

    input  logic            start_valid,
    output logic            start_ready,
    input  logic [63:0]     t0_fx,
    input  logic [63:0]     dt_fx,
    input  logic [23:0]     n_in,
    input  logic [23:0]     j_in,

    input  logic            entry_valid,
    output logic            entry_ready,
    input  logic [BW+7:0]   lnn2pi,
    input  logic [MPW-1:0]  amp,

    output logic            point_valid,  // one pulse per grid point, in order
    output logic [MPW-1:0]  s_re,
    output logic [MPW-1:0]  s_im,
    output logic            done
);

  localparam int unsigned ZERO_B = WIDTH + EXPW + 1;
  localparam int unsigned SIGN_B = WIDTH + EXPW;
  localparam int unsigned EXP_LO = WIDTH;

  // ---- constants ---------------------------------------------------------------
  logic [63:0] osrom [0:P+1];  // 0..14: 1/p! Q2.62; 15: 2*pi at 2^60

  initial begin
    $readmemh(OS_ROM, osrom);
  end

  // ---- bin / spectrum storage ----------------------------------------------------
  logic [2*DW-1:0] binram [0:(P+1)*M-1];

  // ---- phase paths ---------------------------------------------------------------
  logic [63:0] tc_q, dt_q;
  logic [23:0] n_q, j_q, ectr, jctr;

  // both phase products are taken from the LIVE entry inputs (valid during
  // the SPH capture cycle) — never from a same-cycle register
  logic [PHW-1:0] phi, nu_raw;

  fx_mul_mod1 #(
      .AW(64), .AF(32), .BW(BW), .BI(8), .PHW(PHW)
  ) u_phic (
      .a(tc_q), .b(lnn2pi), .frac(phi)
  );

  fx_mul_mod1 #(
      .AW(64), .AF(32), .BW(BW), .BI(8), .PHW(PHW)
  ) u_phid (
      .a(dt_q), .b(lnn2pi), .frac(nu_raw)
  );

  // rate split: nu = -nu_raw mod 1; k = round(nu*M); eps-hat at Q1.62
  logic [PHW-1:0] nu_neg;
  logic [LOG2M:0] kfull;  // may reach M when nu rounds up to 1.0 (wraps to bin 0)
  logic [LOG2M-1:0] k_c;
  logic signed [PHW+1:0] eh;
  logic signed [DW-1:0] e62_c;

  assign nu_neg = ~nu_raw + PHW'(1);
  assign kfull  = (LOG2M + 1)'(((PHW + 1)'(nu_neg) + ((PHW + 1)'(1) << (SHIFT - 1))) >> SHIFT);
  assign k_c    = kfull[LOG2M-1:0];
  assign eh     = signed'({2'b00, nu_neg}) - (signed'((PHW + 2)'(kfull)) <<< SHIFT);
  assign e62_c  = DW'(eh >>> (SHIFT - 62));

  // ---- cexp + amplitude muls ------------------------------------------------------
  logic ce_go, ce_ready, ce_done;
  logic [PHW-1:0] phi_q;
  logic [MPW-1:0] ce_cos, ce_sin;

  cexp_turns #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW), .PHW(PHW), .FG(FG),
      .CONSTW(CONSTW), .SEGW(SEGW), .TERMS(CTERMS),
      .CONST_LINES(EXP_TERMS + 2), .CEXP_ROM(CEXP_ROM), .EXP_ROM(EXP_ROM)
  ) u_cexp (
      .clk(clk), .rst_n(rst_n),
      .in_valid(ce_go), .in_ready(ce_ready), .phi(phi_q),
      .out_valid(ce_done), .cos_o(ce_cos), .sin_o(ce_sin)
  );

  logic mul_go, mul_done;
  logic [MPW-1:0] mul_x, mul_y, mul_out;

  /* verilator lint_off PINCONNECTEMPTY */
  mpf_mul #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW), .TW(TW), .TILE_LATENCY(TILE_LATENCY)
  ) u_mul (
      .clk(clk), .rst_n(rst_n), .in_valid(mul_go),
      .x(mul_x), .y(mul_y),
      .out_valid(mul_done), .result(mul_out), .ovf(), .unf()
  );
  /* verilator lint_on PINCONNECTEMPTY */

  // MPF -> Q9.54 (truncate toward zero; |v| <= 1)
  function automatic logic signed [DW-1:0] to_fx54(input logic [MPW-1:0] v);
    logic signed [EXPW-1:0] e;
    // only the low DW bits survive (|v| <= 1 fits Q9.54); the wide
    // intermediate exists for the left-shift case at LIMBS > 1
    // verilator lint_off UNUSEDSIGNAL
    logic [WIDTH+63:0] mag;
    // verilator lint_on UNUSEDSIGNAL
    int sh;
    if (v[ZERO_B]) return '0;
    e  = signed'(v[EXP_LO+:EXPW]);
    sh = 54 + int'(e) - int'(WIDTH);
    if (sh >= 0) begin
      mag = (WIDTH + 64)'(v[WIDTH-1:0]) << unsigned'(sh);
    end else if (-sh < int'(WIDTH)) begin
      mag = (WIDTH + 64)'(v[WIDTH-1:0]) >> unsigned'(-sh);
    end else begin
      mag = '0;
    end
    return v[SIGN_B] ? -signed'(mag[DW-1:0]) : signed'(mag[DW-1:0]);
  endfunction

  // ---- FFT ------------------------------------------------------------------------
  logic fft_iv, fft_ir, fft_ov;
  logic fft_or;
  logic [2*DW-1:0] fft_id, fft_od;

  fft_radix2 #(
      .M(M), .DW(DW), .CW(64), .ROM(FFT_ROM)
  ) u_fft (
      .clk(clk), .rst_n(rst_n),
      .in_valid(fft_iv), .in_ready(fft_ir), .in_data(fft_id),
      .out_valid(fft_ov), .out_ready(fft_or), .out_data(fft_od)
  );

  // ---- output normalizer ------------------------------------------------------------
  logic signed [DW-1:0] norm_in;
  logic [MPW-1:0] norm_word;

  rs_acc_norm #(
      .EXPW(EXPW), .WIDTH(WIDTH), .FRAC(54), .ACCW(DW)
  ) u_norm (
      .val (norm_in),
      .word(norm_word)
  );

  // ---- FSM ------------------------------------------------------------------------
  typedef enum logic [4:0] {
    IDLE, CLR, SPH, CEW, M1, M1W, M2, M2W, BR, BWS,
    FLD, FCOL, CRD, HI, HP, NR, NI, EMIT, FIN
  } state_e;
  state_e state;

  logic [31:0] clr_ctr, fcnt, gcnt;
  logic [3:0] p_ctr, h_ctr;
  logic [LOG2M-1:0] k_q;
  logic signed [DW-1:0] e62_q, wre_q, wim_q, m1fx_q;
  logic [2*DW-1:0] rd_q;
  logic [2*DW-1:0] g_q [0:P];
  logic signed [DW-1:0] y62_q, ar_q, ai_q;

  // centered grid index: jp = jctr - mid; idx = jp mod M (low bits)
  logic signed [24:0] jp;
  logic [LOG2M-1:0] idx;

  assign jp  = signed'({1'b0, jctr}) - signed'({1'b0, 24'(j_q >> 1)});
  assign idx = LOG2M'(unsigned'(jp));

  // combine-step products
  logic signed [2*DW-1:0] p_epr, p_epi, p_ar, p_ai, p_fr, p_fi;

  assign p_epr = wre_q * e62_q;
  assign p_epi = wim_q * e62_q;
  assign p_ar  = ai_q * y62_q;
  assign p_ai  = ar_q * y62_q;
  assign p_fr  = signed'(g_q[h_ctr][DW-1:0]) * signed'(osrom[h_ctr]);
  assign p_fi  = signed'(g_q[h_ctr][2*DW-1:DW]) * signed'(osrom[h_ctr]);

  // y62 = (jp * twopi60) >>> (LOG2M - 2)
  logic signed [DW+24:0] p_jt;
  assign p_jt = jp * signed'(osrom[P+1]);

  assign start_ready = (state == IDLE);
  assign entry_ready = (state == SPH) && entry_valid;
  assign fft_iv      = (state == FLD);
  assign fft_id      = binram[32'(p_ctr)*M+fcnt];
  assign fft_or      = (state == FCOL);
  assign norm_in     = (state == NR) ? ar_q : ai_q;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state       <= IDLE;
      point_valid <= 1'b0;
      done        <= 1'b0;
    end else begin
      point_valid <= 1'b0;
      done        <= 1'b0;
      ce_go       <= 1'b0;
      mul_go      <= 1'b0;
      unique case (state)
        IDLE: begin
          if (start_valid) begin
            tc_q    <= t0_fx + dt_fx * (64'(j_in) >> 1);
            dt_q    <= dt_fx;
            n_q     <= n_in;
            j_q     <= j_in;
            ectr    <= 24'd0;
            clr_ctr <= 32'd0;
            state   <= CLR;
          end
        end
        CLR: begin
          binram[clr_ctr] <= '0;
          if (clr_ctr == (P + 1) * M - 1) begin
            state <= SPH;
          end
          clr_ctr <= clr_ctr + 32'd1;
        end
        // ---- sweep: one entry -> anchor + rate + 15 bin accumulates ----------
        SPH: begin
          if (entry_valid) begin
            phi_q <= phi;
            ce_go <= 1'b1;
            k_q   <= k_c;
            e62_q <= e62_c;
            mul_x <= amp;  // held for both amplitude multiplies
            state <= CEW;
          end
        end
        CEW: if (ce_done) begin
          mul_y  <= ce_cos;
          mul_go <= 1'b1;
          state  <= M1;
        end
        M1: state <= M1W;  // mul in flight
        M1W: if (mul_done) begin
          m1fx_q <= to_fx54(mul_out);
          mul_y  <= ce_sin;
          mul_go <= 1'b1;
          state  <= M2;
        end
        M2: state <= M2W;
        M2W: if (mul_done) begin
          wre_q <= m1fx_q;
          wim_q <= -to_fx54(mul_out);  // conjugate
          p_ctr <= 4'd0;
          state <= BR;
        end
        BR: begin
          rd_q  <= binram[32'(p_ctr)*M+32'(k_q)];
          state <= BWS;
        end
        BWS: begin
          binram[32'(p_ctr)*M+32'(k_q)] <=
              {DW'(signed'(rd_q[2*DW-1:DW]) + wim_q), DW'(signed'(rd_q[DW-1:0]) + wre_q)};
          wre_q <= DW'(p_epr >>> 62);
          wim_q <= DW'(p_epi >>> 62);
          if (p_ctr == 4'(P)) begin
            p_ctr <= 4'd0;
            if (ectr == n_q - 24'd1) begin
              fcnt  <= 32'd0;
              gcnt  <= 32'd0;
              state <= FLD;
            end else begin
              ectr  <= ectr + 24'd1;
              state <= SPH;
            end
          end else begin
            p_ctr <= p_ctr + 4'd1;
            state <= BR;
          end
        end
        // ---- 15 in-place FFTs ------------------------------------------------
        FLD: begin
          if (fft_ir) begin
            if (fcnt == M - 1) begin
              fcnt  <= 32'd0;
              state <= FCOL;
            end else begin
              fcnt <= fcnt + 32'd1;
            end
          end
        end
        FCOL: begin
          if (fft_ov) begin
            binram[32'(p_ctr)*M+gcnt] <= fft_od;
            if (gcnt == M - 1) begin
              gcnt <= 32'd0;
              if (p_ctr == 4'(P)) begin
                p_ctr <= 4'd0;
                jctr  <= 24'd0;
                state <= CRD;
              end else begin
                p_ctr <= p_ctr + 4'd1;
                state <= FLD;
              end
            end else begin
              gcnt <= gcnt + 32'd1;
            end
          end
        end
        // ---- per-point combine -------------------------------------------------
        CRD: begin
          g_q[p_ctr] <= binram[32'(p_ctr)*M+32'(idx)];
          if (p_ctr == 4'(P)) begin
            p_ctr <= 4'd0;
            y62_q <= DW'(p_jt >>> (LOG2M - 2));
            h_ctr <= 4'(P);
            state <= HI;
          end else begin
            p_ctr <= p_ctr + 4'd1;
          end
        end
        HI: begin
          ar_q  <= DW'(p_fr >>> 62);
          ai_q  <= DW'(p_fi >>> 62);
          h_ctr <= h_ctr - 4'd1;
          state <= HP;
        end
        HP: begin
          ar_q <= DW'(p_fr >>> 62) - DW'(p_ar >>> 62);
          ai_q <= DW'(p_fi >>> 62) + DW'(p_ai >>> 62);
          if (h_ctr == 4'd0) begin
            state <= NR;
          end else begin
            h_ctr <= h_ctr - 4'd1;
          end
        end
        NR: begin
          s_re  <= norm_word;
          state <= NI;
        end
        NI: begin
          s_im  <= norm_word;
          state <= EMIT;
        end
        EMIT: begin
          point_valid <= 1'b1;
          if (jctr == j_q - 24'd1) begin
            state <= FIN;
          end else begin
            jctr  <= jctr + 24'd1;
            state <= CRD;
          end
        end
        FIN: begin
          done  <= 1'b1;
          state <= IDLE;
        end
        default: state <= IDLE;
      endcase
    end
  end

  // Structurally unused.
  // verilator lint_off UNUSEDSIGNAL
  logic unused_bits;
  assign unused_bits = ce_ready;
  // verilator lint_on UNUSEDSIGNAL

endmodule
