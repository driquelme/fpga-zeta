// npow_s_kernel: complex n^(-s), s = sigma + i*t — the shared kernel of the
// whole zeta family. Sequential FSM orchestrating the verified units.
//
//   n^(-s) = exp(-sigma*ln n) * (cos(t*ln n) - i*sin(t*ln n))
//
//   amplitude: sigma (MPF) -> fixed point; yfx = -(sigma_fx * lnn_fx) at FG
//              working bits (fused: no W-bit MPF rounding of the product,
//              which exp would amplify by |sigma*ln n|); exp_mpf in fx mode.
//   phase:     phi = frac(t * lnn2pi) via fx_mul_mod1 (t as Q32.32, lnn2pi
//              the FULL ln(n)/2pi as Q8.BW); cexp_turns; conjugate.
//   assemble:  one mpf_mul instance, time-multiplexed: re = a*cos,
//              im = -(a*sin).
//
// The amplitude and phase paths run concurrently (launched together).
// Golden model: host/zetafpga/golden/npow.py::npow_s
module npow_s_kernel #(
    parameter int unsigned LIMBS = 2,
    parameter int unsigned EXPW = 20,
    parameter int unsigned LIMBW = 64,
    parameter int unsigned TW = 32,
    parameter int unsigned TILE_LATENCY = 2,
    parameter int unsigned PHW = 160,       // WIDTH + 32
    parameter int unsigned FG = 152,
    parameter int unsigned CONSTW = 160,
    parameter int unsigned TERMS = 32,      // exp taylor terms
    parameter int unsigned CTERMS = 17,     // cexp taylor terms
    parameter int unsigned SEGW = 10,
    parameter string EXP_ROM = "expln_w128_exp.mem",
    parameter string CEXP_ROM = "cexp_w128.mem",
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned MPW = WIDTH + EXPW + 3,
    localparam int unsigned XW = FG + 32,
    localparam int unsigned BW = PHW + 32,   // fractional bits of lnn2pi
    localparam int unsigned LNW = FG + 8,    // lnn_fx width (Q8.FG)
    localparam int unsigned YW = FG + 44     // sigma*lnn product headroom
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            in_valid,
    output logic            in_ready,
    input  logic [MPW-1:0]  sigma,
    input  logic [LNW-1:0]  lnn_fx,
    input  logic [BW+7:0]   lnn2pi,   // Q8.BW
    input  logic [63:0]     t_fx,     // Q32.32, t >= 0
    output logic            out_valid,
    output logic [MPW-1:0]  re_o,
    output logic [MPW-1:0]  im_o,
    output logic            ovf,
    output logic            unf
);

  localparam int unsigned EXP_LO = WIDTH;
  localparam int unsigned SIGN_B = WIDTH + EXPW;
  localparam int unsigned ZERO_B = WIDTH + EXPW + 1;
  localparam int unsigned SPEC_B = WIDTH + EXPW + 2;

  typedef enum logic [3:0] {
    IDLE, SCONV, SMUL, LAUNCH, WAIT_AC, MUL_RE, WAIT_RE, MUL_IM, WAIT_IM, EMIT
  } state_e;
  state_e state;

  // ---- latched inputs & bookkeeping registers -------------------------------
  logic [MPW-1:0] sig_q;
  logic [LNW-1:0] lnn_q;
  logic [63:0] t_fx_q;
  logic [BW+7:0] lnn2pi_q;
  logic [MPW-1:0] amp_q, cos_q, sin_q;
  logic a_done, c_done, ovf_r, unf_r;

  // ---- phase path (combinational from the latched inputs) -------------------
  logic [PHW-1:0] phi;

  fx_mul_mod1 #(
      .AW(64), .AF(32), .BW(BW), .BI(8), .PHW(PHW)
  ) u_phase (
      .a   (t_fx_q),
      .b   (lnn2pi_q),
      .frac(phi)
  );

  // ---- amplitude path -------------------------------------------------------
  logic signed [XW-1:0] sig_fx;
  logic signed [YW-1:0] ywide;
  logic signed [XW-1:0] yfx_clamped;
  logic sat_pre;  // sigma too large: saturate without multiplying

  logic signed [YW+LNW-1:0] p_slnn;
  assign p_slnn = sig_fx * signed'({1'b0, lnn_q});

  localparam logic signed [XW-1:0] CLAMP = XW'(1) <<< (FG + 22);

  always_comb begin
    if (ywide > YW'(CLAMP)) begin
      yfx_clamped = CLAMP;
    end else if (ywide < -YW'(CLAMP)) begin
      yfx_clamped = -CLAMP;
    end else begin
      yfx_clamped = XW'(ywide);
    end
  end

  // The launch protocol guarantees the sub-units are idle (single operand in
  // flight), so their ready outputs are structurally unused.
  // verilator lint_off UNUSEDSIGNAL
  logic exp_ready, cexp_ready;
  // verilator lint_on UNUSEDSIGNAL

  logic exp_launch, exp_done_v;
  logic [MPW-1:0] amp;
  logic exp_ovf, exp_unf;

  exp_mpf #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW),
      .FG(FG), .CONSTW(CONSTW), .TERMS(TERMS), .CONSTS_ROM(EXP_ROM)
  ) u_exp (
      .clk(clk), .rst_n(rst_n),
      .in_valid(exp_launch), .in_ready(exp_ready),
      .x('0), .fx_mode(1'b1), .yfx_in(yfx_clamped),
      .out_valid(exp_done_v), .result(amp), .ovf(exp_ovf), .unf(exp_unf)
  );

  logic cexp_launch, cexp_done_v;
  logic [MPW-1:0] cos_w, sin_w;

  cexp_turns #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW), .PHW(PHW),
      .FG(FG), .CONSTW(CONSTW), .SEGW(SEGW), .TERMS(CTERMS),
      .CONST_LINES(TERMS + 2), .CEXP_ROM(CEXP_ROM), .EXP_ROM(EXP_ROM)
  ) u_cexp (
      .clk(clk), .rst_n(rst_n),
      .in_valid(cexp_launch), .in_ready(cexp_ready),
      .phi(phi),
      .out_valid(cexp_done_v), .cos_o(cos_w), .sin_o(sin_w)
  );

  // ---- assembly multiplier (time-multiplexed) -------------------------------
  logic mul_launch, mul_done_v;
  logic [MPW-1:0] mul_y, mul_out;
  logic mul_ovf, mul_unf;

  mpf_mul #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW), .TW(TW), .TILE_LATENCY(TILE_LATENCY)
  ) u_mul (
      .clk(clk), .rst_n(rst_n), .in_valid(mul_launch),
      .x(amp_q), .y(mul_y),
      .out_valid(mul_done_v), .result(mul_out), .ovf(mul_ovf), .unf(mul_unf)
  );

  assign mul_y = (state == MUL_RE || state == WAIT_RE) ? cos_q : sin_q;

  assign in_ready = (state == IDLE);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= IDLE;
      out_valid <= 1'b0;
    end else begin
      out_valid   <= 1'b0;
      exp_launch  <= 1'b0;
      cexp_launch <= 1'b0;
      mul_launch  <= 1'b0;

      // Collect completions from the parallel units in any state.
      if (exp_done_v) begin
        amp_q  <= amp;
        a_done <= 1'b1;
        ovf_r  <= ovf_r | exp_ovf;
        unf_r  <= unf_r | exp_unf;
      end
      if (cexp_done_v) begin
        cos_q  <= cos_w;
        sin_q  <= sin_w;
        c_done <= 1'b1;
      end

      unique case (state)
        IDLE: begin
          if (in_valid) begin
            sig_q     <= sigma;
            lnn_q     <= lnn_fx;
            t_fx_q    <= t_fx;
            lnn2pi_q  <= lnn2pi;
            a_done    <= 1'b0;
            c_done    <= 1'b0;
            ovf_r     <= 1'b0;
            unf_r     <= 1'b0;
            sat_pre   <= 1'b0;
            if (sigma[SPEC_B]) begin
              re_o      <= {1'b1, 1'b0, sigma[SIGN_B], {EXPW{1'b0}}, {WIDTH{1'b0}}};
              im_o      <= {1'b1, 1'b0, sigma[SIGN_B], {EXPW{1'b0}}, {WIDTH{1'b0}}};
              ovf       <= 1'b0;
              unf       <= 1'b0;
              out_valid <= 1'b1;
            end else begin
              state <= SCONV;
            end
          end
        end
        SCONV: begin
          // sigma -> fixed point at scale 2^FG (bounded: exp>30 pre-clamps).
          if (!sig_q[ZERO_B] && signed'(sig_q[EXP_LO+:EXPW]) > EXPW'(30)) begin
            sat_pre <= 1'b1;
            ywide   <= (lnn_q == 0) ? '0
                     : (sig_q[SIGN_B] ? YW'(CLAMP) <<< 1 : -(YW'(CLAMP) <<< 1));
            state   <= LAUNCH;
          end else begin
            automatic int sh = int'(FG) + int'(signed'(sig_q[EXP_LO+:EXPW])) - int'(WIDTH);
            automatic logic [XW-1:0] mag;
            if (sig_q[ZERO_B]) begin
              mag = '0;
            end else if (sh >= 0) begin
              mag = XW'(sig_q[WIDTH-1:0]) << unsigned'(sh);
            end else if (-sh < int'(WIDTH)) begin
              mag = XW'(sig_q[WIDTH-1:0]) >> unsigned'(-sh);
            end else begin
              mag = '0;
            end
            sig_fx <= sig_q[SIGN_B] ? -signed'(mag) : signed'(mag);
            state  <= SMUL;
          end
        end
        SMUL: begin
          ywide <= -(YW'(p_slnn >>> FG));
          state <= LAUNCH;
        end
        LAUNCH: begin
          exp_launch  <= 1'b1;
          cexp_launch <= 1'b1;
          state       <= WAIT_AC;
        end
        WAIT_AC: begin
          if (a_done && c_done) begin
            if (amp_q[SPEC_B]) begin
              re_o      <= amp_q;
              im_o      <= amp_q;
              ovf       <= ovf_r;
              unf       <= unf_r;
              out_valid <= 1'b1;
              state     <= IDLE;
            end else begin
              mul_launch <= 1'b1;
              state      <= WAIT_RE;
            end
          end
        end
        WAIT_RE: begin
          if (mul_done_v) begin
            re_o       <= mul_out;
            ovf_r      <= ovf_r | mul_ovf;
            unf_r      <= unf_r | mul_unf;
            mul_launch <= 1'b1;
            state      <= WAIT_IM;
          end
        end
        WAIT_IM: begin
          if (mul_done_v) begin
            // conjugate: im = -(a*sin)
            im_o  <= {mul_out[MPW-1:MPW-2], ~mul_out[SIGN_B], mul_out[SIGN_B-1:0]};
            ovf   <= ovf_r | mul_ovf;
            unf   <= unf_r | mul_unf;
            out_valid <= 1'b1;
            state <= IDLE;
          end
        end
        default: state <= IDLE;
      endcase
    end
  end

  // sat_pre is bookkeeping for waveform debugging; the clamp acts via ywide.
  // verilator lint_off UNUSEDSIGNAL
  logic unused_sat;
  assign unused_sat = sat_pre;
  // verilator lint_on UNUSEDSIGNAL

endmodule
