// euler_maclaurin_top: zeta(sigma + i*t) by Euler-Maclaurin summation.
//
//   zeta(s) = sum_{n=1..N} n^-s + (N+1)^(1-s)/(s-1) + (N+1)^-s/2
//           + sum_{j=1..M} B_2j/(2j)! (s)_{2j-1} (N+1)^(-s-2j+1)
//
// Sequential FSM around one npow_s_kernel, one mpf_mul and one mpf_add,
// with complex-multiply (CM_*) and accumulate (AD_*) subroutine states and
// return registers. The host supplies N, M, 1/(s-1), 1/(N+1)^2, t as MPF,
// and streams the ln(n) entries (n = 1..N+1, then N+1 once more) and the
// M Bernoulli coefficients — the WRITE_TABLE / COMPUTE_EM payloads of the
// M8 descriptor ISA.
//
// Golden model: host/zetafpga/golden/zeta_em.py (bit-exact, same op order).
module euler_maclaurin_top #(
    parameter int unsigned LIMBS = 2,
    parameter int unsigned EXPW = 20,
    parameter int unsigned LIMBW = 64,
    parameter int unsigned TW = 32,
    parameter int unsigned TILE_LATENCY = 2,
    parameter int unsigned PHW = 160,
    parameter int unsigned FG = 152,
    parameter int unsigned CONSTW = 160,
    parameter int unsigned TERMS = 32,
    parameter int unsigned CTERMS = 17,
    parameter int unsigned SEGW = 10,
    parameter string EXP_ROM = "expln_w128_exp.mem",
    parameter string CEXP_ROM = "cexp_w128.mem",
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned MPW = WIDTH + EXPW + 3,
    localparam int unsigned BW = PHW + 32,
    localparam int unsigned LNW = FG + 8
) (
    input  logic            clk,
    input  logic            rst_n,

    // configuration + start
    input  logic            start_valid,
    output logic            start_ready,
    input  logic            ps_only,  // COMPUTE_PS: finish after the power sum
    input  logic [MPW-1:0]  sigma,
    input  logic [MPW-1:0]  t_mpf,
    input  logic [63:0]     t_fx,
    input  logic [23:0]     n_in,
    input  logic [11:0]     m_in,
    input  logic [MPW-1:0]  inv_sm1_re,
    input  logic [MPW-1:0]  inv_sm1_im,
    input  logic [MPW-1:0]  inv_np2,

    // streamed ln(n) entries: n = 1..N+1, then N+1 again (N+2 beats)
    input  logic            entry_valid,
    output logic            entry_ready,
    input  logic [LNW-1:0]  entry_lnn_fx,
    input  logic [BW+7:0]   entry_lnn2pi,

    // streamed Bernoulli coefficients B_2j/(2j)!: j = 1..M
    input  logic            bern_valid,
    output logic            bern_ready,
    input  logic [MPW-1:0]  bern_data,

    output logic            out_valid,
    output logic [MPW-1:0]  z_re,
    output logic [MPW-1:0]  z_im,
    output logic            ovf,
    output logic            unf
);

  localparam int unsigned SIGN_B = WIDTH + EXPW;
  localparam int unsigned ZERO_B = WIDTH + EXPW + 1;
  localparam int unsigned SPEC_B = WIDTH + EXPW + 2;
  localparam int EMIN = -(1 << (EXPW - 1));

  localparam logic [MPW-1:0] ONE_W =
      {1'b0, 1'b0, 1'b0, EXPW'(1), {1'b1, {(WIDTH - 1) {1'b0}}}};

  // ---- helpers --------------------------------------------------------------
  function automatic logic [MPW-1:0] int2mpf(input logic [23:0] v);
    int msb;
    logic [WIDTH-1:0] mant;
    msb = 0;
    for (int i = 0; i < 24; i++) begin
      if (v[i]) msb = i;
    end
    if (v == 0) begin
      return {1'b0, 1'b1, 1'b0, {EXPW{1'b0}}, {WIDTH{1'b0}}};
    end
    mant = WIDTH'(v) << (WIDTH - 1 - msb);
    return {1'b0, 1'b0, 1'b0, EXPW'(msb + 1), mant};
  endfunction

  function automatic logic [MPW:0] scale_half(input logic [MPW-1:0] v);
    // returns {unf, word}: exact /2 via exponent decrement, saturating.
    logic signed [EXPW-1:0] e;
    e = signed'(v[WIDTH+:EXPW]);
    if (v[ZERO_B] || v[SPEC_B]) begin
      return {1'b0, v};
    end
    if (int'(e) - 1 < EMIN) begin
      return {1'b1, {1'b0, 1'b1, v[SIGN_B], {EXPW{1'b0}}, {WIDTH{1'b0}}}};
    end
    return {1'b0, {v[MPW-1:SIGN_B], EXPW'(e) - EXPW'(1), v[WIDTH-1:0]}};
  endfunction

  function automatic logic [MPW-1:0] negw(input logic [MPW-1:0] v);
    return {v[MPW-1:SIGN_B+1], ~v[SIGN_B], v[SIGN_B-1:0]};
  endfunction

  // ---- datapath units --------------------------------------------------------
  logic np_go, np_ready, np_done;
  logic [MPW-1:0] np_sigma, np_re, np_im;
  logic [LNW-1:0] lnn_q;
  logic [BW+7:0] lnn2pi_q;
  logic np_ovf, np_unf;

  npow_s_kernel #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW), .TW(TW),
      .TILE_LATENCY(TILE_LATENCY), .PHW(PHW), .FG(FG), .CONSTW(CONSTW),
      .TERMS(TERMS), .CTERMS(CTERMS), .SEGW(SEGW),
      .EXP_ROM(EXP_ROM), .CEXP_ROM(CEXP_ROM)
  ) u_npow (
      .clk(clk), .rst_n(rst_n),
      .in_valid(np_go), .in_ready(np_ready),
      .sigma(np_sigma), .lnn_fx(lnn_q), .lnn2pi(lnn2pi_q), .t_fx(t_fx_q),
      .out_valid(np_done), .re_o(np_re), .im_o(np_im),
      .ovf(np_ovf), .unf(np_unf)
  );

  logic mul_go, mul_done, mul_ovf, mul_unf;
  logic [MPW-1:0] mul_x, mul_y, mul_out;

  mpf_mul #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW), .TW(TW), .TILE_LATENCY(TILE_LATENCY)
  ) u_mul (
      .clk(clk), .rst_n(rst_n), .in_valid(mul_go),
      .x(mul_x), .y(mul_y),
      .out_valid(mul_done), .result(mul_out), .ovf(mul_ovf), .unf(mul_unf)
  );

  logic add_go, add_done, add_ovf, add_unf;
  logic [MPW-1:0] add_x, add_y, add_out;

  mpf_add #(
      .LIMBS(LIMBS), .EXPW(EXPW), .LIMBW(LIMBW)
  ) u_add (
      .clk(clk), .rst_n(rst_n), .in_valid(add_go),
      .x(add_x), .y(add_y),
      .out_valid(add_done), .result(add_out), .ovf(add_ovf), .unf(add_unf)
  );

  // ---- state ------------------------------------------------------------------
  typedef enum logic [5:0] {
    IDLE,
    PS_REQ, PS_W,
    P_REQ, P_W,
    INT_M1, INT_M1W, INT_M2, INT_M2W, INT_CM, INT_ACC,
    HALF,
    T_S1, T_S1W,
    T_REQ, T_W, T_CM, T_U0,
    TB_REQ, TB_M1, TB_M1W, TB_M2, TB_M2W, TB_ACC,
    TB_NEXT, TB_C1, TB_C1W, TB_C2, TB_C2W, TB_UCM1, TB_U1, TB_UCM2, TB_U2,
    TB_S1, TB_S1W, TB_S2, TB_S2W,
    CM_P1, CM_W1, CM_P2, CM_W2, CM_P3, CM_W3, CM_P4, CM_W4,
    CM_A1, CM_WA1, CM_A2, CM_WA2,
    AD_R, AD_WR, AD_I, AD_WI,
    DONE
  } state_e;

  state_e state, ret_cm, ret_ad;

  // configuration
  logic ps_q;
  logic [MPW-1:0] sigma_q, t_mpf_q, inv_re_q, inv_im_q, invnp2_q, sigma1_q, c2r_q;
  logic [63:0] t_fx_q;
  logic [23:0] n_q, n_ctr;
  logic [11:0] m_q, j_ctr;

  // working registers
  logic [MPW-1:0] acc_r, acc_i, p_r, p_i, u_r, u_i, x_r, x_i, bern_q;
  logic [MPW-1:0] cma_r, cma_i, cmb_r, cmb_i, cmr_r, cmr_i, t1, t2, t3, t4;
  logic ovf_r, unf_r;

  assign start_ready = (state == IDLE);
  assign entry_ready = (state == PS_REQ) || (state == P_REQ) || (state == T_REQ);
  assign bern_ready  = (state == TB_REQ);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= IDLE;
      out_valid <= 1'b0;
    end else begin
      out_valid <= 1'b0;
      np_go     <= 1'b0;
      mul_go    <= 1'b0;
      add_go    <= 1'b0;

      // sticky flag collection from every unit completion
      if (np_done) begin
        ovf_r <= ovf_r | np_ovf;
        unf_r <= unf_r | np_unf;
      end
      if (mul_done) begin
        ovf_r <= ovf_r | mul_ovf;
        unf_r <= unf_r | mul_unf;
      end
      if (add_done) begin
        ovf_r <= ovf_r | add_ovf;
        unf_r <= unf_r | add_unf;
      end

      unique case (state)
        IDLE: begin
          if (start_valid) begin
            ps_q     <= ps_only;
            sigma_q  <= sigma;
            t_mpf_q  <= t_mpf;
            t_fx_q   <= t_fx;
            n_q      <= n_in;
            m_q      <= m_in;
            inv_re_q <= inv_sm1_re;
            inv_im_q <= inv_sm1_im;
            invnp2_q <= inv_np2;
            acc_r    <= {1'b0, 1'b1, 1'b0, {EXPW{1'b0}}, {WIDTH{1'b0}}};
            acc_i    <= {1'b0, 1'b1, 1'b0, {EXPW{1'b0}}, {WIDTH{1'b0}}};
            ovf_r    <= 1'b0;
            unf_r    <= 1'b0;
            n_ctr    <= 24'd1;
            state    <= PS_REQ;
          end
        end

        // ---- power sum ------------------------------------------------------
        PS_REQ: begin
          if (entry_valid) begin
            lnn_q    <= entry_lnn_fx;
            lnn2pi_q <= entry_lnn2pi;
            np_sigma <= sigma_q;
            np_go    <= 1'b1;
            state    <= PS_W;
          end
        end
        PS_W: begin
          if (np_done) begin
            x_r    <= np_re;
            x_i    <= np_im;
            ret_ad <= (n_ctr == n_q) ? (ps_q ? DONE : P_REQ) : PS_REQ;
            n_ctr  <= n_ctr + 24'd1;
            state  <= AD_R;
          end
        end

        // ---- (N+1)^-s ---------------------------------------------------------
        P_REQ: begin
          if (entry_valid) begin
            lnn_q    <= entry_lnn_fx;
            lnn2pi_q <= entry_lnn2pi;
            np_sigma <= sigma_q;
            np_go    <= 1'b1;
            state    <= P_W;
          end
        end
        P_W: begin
          if (np_done) begin
            p_r   <= np_re;
            p_i   <= np_im;
            state <= INT_M1;
          end
        end

        // ---- integral term: (N+1)*P (x) inv_sm1 -------------------------------
        INT_M1: begin
          mul_x  <= p_r;
          mul_y  <= int2mpf(n_q + 24'd1);
          mul_go <= 1'b1;
          state  <= INT_M1W;
        end
        INT_M1W: if (mul_done) begin cma_r <= mul_out; state <= INT_M2; end
        INT_M2: begin
          mul_x  <= p_i;
          mul_y  <= int2mpf(n_q + 24'd1);
          mul_go <= 1'b1;
          state  <= INT_M2W;
        end
        INT_M2W: if (mul_done) begin cma_i <= mul_out; state <= INT_CM; end
        INT_CM: begin
          cmb_r  <= inv_re_q;
          cmb_i  <= inv_im_q;
          ret_cm <= INT_ACC;
          state  <= CM_P1;
        end
        INT_ACC: begin
          x_r    <= cmr_r;
          x_i    <= cmr_i;
          ret_ad <= HALF;
          state  <= AD_R;
        end

        // ---- half term: + P/2 ---------------------------------------------------
        HALF: begin
          x_r    <= scale_half(p_r)[MPW-1:0];
          x_i    <= scale_half(p_i)[MPW-1:0];
          unf_r  <= unf_r | scale_half(p_r)[MPW] | scale_half(p_i)[MPW];
          ret_ad <= T_S1;
          state  <= AD_R;
        end

        // ---- tail setup ----------------------------------------------------------
        T_S1: begin
          add_x  <= sigma_q;
          add_y  <= ONE_W;
          add_go <= 1'b1;
          state  <= T_S1W;
        end
        T_S1W: if (add_done) begin sigma1_q <= add_out; state <= T_REQ; end
        T_REQ: begin
          if (entry_valid) begin
            lnn_q    <= entry_lnn_fx;
            lnn2pi_q <= entry_lnn2pi;
            np_sigma <= sigma1_q;
            np_go    <= 1'b1;
            state    <= T_W;
          end
        end
        T_W: begin
          if (np_done) begin
            cma_r  <= sigma_q;   // s = (sigma, t)
            cma_i  <= t_mpf_q;
            cmb_r  <= np_re;     // (N+1)^(-s-1)
            cmb_i  <= np_im;
            ret_cm <= T_U0;
            state  <= CM_P1;
          end
        end
        T_U0: begin
          u_r   <= cmr_r;
          u_i   <= cmr_i;
          j_ctr <= 12'd1;
          state <= TB_REQ;
        end

        // ---- Bernoulli tail loop --------------------------------------------------
        TB_REQ: begin
          if (bern_valid) begin
            bern_q <= bern_data;
            state  <= TB_M1;
          end
        end
        TB_M1: begin
          mul_x  <= u_r;
          mul_y  <= bern_q;
          mul_go <= 1'b1;
          state  <= TB_M1W;
        end
        TB_M1W: if (mul_done) begin x_r <= mul_out; state <= TB_M2; end
        TB_M2: begin
          mul_x  <= u_i;
          mul_y  <= bern_q;
          mul_go <= 1'b1;
          state  <= TB_M2W;
        end
        TB_M2W: if (mul_done) begin x_i <= mul_out; ret_ad <= TB_NEXT; state <= AD_R; end

        TB_NEXT: begin
          if (j_ctr == m_q) begin
            state <= DONE;
          end else begin
            state <= TB_C1;
          end
        end
        TB_C1: begin
          add_x  <= sigma_q;
          add_y  <= int2mpf(24'(2 * j_ctr) - 24'd1);
          add_go <= 1'b1;
          state  <= TB_C1W;
        end
        TB_C1W: if (add_done) begin cmb_r <= add_out; state <= TB_C2; end
        TB_C2: begin
          add_x  <= sigma_q;
          add_y  <= int2mpf(24'(2 * j_ctr));
          add_go <= 1'b1;
          state  <= TB_C2W;
        end
        TB_C2W: if (add_done) begin c2r_q <= add_out; state <= TB_UCM1; end
        TB_UCM1: begin
          cma_r  <= u_r;
          cma_i  <= u_i;
          cmb_i  <= t_mpf_q;  // cmb_r already holds sigma + (2j-1)
          ret_cm <= TB_U1;
          state  <= CM_P1;
        end
        TB_U1: begin
          u_r    <= cmr_r;
          u_i    <= cmr_i;
          state  <= TB_UCM2;
        end
        TB_UCM2: begin
          cma_r  <= u_r;
          cma_i  <= u_i;
          cmb_r  <= c2r_q;
          cmb_i  <= t_mpf_q;
          ret_cm <= TB_U2;
          state  <= CM_P1;
        end
        TB_U2: begin
          u_r   <= cmr_r;
          u_i   <= cmr_i;
          state <= TB_S1;
        end
        TB_S1: begin
          mul_x  <= u_r;
          mul_y  <= invnp2_q;
          mul_go <= 1'b1;
          state  <= TB_S1W;
        end
        TB_S1W: if (mul_done) begin u_r <= mul_out; state <= TB_S2; end
        TB_S2: begin
          mul_x  <= u_i;
          mul_y  <= invnp2_q;
          mul_go <= 1'b1;
          state  <= TB_S2W;
        end
        TB_S2W: if (mul_done) begin
          u_i   <= mul_out;
          j_ctr <= j_ctr + 12'd1;
          state <= TB_REQ;
        end

        // ---- subroutine: CMR = CMA (x) CMB ------------------------------------
        CM_P1: begin mul_x <= cma_r; mul_y <= cmb_r; mul_go <= 1'b1; state <= CM_W1; end
        CM_W1: if (mul_done) begin t1 <= mul_out; state <= CM_P2; end
        CM_P2: begin mul_x <= cma_i; mul_y <= cmb_i; mul_go <= 1'b1; state <= CM_W2; end
        CM_W2: if (mul_done) begin t2 <= mul_out; state <= CM_P3; end
        CM_P3: begin mul_x <= cma_r; mul_y <= cmb_i; mul_go <= 1'b1; state <= CM_W3; end
        CM_W3: if (mul_done) begin t3 <= mul_out; state <= CM_P4; end
        CM_P4: begin mul_x <= cma_i; mul_y <= cmb_r; mul_go <= 1'b1; state <= CM_W4; end
        CM_W4: if (mul_done) begin t4 <= mul_out; state <= CM_A1; end
        CM_A1: begin add_x <= t1; add_y <= negw(t2); add_go <= 1'b1; state <= CM_WA1; end
        CM_WA1: if (add_done) begin cmr_r <= add_out; state <= CM_A2; end
        CM_A2: begin add_x <= t3; add_y <= t4; add_go <= 1'b1; state <= CM_WA2; end
        CM_WA2: if (add_done) begin cmr_i <= add_out; state <= ret_cm; end

        // ---- subroutine: acc += X ------------------------------------------------
        AD_R:  begin add_x <= acc_r; add_y <= x_r; add_go <= 1'b1; state <= AD_WR; end
        AD_WR: if (add_done) begin acc_r <= add_out; state <= AD_I; end
        AD_I:  begin add_x <= acc_i; add_y <= x_i; add_go <= 1'b1; state <= AD_WI; end
        AD_WI: if (add_done) begin acc_i <= add_out; state <= ret_ad; end

        DONE: begin
          z_re      <= acc_r;
          z_im      <= acc_i;
          ovf       <= ovf_r;
          unf       <= unf_r;
          out_valid <= 1'b1;
          state     <= IDLE;
        end
        default: state <= IDLE;
      endcase
    end
  end

  // np_ready is guaranteed by construction (single operand in flight).
  // verilator lint_off UNUSEDSIGNAL
  logic unused_np_ready;
  assign unused_np_ready = np_ready;
  // verilator lint_on UNUSEDSIGNAL

endmodule
