// cexp_pipe: fully pipelined e^(2*pi*i*phi), II = 1.
//
// Same decomposition and bit-exact math as cexp_turns (table factor for the
// top SEGW phase bits x complex Taylor in the residual), but with the Horner
// recurrence unrolled into TERMS-1 pipeline stages so a new phase enters
// every cycle. This is the hot datapath of the pipelined RS power-sum engine.
//
//   LATENCY = TERMS + 2   (S0 + (TERMS-1) Horner + table-mul + normalize)
//
// Golden model: host/zetafpga/golden/cexp.py::cexp_turns (identical results).
module cexp_pipe #(
    parameter int unsigned LIMBS = 1,
    parameter int unsigned EXPW = 20,
    parameter int unsigned LIMBW = 64,
    parameter int unsigned PHW = 96,
    parameter int unsigned FG = 88,
    parameter int unsigned CONSTW = 96,
    parameter int unsigned SEGW = 10,
    parameter int unsigned TERMS = 11,
    parameter int unsigned CONST_LINES = 24,
    parameter string CEXP_ROM = "cexp_w64.mem",
    parameter string EXP_ROM = "expln_w64_exp.mem",
    localparam int unsigned WIDTH = LIMBS * LIMBW,
    localparam int unsigned MPW = WIDTH + EXPW + 3,
    localparam int unsigned XW = FG + 32,
    localparam int unsigned NTBL = 2 * (1 << SEGW) + 1,
    localparam int unsigned LATENCY = TERMS + 2
) (
    input  logic           clk,
    input  logic           rst_n,
    input  logic           in_valid,
    input  logic [PHW-1:0] phi,
    output logic           out_valid,
    output logic [MPW-1:0] cos_o,
    output logic [MPW-1:0] sin_o
);

  localparam int unsigned LOW = PHW - SEGW;

  logic signed [CONSTW-1:0] tbl [0:NTBL-1];
  // verilator lint_off UNUSEDSIGNAL
  logic signed [CONSTW-1:0] econsts [0:CONST_LINES-1];
  // verilator lint_on UNUSEDSIGNAL

  initial begin
    $readmemh(CEXP_ROM, tbl);
    $readmemh(EXP_ROM, econsts);
  end

  // ---- stage 0: z, table factor -------------------------------------------
  logic [SEGW-1:0] hi;
  logic [LOW-1:0] lo;
  assign hi = phi[PHW-1-:SEGW];
  assign lo = phi[LOW-1:0];

  logic [LOW+CONSTW-1:0] p_z;
  assign p_z = lo * unsigned'(tbl[NTBL-1]);  // 2*pi at scale 2^(FG-3)

  // ---- per-stage registers (index = pipeline stage) -------------------------
  logic signed [XW-1:0] z_p [0:TERMS-1];
  logic signed [XW-1:0] are_p [0:TERMS-1];
  logic signed [XW-1:0] aim_p [0:TERMS-1];
  logic signed [CONSTW-1:0] tc_p [0:TERMS-1];
  logic signed [CONSTW-1:0] ts_p [0:TERMS-1];

  always_ff @(posedge clk) begin
    z_p[0]   <= XW'(p_z >> (PHW - 3));
    are_p[0] <= XW'(econsts[TERMS+1]);  // invfact[TERMS-1]
    aim_p[0] <= '0;
    tc_p[0]  <= tbl[32'(hi)*2];
    ts_p[0]  <= tbl[32'(hi)*2+1];
  end

  for (genvar gk = 1; gk < int'(TERMS); gk++) begin : gen_horner
    logic signed [2*XW-1:0] p_zre, p_zim;
    assign p_zre = z_p[gk-1] * are_p[gk-1];
    assign p_zim = z_p[gk-1] * aim_p[gk-1];
    always_ff @(posedge clk) begin
      are_p[gk] <= XW'(econsts[TERMS + 1 - gk]) - XW'(p_zim >>> FG);
      aim_p[gk] <= XW'(p_zre >>> FG);
      z_p[gk]   <= z_p[gk-1];
      tc_p[gk]  <= tc_p[gk-1];
      ts_p[gk]  <= ts_p[gk-1];
    end
  end

  // ---- table-factor complex multiply -----------------------------------------
  logic signed [XW+CONSTW-1:0] p_tcre, p_tsim, p_tcim, p_tsre;
  assign p_tcre = tc_p[TERMS-1] * are_p[TERMS-1];
  assign p_tsim = ts_p[TERMS-1] * aim_p[TERMS-1];
  assign p_tcim = tc_p[TERMS-1] * aim_p[TERMS-1];
  assign p_tsre = ts_p[TERMS-1] * are_p[TERMS-1];

  logic signed [XW-1:0] ore, oim;

  always_ff @(posedge clk) begin
    ore <= XW'((p_tcre - p_tsim) >>> FG);
    oim <= XW'((p_tcim + p_tsre) >>> FG);
  end

  // ---- two parallel normalizers ------------------------------------------------
  logic [MPW-1:0] cos_n, sin_n;

  cexp_pipe_norm #(
      .EXPW(EXPW), .WIDTH(WIDTH), .FG(FG), .XW(XW)
  ) u_norm_c (
      .val (ore),
      .word(cos_n)
  );

  cexp_pipe_norm #(
      .EXPW(EXPW), .WIDTH(WIDTH), .FG(FG), .XW(XW)
  ) u_norm_s (
      .val (oim),
      .word(sin_n)
  );

  logic [LATENCY-1:0] vpipe;

  always_ff @(posedge clk) begin
    cos_o <= cos_n;
    sin_o <= sin_n;
    if (!rst_n) begin
      vpipe <= '0;
    end else begin
      vpipe <= {vpipe[LATENCY-2:0], in_valid};
    end
  end

  assign out_valid = vpipe[LATENCY-1];

endmodule

// Combinational FG-scale fixed point -> MPF normalizer (RNE), shared shape
// with cexp_turns' NORM stage / golden _fix_to_mpf. Local helper module.
// verilator lint_off DECLFILENAME
module cexp_pipe_norm #(
    parameter int unsigned EXPW = 20,
    parameter int unsigned WIDTH = 64,
    parameter int unsigned FG = 88,
    parameter int unsigned XW = 120,
    localparam int unsigned MPW = WIDTH + EXPW + 3
) (
    input  logic signed [XW-1:0] val,
    output logic [MPW-1:0]       word
);

  logic sign;
  logic [XW-1:0] mag;
  logic [$clog2(XW+1)-1:0] lz;
  logic zero;

  assign sign = val < 0;
  assign mag  = sign ? unsigned'(-val) : unsigned'(val);

  lzc #(
      .WIDTH(XW)
  ) u_lzc (
      .data    (mag),
      .count   (lz),
      .all_zero(zero)
  );

  logic [WIDTH+1:0] t_norm;
  logic [WIDTH:0] mant_rnd;
  // |values| <= 1: exponent far from saturation, top bits structurally unused.
  // verilator lint_off UNUSEDSIGNAL
  logic signed [23:0] e_norm, e_f;
  // verilator lint_on UNUSEDSIGNAL
  logic [WIDTH-1:0] mant_f;

  always_comb begin
    automatic int p = int'(XW) - 1 - int'(32'(lz));
    e_norm = 24'(p - int'(FG) + 1);
    if (p >= int'(WIDTH)) begin
      t_norm = (WIDTH+2)'(mag >> unsigned'(p - int'(WIDTH)));
    end else begin
      t_norm = (WIDTH+2)'(mag) << unsigned'(int'(WIDTH) - p);
    end
    mant_rnd = (WIDTH+1)'((t_norm + (WIDTH+2)'(1)) >> 1);
    if (mant_rnd[WIDTH]) begin
      mant_f = mant_rnd[WIDTH:1];
      e_f    = e_norm + 24'sd1;
    end else begin
      mant_f = mant_rnd[WIDTH-1:0];
      e_f    = e_norm;
    end
    if (zero) begin
      word = {1'b0, 1'b1, 1'b0, {EXPW{1'b0}}, {WIDTH{1'b0}}};
    end else begin
      word = {1'b0, 1'b0, sign, e_f[EXPW-1:0], mant_f};
    end
  end

endmodule
// verilator lint_on DECLFILENAME
