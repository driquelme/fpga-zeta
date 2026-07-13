// fft_radix2: sequential in-place complex fixed-point FFT (M20).
//
// The butterfly stage of the Odlyzko-Schonhage multi-evaluation
// (host/zetafpga/kernel/os_multieval.py is the algorithm blueprint;
// golden/fft.py is the bit-exact model of THIS unit).
//
//   kernel e^(+2 pi i jk/M);  t = (w * b) >>> (CW-2);  a' = a + t, b' = a - t
//
// Values are signed DW-bit fixed point per component (caller's scale).
// There is no per-stage growth to scale away: every DIT intermediate is a
// DFT of a subset of the inputs, hence bounded by the input l1 norm — the
// caller guarantees l1 < 2^(DW-1) (O-S bins carry l1 <= ~2 sqrt(N)).
// Error: ~1 truncation lsb per component per stage + 2^-(CW-2) twiddles.
//
// Protocol: stream M complex words in ({im, re}, natural order — the loader
// bit-reverses addresses), butterflies run (4 cycles each, ~2 M log2 M
// total), then M results stream out in natural order.
module fft_radix2 #(
    parameter int unsigned M = 256,
    parameter int unsigned DW = 64,           // bits per component
    parameter int unsigned CW = 64,           // twiddle bits (Q2.CW-2)
    parameter string ROM = "fft_m256.mem",
    localparam int unsigned LOG2M = $clog2(M),
    localparam int unsigned WW = 2 * DW       // packed {im, re}
) (
    input  logic          clk,
    input  logic          rst_n,
    input  logic          in_valid,
    output logic          in_ready,
    input  logic [WW-1:0] in_data,
    output logic          out_valid,
    input  logic          out_ready,
    output logic [WW-1:0] out_data
);

  localparam int unsigned SH = CW - 2;

  logic [2*CW-1:0] rom [0:M/2-1];

  initial begin
    $readmemh(ROM, rom);
  end

  logic [WW-1:0] mem [0:M-1];

  function automatic logic [LOG2M-1:0] bitrev(input logic [LOG2M-1:0] i);
    logic [LOG2M-1:0] r;
    for (int unsigned b = 0; b < LOG2M; b++) begin
      r[LOG2M-1-b] = i[b];
    end
    return r;
  endfunction

  typedef enum logic [2:0] { IDLE, LOAD, RDA, RDB, WRA, WRB, OUTP, OUTV } state_e;
  state_e state;

  logic [LOG2M:0] cnt;                 // load/output element counter
  logic [$clog2(LOG2M+1)-1:0] stage;
  logic [LOG2M-1:0] base, k_ctr;       // butterfly block base and offset
  logic [WW-1:0] a_q, b_q;

  // span = 1 << stage; b address = a address + span
  logic [LOG2M-1:0] a_addr, b_addr, span;

  assign span   = LOG2M'(32'd1 << stage);
  assign a_addr = base | k_ctr;
  assign b_addr = a_addr | span;

  logic [LOG2M-1:0] load_addr;
  assign load_addr = bitrev(cnt[LOG2M-1:0]);

  // twiddle index k_ctr * (M / (2*span)): stride is a power of two
  logic [LOG2M-1:0] tw_idx;
  assign tw_idx = LOG2M'(32'(k_ctr) << (LOG2M - 1 - 32'(stage)));

  // butterfly arithmetic (combinational; behavioral multiplies like fn/)
  logic signed [DW-1:0] are, aim, bre, bim, tre, tim;
  logic signed [CW-1:0] wre, wim;
  logic signed [DW+CW-1:0] p_rr, p_ii, p_ri, p_ir;

  assign are  = signed'(a_q[DW-1:0]);
  assign aim  = signed'(a_q[WW-1:DW]);
  assign bre  = signed'(b_q[DW-1:0]);
  assign bim  = signed'(b_q[WW-1:DW]);
  assign wre  = signed'(rom[32'(tw_idx)][CW-1:0]);
  assign wim  = signed'(rom[32'(tw_idx)][2*CW-1:CW]);
  assign p_rr = bre * wre;
  assign p_ii = bim * wim;
  assign p_ri = bre * wim;
  assign p_ir = bim * wre;
  assign tre  = DW'((p_rr - p_ii) >>> SH);
  assign tim  = DW'((p_ri + p_ir) >>> SH);

  assign in_ready = (state == IDLE) || (state == LOAD);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= IDLE;
      out_valid <= 1'b0;
      cnt       <= '0;
    end else begin
      unique case (state)
        IDLE, LOAD: begin
          if (in_valid) begin
            mem[32'(load_addr)] <= in_data;
            if (32'(cnt) == M - 1) begin
              cnt   <= '0;
              stage <= '0;
              base  <= '0;
              k_ctr <= '0;
              state <= RDA;
            end else begin
              cnt   <= cnt + 1'b1;
              state <= LOAD;
            end
          end
        end
        RDA: begin
          a_q   <= mem[32'(a_addr)];
          state <= RDB;
        end
        RDB: begin
          b_q   <= mem[32'(b_addr)];
          state <= WRA;
        end
        WRA: begin
          mem[32'(a_addr)] <= {aim + tim, are + tre};
          state <= WRB;
        end
        WRB: begin
          mem[32'(b_addr)] <= {aim - tim, are - tre};
          if (32'(k_ctr) != 32'(span) - 1) begin
            k_ctr <= k_ctr + 1'b1;
            state <= RDA;
          end else begin
            k_ctr <= '0;
            if (32'(base) + 2 * 32'(span) < M) begin
              base  <= LOG2M'(32'(base) + 2 * 32'(span));
              state <= RDA;
            end else begin
              base <= '0;
              if (32'(stage) == LOG2M - 1) begin
                cnt   <= '0;
                state <= OUTP;
              end else begin
                stage <= stage + 1'b1;
                state <= RDA;
              end
            end
          end
        end
        OUTP: begin
          out_data  <= mem[32'(cnt[LOG2M-1:0])];
          out_valid <= 1'b1;
          state     <= OUTV;
        end
        OUTV: begin
          if (out_ready) begin
            out_valid <= 1'b0;
            if (32'(cnt) == M - 1) begin
              cnt   <= '0;
              state <= IDLE;
            end else begin
              cnt   <= cnt + 1'b1;
              state <= OUTP;
            end
          end
        end
        default: state <= IDLE;
      endcase
    end
  end

endmodule
