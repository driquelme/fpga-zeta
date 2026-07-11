# fpga-zeta Design

Living document: numeric formats, error budgets, and the decision log — updated with
every design decision.

## Numeric formats

### MPF — limb-based floating point (amplitude / general values)

| Field | Width | Notes |
|---|---|---|
| sign | 1 | |
| exp | `EXPW` (default 20) | two's complement, no bias; parametric, must synthesize up to 64 |
| mantissa | `LIMBS × 64` | normalized, MSB of top limb = 1 (MPFR-style, no hidden bit) |
| flags | 2 | `is_zero`, `is_special` (inf/nan collapsed) |

- Flat `logic [MPW-1:0]` on all buses, `MPW = 1 + EXPW + LIMBS*64 + 2`.
- Rounding: RNE default; truncation allowed per-stage where the error budget covers it.
- **No subnormals** (MPFR precedent — huge emin makes gradual underflow pointless).
- **Overflow policy**: saturate to ±Inf/0 with sticky overflow/underflow flags surfaced
  to the host, so mis-scaled kernels fail loudly.

### PHASE — wide fixed point in turns

`logic [PHW-1:0]` representing phase/2π ∈ [0,1); wrap mod 1 is free. Constants
ln(n)/2π are stored to `PHW + TGUARD` bits with `TGUARD ≥ log2(t_max)`.

### FXA — fixed-point complex accumulator

Signed `ACCW`-bit per re/im, `ACCW = target mantissa + ⌈½·log2(N_max)⌉ guard + headroom`.

### Named configurations (`zeta_cfg_pkg.sv`)

| Config | LIMBS | mantissa | PHW | TGUARD | use |
|---|---|---|---|---|---|
| `Z64` | 1 | 64 b | 96 | 32 | CI fast path, bring-up |
| `Z128` | 2 | 128 b | 160 | 32 | default target |
| `Z256` | 4 | 256 b | 288 | 40 | nightly stress |

## Exponent-range policy (decided 2026-07-10)

Investigated: MPFR (word exponent, ±(2^62−1)), Arb (unlimited fmpz exponents), mpmath
(unlimited), APFP (flat 63-bit word, no overflow policy). Requirements: ζ/Z(t)/θ(t)
need <20 exponent bits even at t=10^24; the Γ/sin(πs/2) factors grow like e^(±πt/2)
(≈2.27·t bits — unrepresentable in any fixed field at large t).

**Policy (hybrid):**
1. `EXPW` parametric, default 20 (binary256-class), verified to 64 (exceeds MPFR default
   range; covers even raw Γ-factors to t ≈ 8·10^18).
2. Dangerous factors (Γ, sin(πs/2), χ, π^(s−1/2)) are computed in **log space**
   (Stirling log-gamma, log-sin) and combined before one final exp — mandated at the
   algorithm/kernel-builder layer. This is universal software practice; with it the
   datapath never sees exponents beyond ~50 bits of magnitude.
3. Sticky overflow/underflow flags to the host; optional host-side X-number escape
   (auxiliary exponent word) for offline rescaling.

Rejected: hardware multiprecision exponents (variable-length compare in the FP-add
alignment critical path, no FPGA precedent), LNS (addition tables explode past ~32-bit
words), level-index/SLI (no hardware lineage), posits (range 2^±248 too small +
tapered precision), subnormals.

## Error budgets

Per-unit worst-case bounds, to be filled as units land (hardware analogue of Arb's
ball arithmetic). RTL is only ever compared bit-exactly against the golden model; the
golden model carries the mathematical error bound vs mpmath.

| Unit | Budget (ulp of target mantissa) | Status |
|---|---|---|
| `mpf_add` | correctly rounded (≤ 0.5 ulp) | **met** (M3: exact-align window, single RNE; golden vs exact-Fraction reference, RTL bit-exact vs golden, 100k vectors × Z64/Z128/Z256) |
| `mpf_mul` | correctly rounded (≤ 0.5 ulp) | **met** (M3: exact 2W-bit product, single RNE; same evidence) |
| `sincos_turns` | ≤ 2 | **met** (M4: 1024-seg deg-5 Chebyshev fit, CF=72 guard; ≤2 ulp of Q2.62 vs mpmath over 10⁶ phases + boundaries; cardinal points exact; RTL bit-exact vs golden) |
| `fx_mul_mod1` | exact truncation to PHW bits | **met** (M4: full-product slice, bit-exact vs golden) |
| `exp_mpf` | ≤ 2 | pending (M5) |
| `log_mpf` | ≤ 2 | pending (M5) |
| `exp_mpf` | ≤ 2 | **met** (M5: FG=W+24 working bits; 10k vectors vs mpmath at L1/L2/L4 incl. saturation edges; RTL bit-exact vs golden) |
| `log_mpf` | ≤ 2 for \|ln x\| ≥ 2⁻⁸ | **met** (M5: same evidence; near x=1 absolute error ≤ 2^-(W+12), relative accuracy degrades — documented band, zeta callers use ln n ≥ ln 2; ln(1)=0 exact) |
| `cexp_turns` | ≤ 2 | **met** (M6: 1024-entry table × complex Taylor at FG bits; vs mpmath at L1/L2/L4) |
| `npow_s_kernel` | ≤ 8 of \|n^(−s)\| | **met** (M6: measured ≤ 0.72 ulp — see error-vs-t table below; RTL bit-exact vs golden at L1/L2/L4) |

### npow error vs t (Z128, empirical — validates the phase-guard design)

Measured max complex error of n^(−s) in ulps of |n^(−s)|, random σ ∈ ±[0.25, 16), n ≤ 10⁵:

| t decade | 1 | 10² | 10⁴ | 10⁶ | 10⁸ | ~4·10⁹ (t_fx max) |
|---|---|---|---|---|---|---|
| max err (ulp) | 0.89 | 0.96 | 0.85 | 0.69 | 0.79 | 0.79 |

(100 samples/decade.) Flat sub-ulp across nine decades of t: the PHW = W+32
phase window with the full-value Q8.BW ln(n)/2π tables (BW = PHW+32) absorbs
t up to 2³² exactly as designed. (Budget ≤ 8 ulp; observed < 1.)
| full ζ(s) E–M | ≥ mantissa−12 correct bits | **met** (M7: vs mp.zeta over the acceptance vector set incl. the first LMFDB zero; scaled by max(\|ζ\|, 1, integral-term conditioning) — see notes below; RTL bit-exact vs golden at Z64/Z128) |

### ζ accuracy notes (M7)

- Near zeros accuracy is **absolute** (|ζ| ≪ the O(1) sum terms; relative accuracy
  there is information-theoretically impossible and zero-finding only needs sign
  changes). Measured absolute error at the first zero: ~2^−(W−1.2) — better than the
  N·ulp accumulation bound.
- For σ < 0, E–M has intrinsic conditioning: the integral term (N+1)^(1−s)/(s−1)
  cancels against the power sum down to |ζ|, losing log₂|I/ζ| bits (measured: the
  pre-cancellation chain error is ~0.85 ulp — the engine adds nothing beyond the
  intrinsic loss). Phase 2's χ-reflection avoids E–M at σ < 1/2 entirely (as all
  software does); until then the host can escalate LIMBS for negative σ.
- **Throughput (sequential v0 engine, simulation cycle counts):** ~8.8k cycles per
  ζ evaluation at Z64 and ~19.7k at Z128 for the acceptance vectors (N ≈ 40–190,
  M = 50/98). At a nominal 250 MHz that is ~35/79 µs per evaluation per engine.
  The Phase-2 throughput plan (pipelined power sum, multiple engines) starts here.

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-07-10 | Simulation-first, board later | No purchase risk; cocotb+cocotbext-pcie de-risks the host contract |
| 2026-07-10 | Multi-precision (limb-based) from day one | Precision-vs-t scaling is the project's point; retrofitting MP is worse. First verified configs kept small (Z64/Z128) |
| 2026-07-10 | Python host stack | mpmath/Arb oracles, cocotb, fast iteration |
| 2026-07-10 | Euler–Maclaurin first, RS second | E–M covers all s and exercises every kernel; RS is a controller swap on the shared phase engine |
| 2026-07-10 | Overlay engine, not per-program bitstreams | Bitstream builds take hours; kernels = descriptor+coefficient blobs over DMA |
| 2026-07-10 | Two number types (PHASE + MPF) | Wide-phase/narrow-amplitude split matches zeta numerics; posits/RNS/LNS rejected (see above) |
| 2026-07-10 | Exponent policy: fixed parametric EXPW + log-space algorithm layer | See exponent-range section |
| 2026-07-10 | No general divider in Phase 1 | Reciprocals via exp/log or host; Newton recip is a stretch goal |
| 2026-07-10 | License posture: no copying from LGPL references | Generate coefficients/RTL from scratch; record references in third_party/NOTICE.md |
| 2026-07-10 | M1 leaf blocks (lzc, limb_addsub, limb_shift) are pure combinational | Pipelining is applied by wrappers (skid_buffer/pipe_reg) when composed into MPF operators in M3 — keeps unit verification simple and reuse maximal (fpnew-style) |
| 2026-07-10 | limb_shift saturates amount at WIDTH | Amounts > WIDTH (expressible in AMTW bits) must report all bits lost; without saturation the double-width shift silently drops bits past the 2·WIDTH window (caught by the golden-model comparison at 3 configs) |
| 2026-07-10 | Multiplier latency is a derived localparam, validity is a shift register | `limb_mul` LATENCY = TILE_LATENCY + $clog2(NT²); consumers never hardcode latencies — they read the valid output. Keeps arch-specific tiles (different DSP pipeline depths) drop-in replaceable |
| 2026-07-10 | Karatsuba handles presum carries explicitly | (a0+a1) is H+1 bits; rather than widening the inner multiplier, the carry cross-terms sa·bs·2^H, sb·as·2^H, sa·sb·2^2H are added back in the mid stage — keeps all three inner multipliers identical half-width schoolbook instances |
| 2026-07-11 | mpf_add uses an exact 2W+3-bit alignment window, not 3-bit GRS | Differences ≤ W+2 lose no bits, so one final RNE rounding is correct by construction — no guard/round/sticky case analysis to get subtly wrong. Costs a ~2W-bit adder (same class as the limb_mul tree); classic GRS is a Phase-2 area optimization if synthesis demands it |
| 2026-07-11 | Correct rounding proven against an exact rational reference | tests/test_golden_mpfloat.py rounds the exact Fraction result once (RNE) and requires golden equality incl. ovf/unf flags — stronger than an mpmath ulp comparison |
| 2026-07-11 | sincos via quadrant fold + piecewise poly, tables committed | Chebyshev-node interpolation (near-minimax, huge margin at 2^-12-turn segments) via mpmath — no sollya dependency; tables are committed artifacts regenerated by `make tables`, so builds never depend on the generator |
| 2026-07-11 | Coefficient/data tables are single-source-of-truth files | Golden model and RTL ($readmemh) read the *same* committed .mem files; a table bug cannot hide as a golden/RTL disagreement |
| 2026-07-11 | exp/log are sequential FSM units, not II=1 pipelines | Piecewise tables don't scale past ~64-bit precision. exp: k·ln2 reduction + Taylor-Horner (TERMS ≈ (W+28)/1.53 coefficients, 1/k! ROM). log: shift-and-add multiplicative normalization (FG iterations, no multiplies). Throughput deferred to M7 bottleneck analysis; interface adds in_ready |
| 2026-07-11 | log_mpf near-1 band documented, ln(1)=0 special-cased | Multiplicative normalization gives absolute (not relative) accuracy; ln x for x→1 loses relative precision. Zeta-family callers (ln n, Stirling) are outside the band |
| 2026-07-11 | ln(n)/2π tables store the FULL value (Q8.BW), not just the fractional part | For non-integer t, frac(t·(K+f)) ≠ frac(t·f): the integer part K of ln n/2π beats against t's fractional bits. Fractional-only tables silently corrupt the phase for every non-integer t — caught by comparing npow against the true n^(−s) instead of a consistent-inputs reference. fx_mul_mod1 gained BI integer bits on the b operand |
| 2026-07-11 | σ·ln n is fused in fixed point, never rounded to MPF | exp amplifies input error by \|y\|: rounding σ·ln n to W bits costs up to ~\|σ·ln n\|/2 ulp (hundreds at large σ). The host table supplies ln n at Q8.FG working precision and exp_mpf gained a fixed-point input port (fx_mode/yfx_in) |
| 2026-07-11 | Full-precision phase factor via cexp_turns; sincos_turns stays the 64-bit fast path | Table+poly sincos cannot reach 128/256-bit outputs; cexp_turns (1024-entry full-precision table × short complex Taylor) delivers any-LIMBS accuracy for the n^(−s) kernel |
| 2026-07-11 | v0 ISA: 4×64-bit-word descriptors over a plain word stream; batching = repeated COMPUTE_EM | Stride-batch (Δs) descriptors and 256-bit alignment deferred until the PCIe DMA framing lands (Phase 2); unknown opcodes must carry no payload (resync rule); READBACK clears the result buffer and sticky flags |
| 2026-07-11 | Tables are runtime WRITE_TABLE payloads, engine RAMs behavioral | lnn/bern tables are per-program data streamed by the host, never bitstream ROMs; the behavioral RAMs move behind the ram_sdp tile at synthesis time |
| 2026-07-11 | GoldenBackend is a full ISA interpreter, byte-equivalent to the RTL engine | Two independent implementations of the descriptor contract must produce identical readback bytes for identical programs — an ISA bug cannot hide in either side. Apps run on the Backend contract and are RTL-ready by construction |
| 2026-07-11 | Verilator socket harness + cocotbext-pcie deferred M9 → Phase 2 | Both exist to exercise DMA framing and batch throughput, which arrive with the PCIe milestone; in Phase 1 the byte-equivalence test carries the same-app-bytes acceptance. Scope deviation from the original plan — flagged for review |
| 2026-07-11 | Zero location via \|ζ\| golden-section (no θ(t) yet) | Z(t) sign-change bisection needs θ(t)/log-gamma (Phase 2); minimizing \|ζ(½+it)\| on brackets from a coarse grid locates the first 10 zeros to ~10⁻⁷ of LMFDB — sufficient for Phase-1 demos |
| 2026-07-11 | M10: Riemann–Siegel = COMPUTE_PS + host epilogue | The RS main sum is the E-M power sum at σ=½ with N=⌊√(t/2π)⌋ — the ζ core gains only a ps_only early exit (~10 lines); θ(t) (mp.siegeltheta) and the C0=Ψ(p) correction are once-per-t host scalars. Zero changes to the datapath |
| 2026-07-11 | M10 accuracy floor: C0-only remainder O(t^(-3/4)) | Z(t) vs mpmath.siegelz within 0.2·t^(-3/4); sign-change zeros to ~10⁻³ at t≈14–50 (N is only 1–2 terms there) in 0.33s vs ~30s for E-M \|ζ\| minimization. Higher C_k corrections (Ψ derivatives, Chebyshev ROMs) are the next accuracy milestone; the speed advantage grows as √t vs t |
