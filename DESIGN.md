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
| `sincos_turns` | ≤ 2 | pending (M4) |
| `exp_mpf` | ≤ 2 | pending (M5) |
| `log_mpf` | ≤ 2 | pending (M5) |
| `npow_s_kernel` | ≤ 8 | pending (M6) |
| full ζ(s) E–M | ≥ mantissa−12 correct bits | pending (M7) |

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
