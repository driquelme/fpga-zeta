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
| `theta_turns` | ≤ 2^−(W+8) turns absolute (mod 1) | **met** (M14: vs mp.siegeltheta from t_min to 4·10⁹ at Z64/Z128; W2 = W+64 internal precision; RTL bit-exact vs golden; valid t ≥ t_min = 18/32/46/61) |
| on-chip Z(t) (`rs_z_unit`) | RS truncation bound 0.1·t^(−11/4) + 2^−(W−16) | **met** (M15/M18: golden Z vs mp.siegelz from t_min to 10⁵ at Z64/Z128; RTL bit-exact vs golden; the engine's own error ~2^−(W−8), dominated by the K=4 remainder except at very large t) |
| `mpf_recip` | ≤ 2 ulp; powers of 2 exact | **met** (M16: vs exact-Fraction reference, 2000 vectors × limbs 1/2/3/4 + directed extremes; RTL bit-exact vs golden) |
| `fft_radix2` | ≤ log2(M)·2^(−FRAC+3) + l1·2^(−52) abs | **met** (M20: worst 2.3×10⁻¹⁵ at M=256, FRAC=54, l1 ≤ 32 vs float FFT reference; RTL bit-exact vs golden at M=64/256) |

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
| 2026-07-11 | M11: Gabcke C₁..C₄ corrections in the host epilogue (kmax=4 default) | C_k as Ψ-derivative combinations via mp.diff, coefficients calibrated against mp.siegelz before adoption (error scaling verified per order: 2.6·10⁻¹¹ at t=10³, K=4). Zeros vs LMFDB improved ~10⁻³ → ~10⁻⁶ (zero 1, N=1) to ~10⁻⁷ (t≈40+), keeping the 100× RS speed. Hardware C_k (Chebyshev-in-p ROMs, Takusagawa pattern) deferred until θ(t) moves on-chip |
| 2026-07-11 | M12: pipelined RS engine — host-supplied amplitudes + FXA accumulator | rs_power_sum consumes one term/cycle: comb phase → cexp_pipe (Horner unrolled into TERMS−1 stages, bit-identical to cexp_turns) → 2× mpf_mul → MPF→Q27.(W+16) fixed-point accumulate (single-cycle add; valid because \|terms\| ≤ 1 on the critical line — the plan's original FXA idea). Amplitude n^(−σ) comes from the host table (Odlyzko/Takusagawa pattern), removing exp from the hot loop. Measured: N=126 in 148 cycles vs ~16.4k sequential — 110× at II=1 |
| 2026-07-11 | M13: COMPUTE_RS (op 7) + TBL_RS (id 2) integrate the pipelined engine into the ISA | RS entry = {amp MPF, ln n/2π Q8.BW}; COMPUTE_RS payload is one word (t_fx), N in the header. zeta_engine feeds rs_power_sum from the RS RAM at one entry/cycle. GoldenBackend interprets via golden rs_pipe; the mixed EM/PS/RS equivalence program is byte-identical RTL↔interpreter at Z64/Z128. apps/zfunc.py now rides the fast path end to end. Future nicety: bisect at K=1, polish at K=4 (the Ψ-derivative epilogue dominates host time) |
| 2026-07-11 | M14: θ(t) on chip — one limb wider internally (W2 = W+64), main term in mod-1 fixed point | θ/2π = (u/2)(ln u − 1) − 1/16 + Σ c_k/2π · t^(1−2k), u = t/2π. The t·ln t magnitude costs ~log₂(t) ≤ 32 mantissa bits, so ln u runs on a log_mpf instance at W2 and the main term is formed in Q(38).(FG2) fixed point where the mod-1 wrap is exact (same wide-phase discipline as npow), then truncated to PHW turns. Tail = MPF@W2 Horner in v = 1/t² over ROM'd c_k/2π (Bernoulli-derived, gen_theta.py). Accuracy: ≤2^−(W+8) turns absolute vs mp.siegeltheta from t_min to 4·10⁹, bit-exact RTL↔golden at Z64/Z128 |
| 2026-07-11 | θ validity floor t_min = ⌈(W+16)·ln2/π⌉ (18/32/61 for Z64/Z128/Z256); 1/t host-supplied | Below t_min the dropped asymptotic remainder e^(−πt) exceeds the format's budget — the host computes θ itself there (documented contract, stored in theta_w*.json). inv_t as MPF@W2 comes from the host because the engine has no divider; an on-chip Newton reciprocal is the standing stretch item |
| 2026-07-11 | limb_mul reduction tree fixed for non-power-of-2 tile counts; limbs=3 made a permanent test config | The registered reduction tree paired elements as NPP/2^level, silently dropping the odd element (LIMBS2=3 → NT=6 → a 9-element level lost a partial product). Fixed with ceil-halving level counts and odd-element passthrough. Latent since M2 — every previously tested config had power-of-2 NT — exposed by θ's LIMBS+1 internal width. tb/conftest.py limbs fixture is now [1, 2, 3, 4] so odd tree shapes stay covered forever |
| 2026-07-12 | M15: fully on-chip Z(t) — COMPUTE_Z (op 8) = RS main sum + rs_z_epilogue | Z = 2Re(e^{iθ}S) + (−1)^{N−1}a^{−1/4}ΣC_k(p)a^{−k/2}. The host epilogue (mp.siegeltheta + Ψ-derivative C_k via mp.diff — the dominant host cost since M13) moves on chip: θ from theta_turns, e^{iθ} from cexp_turns, the corrections from Chebyshev ROMs. Payload = t_fx + 1/t (MPF@W2); N in the header; result = (Z, packed zero). Host fallback below t_min or for kmax ≠ 4 (zfunc auto-selects) |
| 2026-07-12 | Division-free RS correction: C_k(p) as Chebyshev-in-z Clenshaw ROMs, powers of a via exp/log | The classical Ψ(p) cosine quotient needs a divider; but C_k are entire, so gen_rsck.py fits Ψ once at Chebyshev nodes (pointwise evals only), takes Ψ′..Ψ⁽¹²⁾ by exact Chebyshev differentiation, and combines per Gabcke — cross-checked against the mp.diff-based rs_setup._c_k to 2^−(W+8). Clenshaw (O(1) coefficients) is used at runtime because monomial conversion at degree ~100 amplifies by ~2^degree — catastrophic at W bits. nc = 37/58/79/97 terms for W = 64/128/192/256. √a and a^{−1/4} are exp(±ln a/2,4) on one exp_mpf@W2 — no divider, no sqrt unit |
| 2026-07-12 | p = frac(√(t/2π)) formed at W2; theta_turns exports ln(t/2π) | Extracting the fractional part of m = √a amplifies absolute error by m ≤ 2^15, so the p-chain (m, p, z = 2p−1) runs at W2 before narrowing (RNE) to W — epilogue error stays ~2^−(W−8) instead of ~2^−(W−20) at large t. ln a is exported from theta_turns (computed there anyway at W2) instead of instantiating a second log unit. Width-192 θ/rsck tables added so limbs=3 stays a first-class engine test config |
| 2026-07-12 | M16: mpf_recip — Newton reciprocal, ROM-free; 1/t derived on chip | y ← y(2 − my) at F = W+8 guard bits from the 48/17 − 32/17·m minimax seed (rel err 2^−4.09, doubling per step; NITER = clog2(W) over-converges). No divider, no table: the seed constants are elaboration-time wide divisions. ≤ 2 ulp, exact powers of 2, proven vs exact-Fraction reference at limbs 1–4, RTL bit-exact. COMPUTE_Z payload shrinks to t_fx alone |
| 2026-07-12 | M17: rs_power_sum_tiled — L lanes over a banked RS table, exact FXA merge | Entry i lives in bank i mod L (power-of-2 L); lane l sums its stripe into its own Q27.FRAC accumulator; the wrapper merges by plain integer addition — bit-identical to the sequential sum because fixed-point adds are order-independent, so the golden model and the readback bytes are UNCHANGED at any lane count (equivalence test runs RS_LANES=2). Measured: N=126 in ≤123 cycles at 4 lanes vs 148 single-lane; throughput ceil(N/L) + drain |
| 2026-07-12 | M18: prep/post split of the Z unit; N = floor(√(t/2π)) derived on chip; COMPUTE_ZGRID | rs_z_unit runs recip → θ → power chain → N → C_k correction BEFORE the power sum (prep_valid presents n_out), then assembles Z after sum_valid — the engine no longer needs N from the host, and consistency is guaranteed: the N in the main sum is by construction the floor of the same m used for p (at a rounding boundary that consistent pair is exactly what the RS formula requires). Host contract reduces to sizing the RS table (⌊√(t_max/2π)⌋+1 entries) and honoring t ≥ t_min |
| 2026-07-12 | COMPUTE_ZGRID (op 9): J = count Z evaluations from (t0, dt) — the O–S evaluation pattern | One 2-word descriptor evaluates Z on a uniform grid with zero per-point host work (1/t and N on chip). zfunc's coarse zero-scan rides it (below-t_min prefix and bisection midpoints fall back to COMPUTE_Z/host-epilogue singles). The FFT main-sum amortization of full Odlyzko–Schönhage (binning ln n onto a frequency grid + band-limited interpolation) remains the open Phase-2 item — ZGRID is its evaluation substrate |
| 2026-07-12 | M19: Odlyzko–Schönhage FFT multi-evaluation — host algorithm layer (kernel/os_multieval.py) | Grid Z via P+1 binned FFTs: tone n's per-step phase advance ν = −δ·ln n/2π is binned to the nearest k/M (M ≥ 4J), the |ε| ≤ 1/2M offset handled by a P=14 Taylor expansion of e^{2πij′ε} around the centered index — cost O(N·P + P·M log M + J·P) vs O(N·J). Dependency-free radix-2 FFT with table twiddles. Anchors follow the wide-phase discipline host-side: c_n phases from the EXACT fixed-point path (lnn2pi × fx_mul_mod1 — floats on t·ln n ~ 10⁷ rad would cost 9 digits); θ from one mpmath call per segment + a log1p closed-form increment (no large-term cancellation); C_k(p) from the committed rsck Chebyshev tables in float Clenshaw. Grid segmented on exact N boundaries |
| 2026-07-12 | M19 accuracy/perf: double-precision hunting layer, exact-arithmetic polish | O–S main sum matches the direct tone sum < 10⁻⁹; Z matches mp.siegelz within 0.1·t^(−11/4) + 3·10⁻⁹ incl. across N-boundary splits; first 10 zeros ~10⁻⁷ of LMFDB in 0.08 s, 36 zeros of a t≈10⁴ band in 0.18 s (pure Python). Division of labor: O–S floats locate, the engine's exact path (COMPUTE_Z / z_direct) polishes. Hardware mapping: the binning pass is the RS table walk + fx_mul_mod1 the engine already has; the FFT stage is the M20 butterfly engine |
| 2026-07-13 | M21: COMPUTE_OS (op 10) — the O–S binned-FFT grid main sum on chip | os_grid_sum: one RS-table sweep bins each tone (anchor cₙ from fx_mul_mod1 + cexp_turns at Q9.54; rate offset ε̂ = ε·M in bin units Q1.62 — scaled so all 15 bin arrays are O(1), with the compensating u = 2πj′/M applied in the combine, otherwise εᵖ underflows fixed point); fft_radix2 runs 15× in place; per-point complex Horner in u with ROM'd 1/p! (fft_os.mem); rs_acc_norm at FRAC=54 emits MPF pairs. Payload (t0, dt, N), count = J. Contracts: one N segment per batch, J ≤ OS_M/4, l1 < 2⁹. v1 returns S(t_j); θ/corrections stay on the host (M19 float layer) — on-chip θ/p polynomial prep is the flagged v2. Golden os_pipe.py < 2×10⁻⁹ vs the direct tone sum; RTL bit-exact at Z64/Z128; engine byte-identical at 8 ops incl. OS. Found: stale same-cycle register feeding fx_mul_mod1 (fixed: live entry inputs) |
| 2026-07-12 | M20: fft_radix2 — sequential in-place complex fixed-point FFT, no scaling passes | The O–S butterfly stage in RTL: radix-2 DIT over an internal RAM, bit-reversed loading, 4 cycles/butterfly, Q2.62 twiddle ROMs (gen_fft.py; kernel e^{+2πijk/M} matching os_multieval), t = (w·b) >>> 62 floor-truncated, adds exact. Key sizing fact: every DIT intermediate is a DFT of a *subset* of inputs, so magnitudes are bounded by the input l1 norm — no per-stage scaling needed; O–S bins carry l1 ≤ ~2√N, so ~7 integer bits of headroom in the 64-bit components suffice (caller contract: l1 < 2^(DW−1)). Error ~1 truncation lsb/component/stage + 2^−62 twiddles — measured < 2.3×10⁻¹⁵ at M=256 with 54 fractional bits, comparable to doubles. Golden fft.py bit-exact (Python >> floor ≡ SV >>>); RTL bit-exact at M=64/256 (M=1024/4096 ROMs committed for the integration milestone) |
