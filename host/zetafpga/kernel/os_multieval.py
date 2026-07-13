"""Odlyzko-Schonhage multi-evaluation of Z(t) on dense grids (M19).

The RS main sum S(t) = sum_{n<=N} n^(-1/2) e^(-i t ln n) is a sum of N fixed
tones; on a uniform grid t_j = t0 + j*d the phase of tone n advances by the
constant nu_n = -d ln n / 2pi turns per step. Binning each nu_n to the
nearest frequency k_n/M of a length-M DFT (offset eps_n, |eps| <= 1/2M) and
Taylor-expanding the residual rotation e^(2 pi i j' eps) to P terms turns the
whole grid into P+1 binned FFTs:

  S(t_j) = sum_p (2 pi i j')^p / p! * FFT_M[ B_p ](j' mod M),
  B_p[k] = sum_{n: k_n = k} c_n eps_n^p,      j' = j - mid (centered)

Cost O(N*P + P*M log M + J*P) instead of O(N*J) — the Odlyzko-Schonhage
amortization (M >= 4J keeps |2 pi j' eps| <= pi/8, so P = 14 reaches ~1e-16).

Numerics follow the project's wide-phase discipline, host-side:
- anchor phases c_n = n^(-1/2) e^(-i t_c ln n) come from the EXACT fixed-point
  path (lnn_entry Q8.BW tables x fx_mul_mod1), never from t_c*ln(n) in floats
  (t ln n ~ 1e7 rad would cost ~9 digits);
- per-step frequencies nu_n are O(1) — floats are exact enough there;
- theta(t_j) = theta(t_c) (one mpmath anchor per segment) + a closed-form
  increment via log1p with no large-term cancellation;
- the C_k(p) corrections reuse the committed Chebyshev tables (rsck_w*.mem)
  evaluated in float Clenshaw.

The grid is segmented on N = floor(sqrt(t/2pi)) boundaries (exact rs_n per
point); each segment gets its own anchor and FFT batch. Everything is plain
double precision on top of the exact anchors: good to ~1e-9 absolute in Z,
which is zero-*hunting* accuracy — candidates are polished by the engine
(COMPUTE_Z) or z_direct. Validity: t > 2 pi (N >= 1).
"""

import cmath
import math
from functools import lru_cache
from itertools import pairwise

import mpmath as mp

from zetafpga.golden import fixedpt, rs_z
from zetafpga.golden import mpfloat as mf
from zetafpga.kernel.em_setup import T_AF, mpf_value
from zetafpga.kernel.rs_setup import rs_n
from zetafpga.kernel.tables import lnn_entry

# Exact-phase working format: Z64's 96-bit phase window absorbs t < 2^32.
_FMT = mf.Format(limbs=1)
_PHW = _FMT.width + 32
_BW = _PHW + 32

P_TERMS = 14  # |2 pi j' eps| <= pi/8 with M >= 4J: (pi/8)^14/14! ~ 4e-17
_FACT = [float(math.factorial(p)) for p in range(P_TERMS + 1)]


@lru_cache(maxsize=4)
def _twiddles(m: int) -> tuple[complex, ...]:
    return tuple(cmath.exp(2j * math.pi * i / m) for i in range(m))


def _fft(v: list[complex]) -> list[complex]:
    """Iterative radix-2 DFT with the e^(+2 pi i jk/M) kernel (table twiddles)."""
    m = len(v)
    tw = _twiddles(m)
    out = list(v)
    j = 0
    for i in range(1, m):
        bit = m >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            out[i], out[j] = out[j], out[i]
    span = 1
    while span < m:
        stride = m // (2 * span)
        for start in range(0, m, 2 * span):
            for k in range(span):
                a = out[start + k]
                b = out[start + k + span] * tw[k * stride]
                out[start + k] = a + b
                out[start + k + span] = a - b
        span *= 2
    return out


def _anchor_phase(t_fx: int, n: int) -> float:
    """frac(t * ln n / 2pi) in turns, from the exact fixed-point path."""
    _, lnn2pi = lnn_entry(n, _FMT, _BW)
    f = fixedpt.fx_mul_mod1(t_fx, lnn2pi, T_AF, _BW, _PHW)
    return f / float(1 << _PHW)


def _main_sum_segment(t0_fx: int, dt_fx: int, j0: int, j1: int, n: int) -> list[complex]:
    """S(t_j) for j in [j0, j1) with constant N = n, via binned FFTs."""
    count = j1 - j0
    mid = count // 2
    tc_fx = t0_fx + (j0 + mid) * dt_fx
    coeffs = [
        k ** (-0.5) * cmath.exp(-2j * math.pi * _anchor_phase(tc_fx, k)) for k in range(1, n + 1)
    ]
    if count <= 2:  # direct: too short to amortize an FFT
        out = []
        for j in range(j0, j1):
            t_fx = t0_fx + j * dt_fx
            out.append(
                sum(
                    k ** (-0.5) * cmath.exp(-2j * math.pi * _anchor_phase(t_fx, k))
                    for k in range(1, n + 1)
                )
            )
        return out

    m = 1 << max(4, (4 * count - 1).bit_length())
    delta = dt_fx / float(1 << T_AF)
    bins = [[0j] * m for _ in range(P_TERMS + 1)]
    for k in range(1, n + 1):
        nu = (-delta * math.log(k) / (2 * math.pi)) % 1.0
        kk = round(nu * m)
        eps = nu - kk / m  # turns per step, |eps| <= 1/(2m)
        kk %= m
        w = coeffs[k - 1]
        for p in range(P_TERMS + 1):
            bins[p][kk] += w
            w *= eps
    g = [_fft(b) for b in bins]

    out = []
    for j in range(count):
        jp = j - mid
        idx = jp % m
        z = 2j * math.pi * jp
        acc = g[P_TERMS][idx] / _FACT[P_TERMS]
        for p in range(P_TERMS - 1, -1, -1):
            acc = acc * z + g[p][idx] / _FACT[p]
        out.append(acc)
    return out


def _theta_anchor(tc_fx: int) -> tuple[float, float, float]:
    """(theta(t_c)/2pi mod 1, ln(t_c/2pi), t_c) — the one mpmath call per segment."""
    with mp.workprec(120):
        tc = mp.mpf(tc_fx) / (1 << T_AF)
        return (
            float(mp.frac(mp.siegeltheta(tc) / (2 * mp.pi))),
            float(mp.ln(tc / (2 * mp.pi))),
            float(tc),
        )


def _theta_inc(x: float, lnc: float, tc: float) -> float:
    """theta(t_c + x) - theta(t_c) in radians, cancellation-free closed form."""
    main = 0.5 * x * lnc + 0.5 * (tc + x) * math.log1p(x / tc) - 0.5 * x
    tail = (1.0 / (tc + x) - 1.0 / tc) / 48.0
    tail -= 7.0 / 5760.0 * (1.0 / (tc + x) ** 3 - 1.0 / tc**3)
    return main + tail


@lru_cache(maxsize=4)
def _ck_float(width: int) -> tuple[tuple[tuple[float, ...], ...], int, int]:
    """The rsck Chebyshev tables as floats: (rows[k][j], nc, kmax)."""
    cfg = rs_z.load_cfg(width)
    fmt = mf.Format(limbs=width // 64)
    flat = rs_z._coeffs(cfg.stem, cfg.width)
    rows = tuple(
        tuple(float(mpf_value(flat[k * cfg.nc + j], fmt)) for j in range(cfg.nc))
        for k in range(cfg.kmax + 1)
    )
    return rows, cfg.nc, cfg.kmax


def _correction(t: float, n: int) -> float:
    """(-1)^(N-1) a^(-1/4) sum_k C_k(p) a^(-k/2) in floats (Chebyshev tables)."""
    rows, nc, kmax = _ck_float(_FMT.width)
    a = t / (2 * math.pi)
    p = math.sqrt(a) - n
    zc = 2.0 * p - 1.0
    q = a**-0.5
    corr = 0.0
    for k in range(kmax, -1, -1):
        b1 = 0.0
        b2 = 0.0
        for j in range(nc - 1, 0, -1):
            b1, b2 = rows[k][j] + 2.0 * zc * b1 - b2, b1
        ck = rows[k][0] + zc * b1 - b2
        corr = ck + q * corr
    return float((1.0 if n % 2 else -1.0) * a**-0.25 * corr)


def z_grid_os(t0_fx: int, dt_fx: int, count: int) -> list[tuple[float, float]]:
    """Z(t) at t_j = (t0_fx + j*dt_fx)/2^32 for j < count, via O-S batches."""
    assert count >= 1 and dt_fx > 0 and t0_fx > int(2 * math.pi * (1 << T_AF))
    ns = [rs_n(t0_fx + j * dt_fx) for j in range(count)]  # exact N per point

    out: list[tuple[float, float]] = []
    j0 = 0
    while j0 < count:
        j1 = j0
        while j1 < count and ns[j1] == ns[j0]:
            j1 += 1
        n = ns[j0]
        s_vals = _main_sum_segment(t0_fx, dt_fx, j0, j1, n)
        mid = j0 + (j1 - j0) // 2
        tc_fx = t0_fx + mid * dt_fx
        th_mod, lnc, tc = _theta_anchor(tc_fx)
        for j, s in zip(range(j0, j1), s_vals, strict=True):
            t = (t0_fx + j * dt_fx) / float(1 << T_AF)
            x = (j - mid) * (dt_fx / float(1 << T_AF))
            theta_turns = th_mod + _theta_inc(x, lnc, tc) / (2 * math.pi)
            eith = cmath.exp(2j * math.pi * theta_turns)
            z = 2.0 * (eith * s).real + _correction(t, n)
            out.append((t, z))
        j0 = j1
    return out


def z_direct(t_fx: int) -> float:
    """Single-point Z(t) with the same numerics (for refining O-S candidates)."""
    n = rs_n(t_fx)
    s = sum(
        k ** (-0.5) * cmath.exp(-2j * math.pi * _anchor_phase(t_fx, k)) for k in range(1, n + 1)
    )
    th_mod, _, _ = _theta_anchor(t_fx)
    t = t_fx / float(1 << T_AF)
    return float(2.0 * (cmath.exp(2j * math.pi * th_mod) * s).real + _correction(t, n))


def hunt_zeros_os(t0: float, t1: float, dt: float = 0.25, tol: float = 1e-9) -> list[float]:
    """Zeros of Z in [t0, t1]: one O-S grid pass + bisection on z_direct."""
    t0_fx = int(mp.nint(mp.mpf(t0) * (1 << T_AF)))
    dt_fx = int(mp.nint(mp.mpf(dt) * (1 << T_AF)))
    count = int((t1 - t0) / dt) + 1
    grid = z_grid_os(t0_fx, dt_fx, count)

    zeros: list[float] = []
    for (ta, za), (tb, zb) in pairwise(grid):
        if za == 0.0:
            zeros.append(ta)
            continue
        if za * zb >= 0:
            continue
        a_fx = int(mp.nint(mp.mpf(ta) * (1 << T_AF)))
        b_fx = int(mp.nint(mp.mpf(tb) * (1 << T_AF)))
        fa = za
        while (b_fx - a_fx) / (1 << T_AF) > tol:
            m_fx = (a_fx + b_fx) // 2
            fm = z_direct(m_fx)
            if fm == 0.0:
                a_fx = b_fx = m_fx
            elif fa * fm < 0:
                b_fx = m_fx
            else:
                a_fx, fa = m_fx, fm
        zeros.append((a_fx + b_fx) / 2 / float(1 << T_AF))
    return zeros
