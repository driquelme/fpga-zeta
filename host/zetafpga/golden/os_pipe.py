"""Bit-true golden model of os_grid_sum: on-chip binned-FFT grid sum (M21).

S(t_j) = sum_{n<=N} amp_n e^(-i t_j ln n) for J grid points t_j = t0 + j*dt,
computed the Odlyzko-Schonhage way, all in engine fixed point:

- anchor c_n at the grid center from the exact phase path (fx_mul_mod1 +
  cexp_turns + amp muls), converted to Q9.54 complex (FRAC = 54; bins carry
  l1 <= 2 sqrt(N) < 2^9);
- per-step rate nu_n = frac(-dt ln n / 2pi) from a second fx_mul_mod1, split
  into bin k (round to M grid) and offset EPS-HAT = eps*M in [-1/2, 1/2)
  "bin units" at Q1.62 — scaled so every bin array is O(1), with the
  compensating u = 2 pi j'/M applied in the combine;
- P+1 = 15 bin arrays -> fft_fx (golden/fft.py) -> per-point complex Horner
  sum_p (i u)^p/p! G_p[j' mod M] with ROM'd 1/p! (fft_os.mem).

Contract: constant N across the batch (host slices at N boundaries),
J <= M/4 (keeps |u| <= pi/4 so P = 14 reaches ~1e-16), l1 < 2^9.
Mirrors rtl/common/zeta/os_grid_sum.sv step for step.
"""

from functools import lru_cache
from pathlib import Path

from zetafpga.golden import cexp, fft, fixedpt
from zetafpga.golden import mpfloat as mf
from zetafpga.kernel.em_setup import T_AF

TABLES_DIR = Path(__file__).resolve().parents[3] / "rtl" / "common" / "fn" / "tables"

P_TERMS = 14
FRAC = 54  # Q9.54 bin/data fixed point


@lru_cache(maxsize=1)
def _os_consts() -> tuple[tuple[int, ...], int]:
    """(1/p! at Q2.62 for p <= 14, 2*pi at scale 2^60)."""
    vals = []
    for line in (TABLES_DIR / "fft_os.mem").read_text().splitlines():
        v = int(line, 16)
        if v >> 63:
            v -= 1 << 64
        vals.append(v)
    return tuple(vals[: P_TERMS + 1]), vals[P_TERMS + 1]


def to_fx54(v: mf.MPF) -> int:
    """MPF -> Q9.54 signed, truncated toward zero (|v| <= 1 per entry)."""
    if v.is_zero:
        return 0
    assert not v.is_special
    w = v.mant.bit_length()  # == width (normalized)
    sh = FRAC + v.exp - w
    mag = (v.mant << sh) if sh >= 0 else (v.mant >> -sh)
    return -mag if v.sign else mag


def fx54_to_mpf(a: int, fmt: mf.Format) -> mf.MPF:
    """Q9.54 -> MPF, RNE (the rs_acc_norm semantics at FRAC = 54)."""
    if a == 0:
        return mf.zero(0)
    sign = 1 if a < 0 else 0
    mag = -a if sign else a
    p = mag.bit_length() - 1
    e = p - FRAC + 1
    w = fmt.width
    t = (mag >> (p - w)) if p >= w else (mag << (w - p))
    mant = (t + 1) >> 1
    if mant >> w:
        mant >>= 1
        e += 1
    return mf.MPF(sign, e, mant)


def os_grid_sum(
    t0_fx: int,
    dt_fx: int,
    n: int,
    count: int,
    entries: list[tuple[int, mf.MPF]],
    fmt: mf.Format,
    m: int,
) -> list[tuple[int, int]]:
    """S(t_j) as (re, im) Q9.54 pairs for j < count. entries: (lnn2pi, amp)."""
    assert count >= 1 and 4 * count <= m and n >= 1
    phw = fmt.width + 32
    bw = phw + 32
    log2m = m.bit_length() - 1
    shift = phw - log2m
    assert shift > 62
    mid = count >> 1
    tc_fx = (t0_fx + mid * dt_fx) & ((1 << 64) - 1)
    ccfg = cexp.load_cfg(fmt.width)
    invf, twopi60 = _os_consts()

    bins = [[(0, 0)] * m for _ in range(P_TERMS + 1)]
    for lnn2pi, amp in entries[:n]:
        phi = fixedpt.fx_mul_mod1(tc_fx, lnn2pi, T_AF, bw, phw)
        c, s = cexp.cexp_turns(phi, phw, fmt, ccfg)
        pr, _, _ = mf.mpf_mul(amp, c, fmt)
        pi_, _, _ = mf.mpf_mul(amp, s, fmt)
        wre = to_fx54(pr)
        wim = -to_fx54(pi_)  # conjugate: e^(-i t ln n)

        nu = fixedpt.fx_mul_mod1(dt_fx, lnn2pi, T_AF, bw, phw)
        nu = (-nu) & ((1 << phw) - 1)  # rate is -dt ln n / 2pi
        kfull = (nu + (1 << (shift - 1))) >> shift
        k = kfull & (m - 1)
        e62 = (nu - (kfull << shift)) >> (shift - 62)  # eps-hat, Q1.62

        for p in range(P_TERMS + 1):
            br, bi = bins[p][k]
            bins[p][k] = (br + wre, bi + wim)
            wre = (wre * e62) >> 62
            wim = (wim * e62) >> 62

    fcfg = fft.load_cfg(m)
    g = [fft.fft_fx(bins[p], fcfg) for p in range(P_TERMS + 1)]

    out = []
    for j in range(count):
        jp = j - mid
        idx = jp % m
        y62 = (jp * twopi60) >> (log2m - 2)  # u = j' 2pi/M at Q2.62
        ar = (g[P_TERMS][idx][0] * invf[P_TERMS]) >> 62
        ai = (g[P_TERMS][idx][1] * invf[P_TERMS]) >> 62
        for p in range(P_TERMS - 1, -1, -1):
            nr = -((ai * y62) >> 62)  # acc * (i u)
            ni = (ar * y62) >> 62
            ar = nr + ((g[p][idx][0] * invf[p]) >> 62)
            ai = ni + ((g[p][idx][1] * invf[p]) >> 62)
        out.append((ar, ai))
    return out
