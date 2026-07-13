"""Bit-true golden model of rs_z_unit: fully on-chip Z(t) (M15/M16/M18).

  Z(t) = 2 Re(e^(i theta) S),  S = sum_{n<=N} n^(-1/2) e^(-i t ln n)
       + (-1)^(N-1) a^(-1/4) sum_{k<=KMAX} C_k(p) a^(-k/2),   a = t/2pi

computed in two phases mirroring rtl/common/zeta/rs_z_unit.sv step for step
(the split lets the engine derive N BEFORE running the power sum, which is
what makes grid batching (COMPUTE_ZGRID) possible with a 2-word payload):

z_prep(t):  everything that does not need S —
- 1/t (theta's tail input) from mpf_recip@W2 — Newton, no divider (M16).
- theta(t)/2pi from theta_turns, which also exports ln a at W2 — the seed of
  the division-free power chain: m = sqrt(a) = exp(ln a / 2) and
  r = a^(-1/4) = exp(-ln a / 4) via exp_mpf@W2 (a divider-less engine).
- N = floor(m) ON CHIP (M18): the engine's N is always consistent with its
  own p = m - N, which is all the RS formula requires; the host only sizes
  the table (>= floor(sqrt(t_max/2pi)) + 1 entries).
- p at W2 (fractional-part extraction must not amplify W-level rounding),
  z = 2p - 1 narrowed to the target format; C_k(p) by Clenshaw over the
  committed Chebyshev ROMs (rsck_w*.mem) — entire functions, no Psi division.

z_post(prep, S): e^(i theta) main term + the prepped correction.

Validity: t >= t_min (theta contract); the host computes Z itself below.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from zetafpga.golden import cexp, expln, recip, theta
from zetafpga.golden import mpfloat as mf

TABLES_DIR = Path(__file__).resolve().parents[3] / "rtl" / "common" / "fn" / "tables"


@dataclass(frozen=True)
class RsckCfg:
    width: int
    nc: int
    kmax: int

    @property
    def stem(self) -> str:
        return f"rsck_w{self.width}"


def load_cfg(width: int) -> RsckCfg:
    meta = json.loads((TABLES_DIR / f"rsck_w{width}.json").read_text())
    return RsckCfg(**meta)


@lru_cache(maxsize=8)
def _coeffs(stem: str, width: int) -> tuple[mf.MPF, ...]:
    fmt = mf.Format(limbs=width // 64)
    return tuple(
        mf.unpack(int(line, 16), fmt)
        for line in (TABLES_DIR / f"{stem}.mem").read_text().splitlines()
    )


def _neg(v: mf.MPF) -> mf.MPF:
    """Flip the sign bit (exact; applied to zeros too, matching the RTL)."""
    return mf.MPF(1 - v.sign, v.exp, v.mant, v.is_zero, v.is_special)


def _half(v: mf.MPF) -> mf.MPF:
    """Exponent decrement: exact /2 of a nonzero finite value."""
    assert not (v.is_zero or v.is_special)
    return mf.MPF(v.sign, v.exp - 1, v.mant)


def _int_mpf(n: int, fmt: mf.Format) -> mf.MPF:
    """Exact small-integer conversion (n < 2^width)."""
    assert 0 < n < (1 << fmt.width)
    e = n.bit_length()
    return mf.MPF(0, e, n << (fmt.width - e))


def _narrow(v: mf.MPF, fmt: mf.Format) -> mf.MPF:
    """Round an MPF@W2 (W2 = width+64) to the target format, RNE."""
    if v.is_zero or v.is_special:
        return mf.MPF(v.sign, 0, 0, v.is_zero, v.is_special)
    d = 64
    low = v.mant & ((1 << d) - 1)
    mant = v.mant >> d
    e = v.exp
    tie = 1 << (d - 1)
    if low > tie or (low == tie and (mant & 1)):
        mant += 1
        if mant >> fmt.width:
            mant >>= 1
            e += 1
    return mf.MPF(v.sign, e, mant)


def _tfx_mpf2(t_fx: int, fmt2: mf.Format) -> mf.MPF:
    """Exact Q32.32 -> MPF@W2 conversion (t_fx < 2^64 <= W2 bits)."""
    assert t_fx > 0
    e = t_fx.bit_length() - 32
    return mf.MPF(0, e, t_fx << (fmt2.width - t_fx.bit_length()))


@dataclass(frozen=True)
class ZPrep:
    """Everything the Z assembly needs besides the power sum itself."""

    theta_phi: int  # theta/2pi mod 1, PHW turns
    n: int  # floor(sqrt(t/2pi)) — the main-sum length
    corr: mf.MPF  # full signed correction term


def z_prep(t_fx: int, fmt: mf.Format) -> ZPrep:
    """Pre-sum phase: 1/t, theta, N, and the C_k correction."""
    phw = fmt.width + 32
    tcfg = theta.load_cfg(fmt.width)
    rcfg = load_cfg(fmt.width)
    fmt2 = mf.Format(limbs=tcfg.w2 // 64)
    ecfg2 = expln.load_cfg(tcfg.w2)
    coef = _coeffs(rcfg.stem, rcfg.width)

    inv_t, _, _ = recip.mpf_recip(_tfx_mpf2(t_fx, fmt2), fmt2)
    th, lnu = theta.theta_turns(t_fx, inv_t, fmt, phw, tcfg)

    # power chain at W2: m = sqrt(a), r = a^(-1/4), N = floor(m), p = m - N
    lm = _half(lnu)
    m_w2, _, _ = expln.exp_mpf(lm, fmt2, ecfg2)
    r_w2, _, _ = expln.exp_mpf(_neg(_half(lm)), fmt2, ecfg2)
    r = _narrow(r_w2, fmt)
    q, _, _ = mf.mpf_mul(r, r, fmt)
    assert m_w2.exp >= 1
    n = m_w2.mant >> (fmt2.width - m_w2.exp)
    p_w2, _, _ = mf.mpf_add(m_w2, _neg(_int_mpf(n, fmt2)), fmt2)
    tw, _, _ = mf.mpf_add(p_w2, p_w2, fmt2)
    one2 = mf.MPF(1, 1, 1 << (fmt2.width - 1))  # -1.0 @W2
    zc_w2, _, _ = mf.mpf_add(tw, one2, fmt2)
    zc = _narrow(zc_w2, fmt)
    zcd, _, _ = mf.mpf_add(zc, zc, fmt)

    # correction: Horner in q = a^(-1/2) over Clenshaw-evaluated C_k(z)
    corr = mf.zero(0)
    for k in range(rcfg.kmax, -1, -1):
        b1 = mf.zero(0)
        b2 = mf.zero(0)
        for j in range(rcfg.nc - 1, 0, -1):
            t1, _, _ = mf.mpf_mul(zcd, b1, fmt)
            t2, _, _ = mf.mpf_add(t1, _neg(b2), fmt)
            b, _, _ = mf.mpf_add(coef[k * rcfg.nc + j], t2, fmt)
            b2, b1 = b1, b
        fm, _, _ = mf.mpf_mul(zc, b1, fmt)
        f1, _, _ = mf.mpf_add(fm, _neg(b2), fmt)
        ck, _, _ = mf.mpf_add(coef[k * rcfg.nc], f1, fmt)
        hm, _, _ = mf.mpf_mul(q, corr, fmt)
        corr, _, _ = mf.mpf_add(ck, hm, fmt)

    corr, _, _ = mf.mpf_mul(r, corr, fmt)
    if n % 2 == 0:
        corr = _neg(corr)  # (-1)^(N-1)
    return ZPrep(th, n, corr)


def z_post(prep: ZPrep, s_re: mf.MPF, s_im: mf.MPF, fmt: mf.Format) -> mf.MPF:
    """Post-sum phase: Z = 2 Re(e^(i theta) S) + corr."""
    ccfg = cexp.load_cfg(fmt.width)
    c, s = cexp.cexp_turns(prep.theta_phi, fmt.width + 32, fmt, ccfg)
    m1, _, _ = mf.mpf_mul(c, s_re, fmt)
    m2, _, _ = mf.mpf_mul(s, s_im, fmt)
    main, _, _ = mf.mpf_add(m1, _neg(m2), fmt)
    main2, _, _ = mf.mpf_add(main, main, fmt)
    z, _, _ = mf.mpf_add(main2, prep.corr, fmt)
    return z
