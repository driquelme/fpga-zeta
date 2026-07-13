"""Bit-true golden model of the pipelined RS power-sum engine (M12).

Computes S = sum_n amp_n * e^(-i * t * ln n) with host-supplied amplitudes
(amp_n = n^-sigma, |amp| <= 1) and a wide FIXED-POINT complex accumulator:
on the critical line every term is bounded by 1, so a Q(INT).(FRAC) signed
accumulator adds in a single cycle (true II=1, no floating-point adder
dependency chain). Mirrors rtl/common/zeta/rs_power_sum.sv exactly.

Accumulator format: FRAC = width+16 fractional bits (per-term conversion
truncates toward zero at 2^-FRAC), INT = 26 integer bits (N <= 2^24 terms of
magnitude <= 1), plus sign.
"""

from zetafpga.golden import cexp, fixedpt
from zetafpga.golden import mpfloat as mf
from zetafpga.golden.npow import T_AF


def frac_bits(fmt: mf.Format) -> int:
    return fmt.width + 16


def to_fxa(v: mf.MPF, fmt: mf.Format) -> int:
    """MPF -> fixed point at scale 2^FRAC, truncated toward zero. |v| <= 1."""
    if v.is_zero:
        return 0
    assert not v.is_special
    sh = frac_bits(fmt) + v.exp - fmt.width
    mag = (v.mant << sh) if sh >= 0 else (v.mant >> -sh)
    return -mag if v.sign else mag


def fxa_to_mpf(a: int, fmt: mf.Format) -> mf.MPF:
    """Normalize the accumulator into an MPF (RNE), like cexp's output stage."""
    if a == 0:
        return mf.zero(0)
    sign = 1 if a < 0 else 0
    mag = -a if sign else a
    p = mag.bit_length() - 1
    e = p - frac_bits(fmt) + 1
    w = fmt.width
    t = (mag >> (p - w)) if p >= w else (mag << (w - p))
    mant = (t + 1) >> 1
    if mant >> w:
        mant >>= 1
        e += 1
    assert fmt.emin <= e <= fmt.emax
    return mf.MPF(sign, e, mant)


def rs_power_sum(
    t_fx: int,
    entries: list[tuple[int, mf.MPF]],  # (lnn2pi Q8.bw, amplitude MPF)
    fmt: mf.Format,
    phw: int,
    bw: int,
) -> tuple[mf.MPF, mf.MPF]:
    """S = sum amp_n * e^(-i*t*ln n), returned as (re, im) MPF."""
    ccfg = cexp.load_cfg(fmt.width)
    acc_r = 0
    acc_i = 0
    for lnn2pi, amp in entries:
        phi = fixedpt.fx_mul_mod1(t_fx, lnn2pi, T_AF, bw, phw)
        c, s = cexp.cexp_turns(phi, phw, fmt, ccfg)
        pr, _, _ = mf.mpf_mul(amp, c, fmt)
        pi_, _, _ = mf.mpf_mul(amp, s, fmt)
        acc_r += to_fxa(pr, fmt)
        acc_i -= to_fxa(pi_, fmt)  # conjugate: e^(-i t ln n)
    return fxa_to_mpf(acc_r, fmt), fxa_to_mpf(acc_i, fmt)
