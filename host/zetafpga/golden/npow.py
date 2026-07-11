"""Bit-true golden model of npow_s_kernel: complex n^(-s) (M6).

n^(-s) = exp(-sigma*ln n) * (cos(t*ln n) - i*sin(t*ln n))

Composition, in the exact RTL sequence:
  amplitude: sigma -> fixed point (FG frac bits), yfx = -(sigma_fx * lnn_fx)
             at full working precision, then exp_fx. Keeping this product in
             extended fixed point is essential: rounding sigma*ln(n) to a
             W-bit MPF first would inject 0.5 ulp that the exponential
             amplifies by |sigma*ln n| (hundreds of ulps at large sigma).
  phase:     fx_mul_mod1(t_fx, frac(ln n/2pi)) -> cexp_turns -> conjugate
  assemble:  mpf_mul(a, cos), -mpf_mul(a, sin)

Inputs:
  sigma:   MPF (any representable value; |sigma| beyond e-range saturates)
  lnn_fx:  ln(n) as unsigned fixed point Q8.FG (host-generated table entry)
  lnn2pi:  frac(ln n / 2pi) as a BW-bit pure fraction, BW = PHW + 32
  t_fx:    t as unsigned Q32.32 (t >= 0; negative t via conjugate symmetry)
"""

from dataclasses import replace

from zetafpga.golden import cexp, expln, fixedpt
from zetafpga.golden import mpfloat as mf

T_AF = 32  # fractional bits of the t operand (Q32.32)


def neg(v: mf.MPF) -> mf.MPF:
    return replace(v, sign=1 - v.sign)


def sigma_to_fx(sigma: mf.MPF, fmt: mf.Format, fg: int) -> int:
    """Signed fixed point at scale 2^FG (exp_mpf's IDLE conversion)."""
    if sigma.is_zero:
        return 0
    s = fg + sigma.exp - fmt.width
    mag = (sigma.mant << s) if s >= 0 else (sigma.mant >> -s)
    return -mag if sigma.sign else mag


def npow_s(
    sigma: mf.MPF,
    lnn_fx: int,
    lnn2pi: int,
    t_fx: int,
    fmt: mf.Format,
    phw: int,
    bw: int,
) -> tuple[mf.MPF, mf.MPF, bool, bool]:
    """Returns (re, im, ovf, unf) of n^(-s), s = sigma + i*t."""
    ecfg = expln.load_cfg(fmt.width)
    ccfg = cexp.load_cfg(fmt.width)
    fg = ecfg.fg

    if sigma.is_special:
        sp = mf.special(sigma.sign)
        return sp, sp, False, False

    # Amplitude: yfx = -(sigma * ln n) at scale 2^FG, clamped to the
    # exp saturation range so the product width stays bounded.
    sig_fx = sigma_to_fx(sigma, fmt, fg)
    if sigma.exp > 30:  # |sigma| >= 2^29: |sigma*ln2| alone saturates exp
        yfx = -(1 << (fg + 22)) if sigma.sign == 0 else (1 << (fg + 22))
        if lnn_fx == 0:  # n = 1: 1^(-s) = 1 regardless of sigma
            yfx = 0
    else:
        yfx = -((sig_fx * lnn_fx) >> fg)
    a, ovf, unf = expln.exp_fx(yfx, fmt, ecfg)

    # Phase.
    phi = fixedpt.fx_mul_mod1(t_fx, lnn2pi, T_AF, bw, phw)
    c, s = cexp.cexp_turns(phi, phw, fmt, ccfg)

    if a.is_special:
        return mf.special(0), mf.special(0), ovf, unf
    re, o3, u3 = mf.mpf_mul(a, c, fmt)
    im, o4, u4 = mf.mpf_mul(a, s, fmt)
    return re, neg(im), ovf | o3 | o4, unf | u3 | u4
