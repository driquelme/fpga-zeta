"""Host-side Euler-Maclaurin program construction (the M8 kernel-builder core).

Given s = sigma + i*t and a format, selects the truncation parameters N, M,
and precomputes everything the engine cannot (or should not) compute itself:
1/(s-1) and 1/(N+1)^2 (no divider in Phase 1), t as MPF, the Bernoulli
coefficients B_2j/(2j)!, and the ln(n) table entries for n = 1..N+1.

The formula computed by the engine (Johansson's Hurwitz form with a = 1):

  zeta(s) = sum_{n=1}^{N} n^-s                          (power sum)
          + (N+1)^(1-s) / (s-1)                          (integral term)
          + (N+1)^-s / 2                                 (half term)
          + sum_{j=1}^{M} B_2j/(2j)! (s)_{2j-1} (N+1)^(-s-2j+1)   (tail)

N/M selection: M ~ 0.75*prec; N sized so the remainder ratio
(2*pi*(N+1)/(|t|+2M))^(2M) clears 2^-(prec+16). Validated empirically against
mpmath over the acceptance vector set.
"""

import math
from dataclasses import dataclass

import mpmath as mp

from zetafpga.golden import mpfloat as mf
from zetafpga.kernel.tables import lnn_entry

T_AF = 32


def mpf_from_real(x: mp.mpf, fmt: mf.Format) -> mf.MPF:
    """RNE conversion of an mpmath real to MPF (saturating)."""
    if x == 0:
        return mf.zero(0)
    sign = 1 if x < 0 else 0
    a = abs(x)
    e = int(mp.floor(mp.log(a, 2))) + 1
    while a >= mp.mpf(2) ** e:
        e += 1
    while a < mp.mpf(2) ** (e - 1):
        e -= 1
    mant = int(mp.nint(a * mp.mpf(2) ** (fmt.width - e)))
    if mant >> fmt.width:
        mant >>= 1
        e += 1
    if e > fmt.emax:
        return mf.special(sign)
    if e < fmt.emin:
        return mf.zero(sign)
    return mf.MPF(sign, e, mant)


def mpf_value(v: mf.MPF, fmt: mf.Format) -> mp.mpf:
    if v.is_zero:
        return mp.mpf(0)
    x = mp.mpf(v.mant) * mp.mpf(2) ** (v.exp - fmt.width)
    return -x if v.sign else x


def select_nm(t_abs: float, prec: int) -> tuple[int, int]:
    """Truncation parameters: remainder below ~2^-(prec+16)."""
    m = math.ceil(0.75 * prec) + 2
    r = 2.0 ** ((prec + 16) / (2 * m))
    n = math.ceil(r * (t_abs + 2 * m) / (2 * math.pi)) + 10
    return n, m


@dataclass(frozen=True)
class EmProgram:
    fmt: mf.Format
    phw: int
    bw: int
    sigma: mf.MPF
    t_mpf: mf.MPF
    t_fx: int
    n: int
    m: int
    inv_sm1_re: mf.MPF
    inv_sm1_im: mf.MPF
    inv_np2: mf.MPF
    bern: tuple[mf.MPF, ...]  # B_2j/(2j)!, j = 1..m
    entries: tuple[tuple[int, int], ...]  # (lnn_fx, lnn2pi) for n = 1..N+1
    ps_only: bool = False  # COMPUTE_PS: stop after the power sum


def build_program(sigma: mf.MPF, t_fx: int, fmt: mf.Format) -> EmProgram:
    """sigma and t (as Q32.32) are the exact engine inputs."""
    phw = fmt.width + 32
    bw = phw + 32
    with mp.workprec(2 * fmt.width + 64):
        sv = mpf_value(sigma, fmt)
        tv = mp.mpf(t_fx) / (1 << T_AF)
        n, m = select_nm(float(tv), fmt.width)
        s = mp.mpc(sv, tv)
        inv = 1 / (s - 1)
        inv_np2 = 1 / mp.mpf(n + 1) ** 2
        bern = tuple(
            mpf_from_real(mp.bernoulli(2 * j) / mp.factorial(2 * j), fmt) for j in range(1, m + 1)
        )
        prog = EmProgram(
            fmt=fmt,
            phw=phw,
            bw=bw,
            sigma=sigma,
            t_mpf=mpf_from_real(tv, fmt),
            t_fx=t_fx,
            n=n,
            m=m,
            inv_sm1_re=mpf_from_real(inv.real, fmt),
            inv_sm1_im=mpf_from_real(inv.imag, fmt),
            inv_np2=mpf_from_real(inv_np2, fmt),
            bern=bern,
            entries=tuple(lnn_entry(k, fmt, bw) for k in range(1, n + 2)),
        )
    return prog
