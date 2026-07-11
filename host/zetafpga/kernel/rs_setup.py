"""Riemann-Siegel host layer (M10).

The RS main sum is algebraically the Euler-Maclaurin power sum at sigma = 1/2
truncated at N = floor(sqrt(t/2pi)) terms:

  Z(t) = 2*Re( e^(i*theta(t)) * sum_{n=1..N} n^(-1/2-it) )
       + (-1)^(N-1) * (t/2pi)^(-1/4) * Psi(p) + R_1(t)

with p = frac(sqrt(t/2pi)) and Psi(p) = cos(2pi(p^2 - p - 1/16))/cos(2pi*p)
(the C0 correction; Edwards ch. 7). The engine computes the sum via the
COMPUTE_PS descriptor; theta(t) and the correction are once-per-t host
scalars (mp.siegeltheta), per the research plan. Remainder with C0 only is
O(t^-3/4) — the documented M10 accuracy floor; higher C_k terms are the
follow-up milestone for precision zero-hunting at low t.

Validity: t > 2*pi (N >= 1).
"""

import mpmath as mp

from zetafpga.golden import mpfloat as mf
from zetafpga.kernel.em_setup import T_AF, EmProgram, mpf_from_real, mpf_value
from zetafpga.kernel.tables import lnn_entry


def rs_n(t_fx: int) -> int:
    with mp.workprec(96):
        t = mp.mpf(t_fx) / (1 << T_AF)
        return int(mp.floor(mp.sqrt(t / (2 * mp.pi))))


def rs_program(t_fx: int, fmt: mf.Format) -> EmProgram:
    """Power-sum-only program for the RS main sum at s = 1/2 + it."""
    n = rs_n(t_fx)
    assert n >= 1, "Riemann-Siegel needs t > 2*pi"
    phw = fmt.width + 32
    bw = phw + 32
    with mp.workprec(2 * fmt.width + 64):
        sigma = mpf_from_real(mp.mpf(1) / 2, fmt)
        t_mpf = mpf_from_real(mp.mpf(t_fx) / (1 << T_AF), fmt)
    zero = mf.zero(0)
    return EmProgram(
        fmt=fmt,
        phw=phw,
        bw=bw,
        sigma=sigma,
        t_mpf=t_mpf,
        t_fx=t_fx,
        n=n,
        m=0,
        inv_sm1_re=zero,
        inv_sm1_im=zero,
        inv_np2=zero,
        bern=(),
        entries=tuple(lnn_entry(k, fmt, bw) for k in range(1, n + 1)),
        ps_only=True,
    )


def z_from_powersum(t_fx: int, re: mf.MPF, im: mf.MPF, fmt: mf.Format) -> float:
    """Assemble Z(t) from the engine's power-sum result (host epilogue)."""
    with mp.workprec(160):
        t = mp.mpf(t_fx) / (1 << T_AF)
        a = mp.sqrt(t / (2 * mp.pi))
        n = int(mp.floor(a))
        p = a - n
        theta = mp.siegeltheta(t)
        s = mp.mpc(mpf_value(re, fmt), mpf_value(im, fmt))
        main = 2 * mp.re(mp.exp(mp.mpc(0, theta)) * s)
        psi = mp.cos(2 * mp.pi * (p * p - p - mp.mpf(1) / 16)) / mp.cos(2 * mp.pi * p)
        corr = (-1) ** (n - 1) * (t / (2 * mp.pi)) ** (mp.mpf(-1) / 4) * psi
        return float(main + corr)
