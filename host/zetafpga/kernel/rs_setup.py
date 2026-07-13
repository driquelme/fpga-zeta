"""Riemann-Siegel host layer (M10).

The RS main sum is algebraically the Euler-Maclaurin power sum at sigma = 1/2
truncated at N = floor(sqrt(t/2pi)) terms:

  Z(t) = 2*Re( e^(i*theta(t)) * sum_{n=1..N} n^(-1/2-it) )
       + (-1)^(N-1) * (t/2pi)^(-1/4) * Psi(p) + R_1(t)

with p = frac(sqrt(t/2pi)) and Psi(p) = cos(2pi(p^2 - p - 1/16))/cos(2pi*p)
(the C0 correction; Edwards ch. 7). The engine computes the sum via the
COMPUTE_PS descriptor; theta(t) and the correction are once-per-t host
scalars (mp.siegeltheta), per the research plan.

M11: the full correction sum_{k<=K} C_k(p) (t/2pi)^(-k/2) with Gabcke's
C_1..C_4 (combinations of Psi derivatives, evaluated by mp.diff; coefficients
calibrated against mp.siegelz — error scales ~t^(-(2K+3)/4), e.g. 2.6e-11 at
t=1000 with K=4). Default K=4; K=0 reproduces the M10 behavior.

Validity: t > 2*pi (N >= 1).
"""

from functools import lru_cache

import mpmath as mp

from zetafpga.golden import mpfloat as mf
from zetafpga.kernel.em_setup import T_AF, EmProgram, mpf_from_real, mpf_value
from zetafpga.kernel.tables import lnn_entry


@lru_cache(maxsize=1 << 20)
def rs_entry(n: int, fmt: mf.Format, bw: int, sigma_value: float = 0.5) -> tuple[int, mf.MPF]:
    """Table entry for the pipelined RS engine: (ln n/2pi as Q8.bw, n^-sigma).

    The amplitude is host-precomputed (Odlyzko/Takusagawa pattern) so the
    pipelined datapath needs no on-chip exp. Contract: |amplitude| <= 1
    (sigma_value >= 0), required by the fixed-point accumulator.
    """
    _, lnn2pi = lnn_entry(n, fmt, bw)
    with mp.workprec(2 * fmt.width + 64):
        amp = mpf_from_real(mp.mpf(n) ** (-mp.mpf(sigma_value)), fmt)
    return lnn2pi, amp


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


def _psi(p: mp.mpf) -> mp.mpf:
    return mp.cos(2 * mp.pi * (p * p - p - mp.mpf(1) / 16)) / mp.cos(2 * mp.pi * p)


def _c_k(k: int, p: mp.mpf) -> mp.mpf:
    """Gabcke's C_k(p) as Psi-derivative combinations (calibrated vs siegelz)."""

    def d(order: int) -> mp.mpf:
        return mp.diff(_psi, p, order)

    pi2 = mp.pi**2
    if k == 0:
        return _psi(p)
    if k == 1:
        return -d(3) / (96 * pi2)
    if k == 2:
        return d(2) / (64 * pi2) + d(6) / (18432 * pi2**2)
    if k == 3:
        return -d(1) / (64 * pi2) - d(5) / (3840 * pi2**2) - d(9) / (5308416 * pi2**3)
    if k == 4:
        return (
            _psi(p) / (128 * pi2)
            + 19 * d(4) / (24576 * pi2**2)
            + 11 * d(8) / (5898240 * pi2**3)
            + d(12) / (2038431744 * pi2**4)
        )
    raise ValueError(f"C_{k} not implemented (K <= 4)")


def z_from_powersum(t_fx: int, re: mf.MPF, im: mf.MPF, fmt: mf.Format, kmax: int = 4) -> float:
    """Assemble Z(t) from the engine's power-sum result (host epilogue)."""
    with mp.workprec(160):
        t = mp.mpf(t_fx) / (1 << T_AF)
        a = mp.sqrt(t / (2 * mp.pi))
        n = int(mp.floor(a))
        p = a - n
        theta = mp.siegeltheta(t)
        s = mp.mpc(mpf_value(re, fmt), mpf_value(im, fmt))
        main = 2 * mp.re(mp.exp(mp.mpc(0, theta)) * s)
        tau = t / (2 * mp.pi)
        corr = (
            (-1) ** (n - 1)
            * tau ** (mp.mpf(-1) / 4)
            * sum(_c_k(k, p) * tau ** (-mp.mpf(k) / 2) for k in range(kmax + 1))
        )
        return float(main + corr)
