"""Bit-true golden model of euler_maclaurin_top: zeta(s) via Euler-Maclaurin.

Executes the exact operation sequence of the RTL FSM (same primitive calls in
the same order, so RTL comparison is bit-exact). See kernel/em_setup.py for
the formula and program construction.
"""

from dataclasses import replace

from zetafpga.golden import mpfloat as mf
from zetafpga.golden import npow
from zetafpga.kernel.em_setup import EmProgram

Cplx = tuple[mf.MPF, mf.MPF]


class _Flags:
    def __init__(self) -> None:
        self.ovf = False
        self.unf = False

    def take(self, ovf: bool, unf: bool) -> None:
        self.ovf |= ovf
        self.unf |= unf


def _int2mpf(v: int, fmt: mf.Format) -> mf.MPF:
    """Exact conversion of a small positive integer (RTL helper mirror)."""
    assert 0 < v < (1 << 24)
    e = v.bit_length()
    return mf.MPF(0, e, v << (fmt.width - e))


def _scale_half(v: mf.MPF, fmt: mf.Format, fl: _Flags) -> mf.MPF:
    """Exact multiply by 1/2 (exponent decrement, saturating)."""
    if v.is_zero or v.is_special:
        return v
    if v.exp - 1 < fmt.emin:
        fl.take(False, True)
        return mf.zero(v.sign)
    return replace(v, exp=v.exp - 1)


def _cadd(a: Cplx, b: Cplx, fmt: mf.Format, fl: _Flags) -> Cplx:
    re, o1, u1 = mf.mpf_add(a[0], b[0], fmt)
    im, o2, u2 = mf.mpf_add(a[1], b[1], fmt)
    fl.take(o1 | o2, u1 | u2)
    return re, im


def _cmul(a: Cplx, b: Cplx, fmt: mf.Format, fl: _Flags) -> Cplx:
    """(a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r) in canonical RTL order."""
    p1, o1, u1 = mf.mpf_mul(a[0], b[0], fmt)
    p2, o2, u2 = mf.mpf_mul(a[1], b[1], fmt)
    p3, o3, u3 = mf.mpf_mul(a[0], b[1], fmt)
    p4, o4, u4 = mf.mpf_mul(a[1], b[0], fmt)
    re, o5, u5 = mf.mpf_add(p1, npow.neg(p2), fmt)
    im, o6, u6 = mf.mpf_add(p3, p4, fmt)
    fl.take(o1 | o2 | o3 | o4 | o5 | o6, u1 | u2 | u3 | u4 | u5 | u6)
    return re, im


def _cscale(a: Cplx, r: mf.MPF, fmt: mf.Format, fl: _Flags) -> Cplx:
    re, o1, u1 = mf.mpf_mul(a[0], r, fmt)
    im, o2, u2 = mf.mpf_mul(a[1], r, fmt)
    fl.take(o1 | o2, u1 | u2)
    return re, im


def zeta_em(prog: EmProgram) -> tuple[mf.MPF, mf.MPF, bool, bool]:
    """Returns (re, im, ovf, unf) of zeta(sigma + i*t)."""
    fmt = prog.fmt
    fl = _Flags()

    def npow_at(idx: int, sigma: mf.MPF) -> Cplx:
        lnn_fx, lnn2pi = prog.entries[idx]
        re, im, o, u = npow.npow_s(sigma, lnn_fx, lnn2pi, prog.t_fx, fmt, prog.phw, prog.bw)
        fl.take(o, u)
        return re, im

    # ---- power sum: n = 1..N ----
    acc: Cplx = (mf.zero(0), mf.zero(0))
    for n in range(1, prog.n + 1):
        acc = _cadd(acc, npow_at(n - 1, prog.sigma), fmt, fl)
    if prog.ps_only:
        return acc[0], acc[1], fl.ovf, fl.unf

    # ---- integral term: (N+1)^(1-s)/(s-1) ----
    p = npow_at(prog.n, prog.sigma)  # (N+1)^-s
    np1 = _int2mpf(prog.n + 1, fmt)
    a = _cscale(p, np1, fmt, fl)  # (N+1)^(1-s)
    c1 = _cmul(a, (prog.inv_sm1_re, prog.inv_sm1_im), fmt, fl)
    acc = _cadd(acc, c1, fmt, fl)

    # ---- half term: + (N+1)^-s / 2 ----
    acc = _cadd(acc, (_scale_half(p[0], fmt, fl), _scale_half(p[1], fmt, fl)), fmt, fl)

    # ---- Bernoulli tail ----
    one = mf.MPF(0, 1, 1 << (fmt.width - 1))
    sigma1, o, u = mf.mpf_add(prog.sigma, one, fmt)
    fl.take(o, u)
    q = npow_at(prog.n, sigma1)  # (N+1)^(-s-1)
    u_c = _cmul((prog.sigma, prog.t_mpf), q, fmt, fl)  # s*(N+1)^(-s-1)
    for j in range(1, prog.m + 1):
        acc = _cadd(acc, _cscale(u_c, prog.bern[j - 1], fmt, fl), fmt, fl)
        if j < prog.m:
            cr1, o1, u1 = mf.mpf_add(prog.sigma, _int2mpf(2 * j - 1, fmt), fmt)
            cr2, o2, u2 = mf.mpf_add(prog.sigma, _int2mpf(2 * j, fmt), fmt)
            fl.take(o1 | o2, u1 | u2)
            u_c = _cmul(u_c, (cr1, prog.t_mpf), fmt, fl)
            u_c = _cmul(u_c, (cr2, prog.t_mpf), fmt, fl)
            u_c = _cscale(u_c, prog.inv_np2, fmt, fl)

    return acc[0], acc[1], fl.ovf, fl.unf
