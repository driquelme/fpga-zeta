"""Golden cexp_turns and npow_s vs mpmath, including error-vs-t characterization.

The npow budget is <= 8 ulp of |n^(-s)| (DESIGN.md), measured as complex
distance |got - ref| relative to the ulp of the reference magnitude.
"""

import os
import random

import mpmath as mp
import pytest

from zetafpga.golden import cexp, npow
from zetafpga.golden import mpfloat as mf
from zetafpga.kernel.tables import lnn_entry

FMTS = [mf.Format(limbs=1), mf.Format(limbs=2), mf.Format(limbs=4)]
N = int(os.environ.get("ZETA_TEST_N", "500"))


def _phw(fmt: mf.Format) -> int:
    return fmt.width + 32


def _val(v: mf.MPF, fmt: mf.Format) -> mp.mpf:
    if v.is_zero:
        return mp.mpf(0)
    x = mp.mpf(v.mant) * mp.mpf(2) ** (v.exp - fmt.width)
    return -x if v.sign else x


@pytest.fixture(autouse=True)
def _prec() -> None:
    mp.mp.prec = 700


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_cexp_accuracy(fmt: mf.Format) -> None:
    ccfg = cexp.load_cfg(fmt.width)
    phw = _phw(fmt)
    rng = random.Random(fmt.width)
    ulp = mp.mpf(2) ** (1 - fmt.width)
    phis = [rng.getrandbits(phw) for _ in range(N)]
    phis += [0, 1, (1 << phw) - 1, 1 << (phw - 1), 1 << (phw - cexp.load_cfg(fmt.width).segw)]
    for phi in phis:
        c, s = cexp.cexp_turns(phi, phw, fmt, ccfg)
        t = mp.mpf(phi) / mp.mpf(2) ** phw
        err = abs(mp.mpc(_val(c, fmt), _val(s, fmt)) - mp.expjpi(2 * t))
        assert err <= 2 * ulp, f"phi={phi:#x}: err {mp.nstr(err / ulp, 4)} ulp"


def _npow_err(fmt: mf.Format, sigma: mf.MPF, n: int, t_fx: int, phw: int, bw: int) -> mp.mpf:
    lnn_fx, frac = lnn_entry(n, fmt, bw)
    re, im, ovf, unf = npow.npow_s(sigma, lnn_fx, frac, t_fx, fmt, phw, bw)
    assert not (ovf or unf)
    # Reference against the TRUE n^(-s) for the (exactly representable)
    # sigma, t inputs — the extended-precision tables make input rounding
    # negligible, so the honest end-to-end comparison is the right one.
    sv = _val(sigma, fmt)
    tv = mp.mpf(t_fx) / (1 << npow.T_AF)
    ref = mp.exp(-mp.mpc(sv, tv) * mp.ln(n))
    got = mp.mpc(_val(re, fmt), _val(im, fmt))
    ulp = abs(ref) * mp.mpf(2) ** (1 - fmt.width)
    return abs(got - ref) / ulp


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_npow_accuracy(fmt: mf.Format) -> None:
    phw, bw = _phw(fmt), _phw(fmt) + 32
    rng = random.Random(100 + fmt.width)
    for _ in range(N // 4):
        n = rng.randrange(2, 1_000_000)
        mant = (1 << (fmt.width - 1)) | rng.getrandbits(fmt.width - 1)
        sigma = mf.MPF(rng.getrandbits(1), rng.randrange(-3, 5), mant)
        t_fx = rng.getrandbits(52)  # t up to ~10^6 with Q32.32
        err = _npow_err(fmt, sigma, n, t_fx, phw, bw)
        assert err <= 8, f"n={n}, sigma={sigma}, t_fx={t_fx:#x}: {mp.nstr(err, 4)} ulp"


def test_npow_error_vs_t() -> None:
    """Characterize error growth with t at Z128 (documents phase-guard adequacy)."""
    fmt = mf.Format(limbs=2)
    phw, bw = _phw(fmt), _phw(fmt) + 32
    rng = random.Random(7)
    print("\n| t decade | max err (ulp of |n^-s|) |")
    print("|---|---|")
    for decade in [1e0, 1e2, 1e4, 1e6, 1e8, 4e9]:
        worst = mp.mpf(0)
        for _ in range(max(10, N // 20)):
            n = rng.randrange(2, 100_000)
            mant = (1 << (fmt.width - 1)) | rng.getrandbits(fmt.width - 1)
            sigma = mf.MPF(rng.getrandbits(1), rng.randrange(-2, 4), mant)
            t = rng.random() * decade
            t_fx = min(int(t * (1 << npow.T_AF)), (1 << 64) - 1)
            worst = max(worst, _npow_err(fmt, sigma, n, t_fx, phw, bw))
        print(f"| ~1e{int(mp.log10(decade))} | {mp.nstr(worst, 3)} |")
        assert worst <= 8


def test_npow_n_equals_one() -> None:
    """1^(-s) = 1 exactly."""
    fmt = mf.Format(limbs=2)
    phw, bw = _phw(fmt), _phw(fmt) + 32
    lnn_fx, frac = lnn_entry(1, fmt, bw)
    sigma = mf.MPF(0, 2, 3 << (fmt.width - 2))  # 3.0
    re, im, _, _ = npow.npow_s(sigma, lnn_fx, frac, 123 << 32, fmt, phw, bw)
    assert re == mf.MPF(0, 1, 1 << (fmt.width - 1))  # 1.0
    assert im.is_zero


def test_npow_sigma_saturation() -> None:
    """Huge |sigma| saturates cleanly through the clamp path."""
    fmt = mf.Format(limbs=2)
    phw, bw = _phw(fmt), _phw(fmt) + 32
    lnn_fx, frac = lnn_entry(3, fmt, bw)
    big = mf.MPF(0, 40, 1 << (fmt.width - 1))  # sigma = 2^39
    re, _, ovf, unf = npow.npow_s(big, lnn_fx, frac, 1 << 32, fmt, phw, bw)
    assert re.is_zero and unf and not ovf
    re, _, ovf, unf = npow.npow_s(neg(big), lnn_fx, frac, 1 << 32, fmt, phw, bw)
    assert re.is_special and ovf


def neg(v: mf.MPF) -> mf.MPF:
    return npow.neg(v)
