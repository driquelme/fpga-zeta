"""Golden exp_mpf/log_mpf vs mpmath: <= 2 ulp over the documented ranges."""

import os
import random

import mpmath as mp
import pytest

from zetafpga.golden import expln
from zetafpga.golden import mpfloat as mf

FMTS = [mf.Format(limbs=1), mf.Format(limbs=2), mf.Format(limbs=4)]
N = int(os.environ.get("ZETA_TEST_N", "2000"))
ULP = 2


def _ulp_err(got: mf.MPF, ref: mp.mpf, fmt: mf.Format) -> mp.mpf:
    assert not got.is_special, "unexpected special"
    val = mp.mpf(got.mant) * mp.mpf(2) ** (got.exp - fmt.width)
    if got.sign:
        val = -val
    if ref == 0:
        return mp.mpf(abs(val))
    # ulp of the reference at the format's precision
    ulp = mp.mpf(2) ** (int(mp.floor(mp.log(abs(ref), 2))) + 1 - fmt.width)
    return abs(val - ref) / ulp


def _rand_mpf(fmt: mf.Format, rng: random.Random, emin: int, emax: int) -> mf.MPF:
    mant = (1 << (fmt.width - 1)) | rng.getrandbits(fmt.width - 1)
    return mf.MPF(rng.getrandbits(1), rng.randrange(emin, emax + 1), mant)


@pytest.fixture(autouse=True)
def _prec() -> None:
    mp.mp.prec = 700


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_exp_accuracy(fmt: mf.Format) -> None:
    cfg = expln.load_cfg(fmt.width)
    rng = random.Random(fmt.width)
    for i in range(N):
        # exponent range: tiny arguments through near-saturation magnitudes
        e_hi = 19 if i % 4 == 0 else 8
        y = _rand_mpf(fmt, rng, -fmt.width - 30, e_hi)
        got, ovf, unf = expln.exp_mpf(y, fmt, cfg)
        yv = mp.mpf(y.mant) * mp.mpf(2) ** (y.exp - fmt.width)
        if y.sign:
            yv = -yv
        ref = mp.exp(yv)
        if ovf or unf or got.is_special or got.is_zero:
            # saturation must be consistent with the true magnitude
            ref_e = yv / mp.ln(2)
            assert abs(ref_e) > fmt.emax - 2, f"spurious saturation for y={y}"
            continue
        err = _ulp_err(got, ref, fmt)
        assert err <= ULP, f"y={y}: {mp.nstr(err, 5)} ulp"


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_exp_directed(fmt: mf.Format) -> None:
    cfg = expln.load_cfg(fmt.width)
    one = mf.MPF(0, 1, 1 << (fmt.width - 1))
    got, _, _ = expln.exp_mpf(mf.zero(0), fmt, cfg)
    assert got == one  # e^0 = 1 exactly
    got, ovf, _ = expln.exp_mpf(mf.MPF(0, 21, 1 << (fmt.width - 1)), fmt, cfg)
    assert got.is_special and ovf  # huge positive -> overflow
    got, _, unf = expln.exp_mpf(mf.MPF(1, 22, 1 << (fmt.width - 1)), fmt, cfg)
    assert got.is_zero and unf  # huge negative -> underflow to zero


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_log_accuracy(fmt: mf.Format) -> None:
    cfg = expln.load_cfg(fmt.width)
    rng = random.Random(1000 + fmt.width)
    for _ in range(N):
        x = _rand_mpf(fmt, rng, -(1 << 18), 1 << 18)
        x = mf.MPF(0, x.exp, x.mant)  # positive domain
        if x.exp in (0, 1):  # skip the documented near-1 band conservatively
            continue
        got, ovf, unf = expln.log_mpf(x, fmt, cfg)
        assert not (ovf or unf)
        xv = mp.mpf(x.mant) * mp.mpf(2) ** (x.exp - fmt.width)
        err = _ulp_err(got, mp.ln(xv), fmt)
        assert err <= ULP, f"x={x}: {mp.nstr(err, 5)} ulp"


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_log_zeta_arguments(fmt: mf.Format) -> None:
    """ln n for integer n -- the actual zeta power-sum arguments."""
    cfg = expln.load_cfg(fmt.width)
    for n in [2, 3, 5, 7, 10, 100, 1000, 4096, 10**6]:
        e = n.bit_length()
        mant = n << (fmt.width - e)
        got, _, _ = expln.log_mpf(mf.MPF(0, e, mant), fmt, cfg)
        err = _ulp_err(got, mp.ln(n), fmt)
        assert err <= ULP, f"ln({n}): {mp.nstr(err, 5)} ulp"


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_log_domain_errors(fmt: mf.Format) -> None:
    cfg = expln.load_cfg(fmt.width)
    assert expln.log_mpf(mf.zero(0), fmt, cfg)[0].is_special
    assert expln.log_mpf(mf.MPF(1, 1, 1 << (fmt.width - 1)), fmt, cfg)[0].is_special
    assert expln.log_mpf(mf.special(0), fmt, cfg)[0].is_special
    # ln(1) = 0 exactly
    one = mf.MPF(0, 1, 1 << (fmt.width - 1))
    assert expln.log_mpf(one, fmt, cfg)[0] == mf.zero(0)
