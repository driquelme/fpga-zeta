"""Golden mpf_recip vs the exact reciprocal: <= 2 ulp, exact powers of two."""

import random
from fractions import Fraction

import pytest

from zetafpga.golden import mpfloat as mf
from zetafpga.golden import recip

FMTS = [mf.Format(limbs=n) for n in (1, 2, 3, 4)]


def _value(v: mf.MPF, fmt: mf.Format) -> Fraction:
    assert not (v.is_zero or v.is_special)
    return (-1 if v.sign else 1) * Fraction(v.mant, 1 << fmt.width) * Fraction(2) ** v.exp


def _ulps(got: mf.MPF, exact: Fraction, fmt: mf.Format) -> float:
    err = abs(_value(got, fmt) - exact)
    ulp = Fraction(2) ** (got.exp - fmt.width)
    return float(err / ulp)


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_recip_random(fmt: mf.Format) -> None:
    rng = random.Random(f"recip{fmt.limbs}")
    for _ in range(2000):
        mant = (1 << (fmt.width - 1)) | rng.getrandbits(fmt.width - 1)
        x = mf.MPF(rng.getrandbits(1), rng.randrange(-40, 40), mant)
        y, ovf, unf = recip.mpf_recip(x, fmt)
        assert not ovf and not unf
        assert _ulps(y, 1 / _value(x, fmt), fmt) <= 2.0, f"x={x}"


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_recip_directed(fmt: mf.Format) -> None:
    w = fmt.width
    for exp in (-5, 0, 1, 5, 20):
        # powers of two are exact
        x = mf.MPF(0, exp, 1 << (w - 1))
        y, _, _ = recip.mpf_recip(x, fmt)
        assert _value(y, fmt) == 1 / _value(x, fmt)
    # mantissa extremes
    for mant in ((1 << (w - 1)) + 1, (1 << w) - 1):
        y, _, _ = recip.mpf_recip(mf.MPF(0, 3, mant), fmt)
        assert _ulps(y, 1 / _value(mf.MPF(0, 3, mant), fmt), fmt) <= 2.0
    # 1/0 -> special + ovf; 1/special -> special
    y, ovf, _ = recip.mpf_recip(mf.zero(0), fmt)
    assert y.is_special and ovf
    y, _, _ = recip.mpf_recip(mf.special(1), fmt)
    assert y.is_special and y.sign == 1
