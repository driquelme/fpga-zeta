"""Golden MPF operators vs an exact rational-arithmetic RNE reference.

The golden model must be *correctly rounded*: its result must equal the exact
mathematical result rounded once to the target precision (round-to-nearest,
ties-to-even). Both operands are dyadic rationals, so sums and products are
exact Fractions and the reference rounding below is itself exact.
"""

import random
from fractions import Fraction

import pytest

from zetafpga.golden import mpfloat as mf

FMTS = [mf.Format(limbs=1), mf.Format(limbs=2), mf.Format(limbs=4)]


def _to_fraction(v: mf.MPF, fmt: mf.Format) -> Fraction:
    assert not v.is_special
    if v.is_zero:
        return Fraction(0)
    val = Fraction(v.mant) * Fraction(2) ** (v.exp - fmt.width)
    return -val if v.sign else val


def rne_reference(val: Fraction, fmt: mf.Format) -> tuple[mf.MPF, bool, bool]:
    """Exact RNE rounding of a rational to the MPF format, with saturation."""
    if val == 0:
        return mf.zero(0), False, False
    sign = 1 if val < 0 else 0
    a = abs(val)
    # Find e with 2^(e-1) <= a < 2^e, i.e. a = m * 2^(e-width), m in [2^(w-1), 2^w).
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if a >= Fraction(2) ** e:
        e += 1
    assert Fraction(2) ** (e - 1) <= a < Fraction(2) ** e
    scaled = a * Fraction(2) ** (fmt.width - e)
    m = scaled.numerator // scaled.denominator
    frac = scaled - m
    if frac > Fraction(1, 2) or (frac == Fraction(1, 2) and m % 2 == 1):
        m += 1
    if m >> fmt.width:
        m >>= 1
        e += 1
    if e > fmt.emax:
        return mf.special(sign), True, False
    if e < fmt.emin:
        return mf.zero(sign), False, True
    return mf.MPF(sign, e, m), False, False


def random_mpf(fmt: mf.Format, rng: random.Random, exp_range: int = 300) -> mf.MPF:
    r = rng.random()
    mant = (1 << (fmt.width - 1)) | rng.getrandbits(fmt.width - 1)
    if r < 0.05:
        exp = rng.randrange(fmt.emax - 2, fmt.emax + 1)
    elif r < 0.10:
        exp = rng.randrange(fmt.emin, fmt.emin + 3)
    else:
        exp = rng.randrange(-exp_range, exp_range)
    return mf.MPF(rng.getrandbits(1), exp, mant)


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_pack_unpack_roundtrip(fmt: mf.Format) -> None:
    rng = random.Random(fmt.limbs)
    for _ in range(500):
        v = random_mpf(fmt, rng)
        assert mf.unpack(mf.pack(v, fmt), fmt) == v
    assert mf.unpack(mf.pack(mf.zero(1), fmt), fmt) == mf.zero(1)
    assert mf.unpack(mf.pack(mf.special(0), fmt), fmt) == mf.special(0)


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_mul_correctly_rounded(fmt: mf.Format) -> None:
    rng = random.Random(100 + fmt.limbs)
    for _ in range(2000):
        x, y = random_mpf(fmt, rng), random_mpf(fmt, rng)
        got, ovf, unf = mf.mpf_mul(x, y, fmt)
        exact = _to_fraction(x, fmt) * _to_fraction(y, fmt)
        exp, eovf, eunf = rne_reference(exact, fmt)
        assert (got, ovf, unf) == (exp, eovf, eunf), f"x={x}, y={y}"


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_add_correctly_rounded(fmt: mf.Format) -> None:
    rng = random.Random(200 + fmt.limbs)
    for i in range(3000):
        x = random_mpf(fmt, rng)
        if i % 3 == 0:
            # Cancellation pressure: nearby magnitude, opposite sign.
            mant = x.mant ^ rng.getrandbits(rng.randrange(1, fmt.width))
            mant |= 1 << (fmt.width - 1)
            y = mf.MPF(1 - x.sign, x.exp + rng.randrange(-2, 3), mant)
        elif i % 7 == 0:
            # Far-apart exponents: sticky-only alignment path.
            y = mf.MPF(
                rng.getrandbits(1),
                x.exp - fmt.width - rng.randrange(0, 8),
                (1 << (fmt.width - 1)) | rng.getrandbits(fmt.width - 1),
            )
        else:
            y = random_mpf(fmt, rng)
        got, ovf, unf = mf.mpf_add(x, y, fmt)
        exact = _to_fraction(x, fmt) + _to_fraction(y, fmt)
        exp, eovf, eunf = rne_reference(exact, fmt)
        assert (got, ovf, unf) == (exp, eovf, eunf), f"x={x}, y={y}"


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_add_directed(fmt: mf.Format) -> None:
    w = fmt.width
    one = mf.MPF(0, 1, 1 << (w - 1))  # 1.0
    m_one = mf.MPF(1, 1, 1 << (w - 1))
    # exact cancellation -> +0
    assert mf.mpf_add(one, m_one, fmt)[0] == mf.zero(0)
    # zeros
    assert mf.mpf_add(mf.zero(0), one, fmt)[0] == one
    assert mf.mpf_add(one, mf.zero(1), fmt)[0] == one
    assert mf.mpf_add(mf.zero(1), mf.zero(1), fmt)[0] == mf.zero(1)
    assert mf.mpf_add(mf.zero(0), mf.zero(1), fmt)[0] == mf.zero(0)
    # specials propagate
    assert mf.mpf_add(mf.special(1), one, fmt)[0] == mf.special(1)
    assert mf.mpf_mul(mf.special(0), one, fmt)[0] == mf.special(0)
    # 1.0 + ulp/2 ties to even (stays 1.0)
    tie = mf.MPF(0, 1 - w, 1 << (w - 1))
    assert mf.mpf_add(one, tie, fmt)[0] == one
