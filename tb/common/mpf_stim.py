"""Shared stimulus generation for MPF operator testbenches."""

import random

from zetafpga.golden import mpfloat as mf


def random_mpf(fmt: mf.Format, rng: random.Random, exp_range: int = 300) -> mf.MPF:
    """Random operand: mostly normalized moderate exponents, sprinkled with
    zeros, specials, and near-saturation exponents."""
    r = rng.random()
    if r < 0.02:
        return mf.zero(rng.getrandbits(1))
    if r < 0.03:
        return mf.special(rng.getrandbits(1))
    mant = (1 << (fmt.width - 1)) | rng.getrandbits(fmt.width - 1)
    if r < 0.06:
        exp = rng.randrange(fmt.emax - 2, fmt.emax + 1)
    elif r < 0.09:
        exp = rng.randrange(fmt.emin, fmt.emin + 3)
    else:
        exp = rng.randrange(-exp_range, exp_range)
    return mf.MPF(rng.getrandbits(1), exp, mant)


def cancellation_pair(fmt: mf.Format, rng: random.Random) -> tuple[mf.MPF, mf.MPF]:
    """Opposite signs, nearby magnitudes: stresses the subtract/normalize path."""
    x = random_mpf(fmt, rng)
    while x.is_zero or x.is_special:
        x = random_mpf(fmt, rng)
    mant = x.mant ^ rng.getrandbits(rng.randrange(1, fmt.width))
    mant |= 1 << (fmt.width - 1)
    exp = min(max(x.exp + rng.randrange(-2, 3), fmt.emin), fmt.emax)
    y = mf.MPF(1 - x.sign, exp, mant)
    return x, y
