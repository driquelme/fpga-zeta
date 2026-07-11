"""Package smoke tests (fast job: no simulator required)."""

import mpmath

import zetafpga


def test_version() -> None:
    assert zetafpga.__version__


def test_mpmath_oracle_available() -> None:
    # ζ(2) = π²/6 — sanity-check the oracle dependency.
    mpmath.mp.dps = 30
    assert mpmath.almosteq(mpmath.zeta(2), mpmath.pi**2 / 6)
