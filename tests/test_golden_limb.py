"""Directed + random unit tests for the limb golden model (fast job, no simulator)."""

import random

import pytest

from zetafpga.golden import limb

WIDTHS = [16, 64, 128, 256]


@pytest.mark.parametrize("width", WIDTHS)
def test_lzc_directed(width: int) -> None:
    assert limb.lzc(0, width) == width
    assert limb.lzc(limb.mask(width), width) == 0
    for i in range(width):
        assert limb.lzc(1 << i, width) == width - 1 - i


@pytest.mark.parametrize("width", WIDTHS)
def test_addsub_directed(width: int) -> None:
    m = limb.mask(width)
    # carry chain across every limb boundary
    assert limb.addsub(m, 0, width, cin=True) == (0, True)
    assert limb.addsub(m, m, width) == (m - 1, True)
    assert limb.addsub(0, 0, width) == (0, False)
    # borrow
    assert limb.addsub(0, 1, width, sub=True) == (m, True)
    assert limb.addsub(5, 5, width, sub=True, cin=True) == (m, True)
    assert limb.addsub(m, m, width, sub=True) == (0, False)


@pytest.mark.parametrize("width", WIDTHS)
def test_shift_directed(width: int) -> None:
    m = limb.mask(width)
    assert limb.shift(m, 0, width) == (m, False)
    assert limb.shift(m, width, width, left=True) == (0, True)
    assert limb.shift(1, 1, width) == (0, True)
    assert limb.shift(1 << (width - 1), 1, width, left=True) == (0, True)
    assert limb.shift(1, width - 1, width, left=True) == (1 << (width - 1), False)


@pytest.mark.parametrize("width", WIDTHS)
def test_random_consistency(width: int) -> None:
    rng = random.Random(width)
    m = limb.mask(width)
    for _ in range(2000):
        a = rng.getrandbits(width)
        b = rng.getrandbits(width)
        s, c = limb.addsub(a, b, width, cin=True)
        assert (a + b + 1) & m == s and ((a + b + 1) > m) == c
        d, br = limb.addsub(a, b, width, sub=True)
        assert (a - b) & m == d and (a < b) == br
        amt = rng.randrange(width + 1)
        out, lost = limb.shift(a, amt, width, left=True)
        assert out == (a << amt) & m
        assert lost == (((a << amt) & ~m) != 0)
