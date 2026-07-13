"""Golden fft_fx vs the float FFT reference: fixed-point error budget."""

import cmath
import math
import random

import pytest

from zetafpga.golden import fft
from zetafpga.kernel.os_multieval import _fft as fft_float

FRAC = 54  # test scale: values |v| <= ~32 fit signed 64 bits with room


def _to_fx(v: complex) -> tuple[int, int]:
    return round(v.real * 2**FRAC), round(v.imag * 2**FRAC)


def _to_c(v: tuple[int, int]) -> complex:
    return complex(v[0] / 2**FRAC, v[1] / 2**FRAC)


@pytest.mark.parametrize("m", [64, 256, 1024])
def test_fft_vs_float(m: int) -> None:
    cfg = fft.load_cfg(m)
    rng = random.Random(f"fft{m}")
    # O-S-like load: sparse-ish bins, l1 <= ~32
    vec = [0j] * m
    for _ in range(m // 4):
        vec[rng.randrange(m)] += cmath.rect(rng.uniform(0, 0.25), rng.uniform(0, 2 * math.pi))
    got = fft.fft_fx([_to_fx(v) for v in vec], cfg)
    ref = fft_float(vec)
    # ~2 lsb truncation per component per stage (t feeds both outputs), plus
    # twiddle quantization and the float reference's own ~1e-16*l1 noise
    budget = math.log2(m) * 2.0**-FRAC * 8 + 64 * 2.0**-52
    worst = max(abs(_to_c(g) - r) for g, r in zip(got, ref, strict=True))
    assert worst < budget, f"m={m}: worst {worst:.3e} vs budget {budget:.3e}"


def test_fft_linearity_and_impulse() -> None:
    cfg = fft.load_cfg(64)
    one = 1 << FRAC
    # impulse at 0 -> constant spectrum (k=0 twiddle is exact)
    vec = [(0, 0)] * 64
    vec[0] = (one, 0)
    got = fft.fft_fx(vec, cfg)
    for g in got:
        assert abs(_to_c(g) - 1.0) < 1e-14
    # DC input -> impulse spectrum at bin 0 of height m
    vec2 = [(one >> 6, 0)] * 64
    got2 = fft.fft_fx(vec2, cfg)
    assert abs(_to_c(got2[0]) - 1.0) < 1e-13
    for g in got2[1:]:
        assert abs(_to_c(g)) < 1e-13
