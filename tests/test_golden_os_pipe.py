"""M21 acceptance (golden level): the fixed-point O-S grid sum matches the
direct exact-phase tone sum to hunting accuracy."""

import cmath
import math

import mpmath as mp
import pytest

from zetafpga.golden import mpfloat as mf
from zetafpga.golden import os_pipe
from zetafpga.kernel.em_setup import T_AF
from zetafpga.kernel.os_multieval import _anchor_phase
from zetafpga.kernel.rs_setup import rs_entry, rs_n

FMT = mf.Format(limbs=1)


@pytest.fixture(autouse=True)
def _prec() -> None:
    mp.mp.prec = 200


def _direct(t_fx: int, n: int) -> complex:
    return complex(
        sum(
            k ** (-0.5) * cmath.exp(-2j * math.pi * _anchor_phase(t_fx, k)) for k in range(1, n + 1)
        )
    )


@pytest.mark.parametrize("m,count", [(128, 32), (256, 64)])
def test_os_grid_vs_direct(m: int, count: int) -> None:
    t0_fx = int(50_000.0 * (1 << T_AF))
    dt_fx = int(0.25 * (1 << T_AF))
    n = rs_n(t0_fx + (count - 1) * dt_fx)
    assert n == rs_n(t0_fx), "batch must sit inside one N segment"
    entries = [rs_entry(k, FMT, FMT.width + 64) for k in range(1, n + 1)]
    got = os_pipe.os_grid_sum(t0_fx, dt_fx, n, count, entries, FMT, m)
    scale = 2.0**os_pipe.FRAC
    for j in [0, 1, count // 2, count - 1]:
        gz = complex(got[j][0] / scale, got[j][1] / scale)
        ref = _direct(t0_fx + j * dt_fx, n)
        assert abs(gz - ref) < 2e-9, f"j={j}: {abs(gz - ref):.3e}"
