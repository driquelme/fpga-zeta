"""M19 acceptance: Odlyzko-Schonhage multi-evaluation.

- the binned-FFT main sum matches the direct tone sum to float accuracy;
- Z on O-S grids matches mp.siegelz within the RS remainder + float budget,
  including across an N-boundary segment split;
- the zero hunter reproduces the first LMFDB zeros.
"""

import cmath
import math

import mpmath as mp
import pytest

from zetafpga.apps.zeros import LMFDB_ZEROS
from zetafpga.kernel import os_multieval as osme
from zetafpga.kernel.em_setup import T_AF
from zetafpga.kernel.rs_setup import rs_n


@pytest.fixture(autouse=True)
def _prec() -> None:
    mp.mp.prec = 200


def _budget(t: float) -> float:
    return float(0.1 * t ** (-11 / 4) + 3e-9)  # K=4 remainder + float floor


def test_main_sum_vs_direct() -> None:
    """O-S segment sum == direct exact-phase tone sum, to float accuracy."""
    t0_fx = int(100_000.0 * (1 << T_AF))
    dt_fx = int(0.25 * (1 << T_AF))
    count = 256
    n = rs_n(t0_fx + (count - 1) * dt_fx)
    assert n == rs_n(t0_fx), "test range must not cross an N boundary"
    got = osme._main_sum_segment(t0_fx, dt_fx, 0, count, n)
    for j in [0, 1, count // 2, count - 1]:
        t_fx = t0_fx + j * dt_fx
        ref = sum(
            k ** (-0.5) * cmath.exp(-2j * math.pi * osme._anchor_phase(t_fx, k))
            for k in range(1, n + 1)
        )
        assert abs(got[j] - ref) < 1e-9, f"j={j}: {abs(got[j] - ref):.3e}"


def test_z_grid_vs_siegelz() -> None:
    t0_fx = int(10_000.0 * (1 << T_AF))
    dt_fx = int(0.5 * (1 << T_AF))
    grid = osme.z_grid_os(t0_fx, dt_fx, 64)
    for t, z in grid[::9]:
        ref = float(mp.siegelz(mp.mpf(t)))
        assert abs(z - ref) <= _budget(t), f"t={t}: {z} vs {ref}"


def test_z_grid_across_n_boundary() -> None:
    """A grid straddling t = 2pi*(N+1)^2 splits segments and stays accurate."""
    n_target = 13
    t_b = 2 * math.pi * (n_target + 1) ** 2  # N: 13 -> 14 here
    t0_fx = int((t_b - 4.0) * (1 << T_AF))
    dt_fx = int(0.5 * (1 << T_AF))
    grid = osme.z_grid_os(t0_fx, dt_fx, 16)
    ns = {rs_n(t0_fx + j * dt_fx) for j in range(16)}
    assert ns == {n_target, n_target + 1}
    for t, z in grid:
        ref = float(mp.siegelz(mp.mpf(t)))
        assert abs(z - ref) <= _budget(t), f"t={t}: {z} vs {ref}"


def test_hunt_zeros_vs_lmfdb() -> None:
    found = osme.hunt_zeros_os(10.0, 51.0)
    assert len(found) >= 10
    for got, ref in zip(found[:10], LMFDB_ZEROS[:10], strict=False):
        assert abs(got - ref) < 1e-5, f"{got} vs {ref}"
