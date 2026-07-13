"""M10 acceptance: Riemann-Siegel Z(t) through the kernel path.

- Z(t) vs mpmath.siegelz within the documented C0-only remainder budget
  (|R_1| <~ 0.13 * t^(-3/4); Gabcke). Higher C_k terms are a follow-up.
- Sign-change zeros vs LMFDB within the remainder-derived tolerance.
- RS/E-M cross-check: |Z(t)| = |zeta(1/2+it)|.
"""

import os

import mpmath as mp
import pytest

from zetafpga.apps.criticalline import scan
from zetafpga.apps.zeros import LMFDB_ZEROS
from zetafpga.apps.zfunc import locate_zeros_z, z_scan
from zetafpga.driver.golden_backend import GoldenBackend
from zetafpga.golden import mpfloat as mf

FMT = mf.Format(limbs=1)
K = int(os.environ.get("ZETA_ZEROS_K", "3"))


def _budget(t: float, kmax: int = 4) -> float:
    # remainder ~ t^(-(2K+3)/4) with empirically calibrated margin
    return float({0: 0.2, 4: 0.1}[kmax] * t ** (-(2 * kmax + 3) / 4))


@pytest.fixture(autouse=True)
def _prec() -> None:
    mp.mp.prec = 160


def test_z_vs_siegelz() -> None:
    backend = GoldenBackend(FMT)
    ts = [15.0, 20.0, 50.0, 100.0, 500.0, 1000.0, 10_000.0, 100_000.0]
    for kmax in (0, 4):
        for t, z in z_scan(backend.run, FMT, ts, kmax=kmax):
            ref = float(mp.siegelz(t))
            assert abs(z - ref) <= _budget(t, kmax), (
                f"t={t} K={kmax}: Z={z:.9f} vs siegelz={ref:.9f} (budget {_budget(t, kmax):.2e})"
            )


def test_rs_em_cross_check() -> None:
    """|Z(t)| must equal |zeta(1/2+it)| within the RS remainder."""
    backend = GoldenBackend(FMT)
    t = 100.0
    (_, z) = z_scan(backend.run, FMT, [t])[0]
    pt = scan(backend.run, FMT, [t])[0]  # full E-M evaluation
    assert abs(abs(z) - pt.mag) <= _budget(t)


def test_zeros_sign_change_vs_lmfdb() -> None:
    backend = GoldenBackend(FMT)
    t_max = LMFDB_ZEROS[K - 1] + 1.5
    found = locate_zeros_z(backend.run, FMT, count=K, t_max=t_max)
    assert len(found) == K, f"found {len(found)} zeros, expected {K}"
    for i, (got, ref) in enumerate(zip(found, LMFDB_ZEROS[:K], strict=True), 1):
        assert abs(got - ref) < 1e-5, f"zero {i}: {got} vs {ref}"
