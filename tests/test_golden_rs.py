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


def _budget(t: float) -> float:
    return float(0.2 * t**-0.75)  # C0-only remainder bound with margin


@pytest.fixture(autouse=True)
def _prec() -> None:
    mp.mp.prec = 160


def test_z_vs_siegelz() -> None:
    backend = GoldenBackend(FMT)
    ts = [15.0, 20.0, 50.0, 100.0, 500.0, 1000.0, 10_000.0, 100_000.0]
    for t, z in z_scan(backend.run, FMT, ts):
        ref = float(mp.siegelz(t))
        assert abs(z - ref) <= _budget(t), (
            f"t={t}: Z={z:.9f} vs siegelz={ref:.9f} (budget {_budget(t):.2e})"
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
        tol = _budget(ref) / 0.5  # |Z'| >= ~0.5 at the first zeros
        assert abs(got - ref) < tol, f"zero {i}: {got} vs {ref} (tol {tol:.2e})"
