"""M9 acceptance: the applications, end to end through the kernel path.

ZETA_ZEROS_K scales the zero hunt (default 3 zeros for the fast tier; the
acceptance run uses 10).
"""

import os

from zetafpga.apps.criticalline import scan
from zetafpga.apps.zeros import LMFDB_ZEROS, locate_zeros
from zetafpga.driver.golden_backend import GoldenBackend
from zetafpga.golden import mpfloat as mf

FMT = mf.Format(limbs=1)
K = int(os.environ.get("ZETA_ZEROS_K", "3"))


def test_criticalline_scan() -> None:
    backend = GoldenBackend(FMT)
    pts = scan(backend.run, FMT, [0.0, 14.134725141734693, 20.0])
    # zeta(1/2) = -1.4603545088...
    assert abs(pts[0].re + 1.4603545088095868) < 1e-9 and abs(pts[0].im) < 1e-12
    # near the first zero |zeta| is tiny
    assert pts[1].mag < 1e-7
    assert pts[2].mag > 1.0  # |zeta(1/2+20i)| ~ 1.15


def test_locate_zeros_vs_lmfdb() -> None:
    backend = GoldenBackend(FMT)
    t_max = LMFDB_ZEROS[K - 1] + 1.5
    found = locate_zeros(backend.run, FMT, count=K, t_max=t_max)
    assert len(found) == K, f"found {len(found)} zeros, expected {K}"
    for i, (got, ref) in enumerate(zip(found, LMFDB_ZEROS[:K], strict=True), 1):
        assert abs(got - ref) < 5e-6, f"zero {i}: {got} vs LMFDB {ref}"
