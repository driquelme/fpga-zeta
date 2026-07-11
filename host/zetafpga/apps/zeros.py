"""Zero locator: find zeros of zeta(1/2 + it) through the kernel path.

Coarse grid scan for local minima of |zeta| below a threshold, then
golden-section refinement of each bracket. Validated against the LMFDB
zero table in tests/test_apps.py.

CLI:  uv run python -m zetafpga.apps.zeros --count 10
"""

import argparse

from zetafpga.apps.criticalline import Runner, scan
from zetafpga.driver.golden_backend import GoldenBackend
from zetafpga.golden import mpfloat as mf

# First ten zeros (LMFDB / Odlyzko), for reference and tests.
LMFDB_ZEROS = [
    14.134725141734693,
    21.022039638771554,
    25.010857580145688,
    30.424876125859513,
    32.935061587739189,
    37.586178158825671,
    40.918719012147495,
    43.327073280914999,
    48.005150881167159,
    49.773832477672302,
]

GOLDEN_RATIO_C = 0.6180339887498949


def _mag(runner: Runner, fmt: mf.Format, t: float) -> float:
    return scan(runner, fmt, [t])[0].mag


def _refine(runner: Runner, fmt: mf.Format, a: float, b: float, tol: float) -> float:
    """Golden-section minimization of |zeta(1/2+it)| on [a, b]."""
    c = b - GOLDEN_RATIO_C * (b - a)
    d = a + GOLDEN_RATIO_C * (b - a)
    fc, fd = _mag(runner, fmt, c), _mag(runner, fmt, d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - GOLDEN_RATIO_C * (b - a)
            fc = _mag(runner, fmt, c)
        else:
            a, c, fc = c, d, fd
            d = a + GOLDEN_RATIO_C * (b - a)
            fd = _mag(runner, fmt, d)
    return (a + b) / 2


def locate_zeros(
    runner: Runner,
    fmt: mf.Format,
    count: int = 10,
    t_min: float = 10.0,
    t_max: float = 51.0,
    coarse: float = 0.05,
    tol: float = 1e-6,
    threshold: float = 0.5,
) -> list[float]:
    steps = int((t_max - t_min) / coarse)
    ts = [t_min + k * coarse for k in range(steps + 1)]
    pts = scan(runner, fmt, ts)
    zeros: list[float] = []
    for i in range(1, len(pts) - 1):
        if pts[i].mag < threshold and pts[i].mag <= pts[i - 1].mag and pts[i].mag < pts[i + 1].mag:
            zeros.append(_refine(runner, fmt, pts[i - 1].t, pts[i + 1].t, tol))
            if len(zeros) == count:
                break
    return zeros


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--limbs", type=int, default=1)
    args = ap.parse_args()
    fmt = mf.Format(limbs=args.limbs)
    backend = GoldenBackend(fmt)
    found = locate_zeros(backend.run, fmt, count=args.count)
    for i, (t, ref) in enumerate(zip(found, LMFDB_ZEROS, strict=False), 1):
        print(f"zero {i:2d}: t = {t:.9f}   (LMFDB {ref:.9f}, diff {abs(t - ref):.2e})")


if __name__ == "__main__":
    main()
