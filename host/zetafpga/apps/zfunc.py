"""Hardy Z(t) through the kernel path, and sign-change zero location (M10).

Z(t) is real with |Z(t)| = |zeta(1/2+it)|; its sign changes bracket the
zeros, so bisection replaces the slower |zeta| golden-section of apps/zeros.py.
The engine computes the O(sqrt(t)) Riemann-Siegel main sum (COMPUTE_PS);
theta(t) and the C0 correction are host scalars (kernel/rs_setup.py).

CLI:  uv run python -m zetafpga.apps.zfunc --count 10
"""

import argparse
from itertools import pairwise

import mpmath as mp

from zetafpga.apps.criticalline import CHUNK, Runner
from zetafpga.apps.zeros import LMFDB_ZEROS
from zetafpga.driver.golden_backend import GoldenBackend
from zetafpga.golden import mpfloat as mf
from zetafpga.kernel.program import Program
from zetafpga.kernel.rs_setup import rs_program, z_from_powersum
from zetafpga.kernel.tables import lnn_entry


def z_scan(runner: Runner, fmt: mf.Format, t_values: list[float]) -> list[tuple[float, float]]:
    """Z(t) at each t (t > 2*pi), batched through COMPUTE_PS programs."""
    out: list[tuple[float, float]] = []
    for lo in range(0, len(t_values), CHUNK):
        ts = t_values[lo : lo + CHUNK]
        t_fxs = [int(mp.nint(mp.mpf(t) * (1 << 32))) for t in ts]
        progs = [rs_program(t_fx, fmt) for t_fx in t_fxs]
        max_n = max(p.n for p in progs)
        entries = [lnn_entry(n, fmt, fmt.width + 64) for n in range(1, max_n + 1)]
        prg = Program(fmt)
        prg.write_lnn_table(entries)
        for p in progs:
            prg.compute_em(p)  # packs as COMPUTE_PS (ps_only)
        prg.readback()
        rb = runner(prg)
        assert not rb.err and len(rb.results) == len(ts)
        for t, t_fx, r in zip(ts, t_fxs, rb.results, strict=True):
            z = z_from_powersum(t_fx, mf.unpack(r.re_word, fmt), mf.unpack(r.im_word, fmt), fmt)
            out.append((t, z))
    return out


def locate_zeros_z(
    runner: Runner,
    fmt: mf.Format,
    count: int = 10,
    t_min: float = 12.0,
    t_max: float = 51.0,
    coarse: float = 0.4,
    tol: float = 1e-8,
) -> list[float]:
    """Zeros of Z(t) by sign-change bracketing + bisection."""
    steps = int((t_max - t_min) / coarse)
    grid = [t_min + k * coarse for k in range(steps + 1)]
    zs = z_scan(runner, fmt, grid)
    zeros: list[float] = []
    for (t1, z1), (t2, z2) in pairwise(zs):
        if z1 == 0.0:
            zeros.append(t1)
        elif z1 * z2 < 0:
            a, fa, b = t1, z1, t2
            while b - a > tol:
                m = (a + b) / 2
                fm = z_scan(runner, fmt, [m])[0][1]
                if fm == 0.0:
                    a = b = m
                elif fa * fm < 0:
                    b = m
                else:
                    a, fa = m, fm
            zeros.append((a + b) / 2)
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
    found = locate_zeros_z(backend.run, fmt, count=args.count)
    for i, t in enumerate(found, 1):
        ref = LMFDB_ZEROS[i - 1] if i <= len(LMFDB_ZEROS) else float("nan")
        print(f"zero {i:2d}: t = {t:.9f}   (LMFDB {ref:.9f}, diff {abs(t - ref):.2e})")


if __name__ == "__main__":
    main()
