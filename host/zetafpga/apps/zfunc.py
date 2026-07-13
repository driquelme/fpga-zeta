"""Hardy Z(t) through the kernel path, and sign-change zero location (M10).

Z(t) is real with |Z(t)| = |zeta(1/2+it)|; its sign changes bracket the
zeros, so bisection replaces the slower |zeta| golden-section of apps/zeros.py.
The engine computes the O(sqrt(t)) Riemann-Siegel main sum; since M15 the
theta(t)/correction epilogue also runs on chip (COMPUTE_Z) for t >= t_min —
1/t comes from the on-chip Newton reciprocal (M16), so the host supplies
only t and N. Below t_min, or for kmax != 4, the COMPUTE_RS + host-epilogue
path (kernel/rs_setup.py) is used instead.

CLI:  uv run python -m zetafpga.apps.zfunc --count 10
"""

import argparse
from itertools import pairwise

import mpmath as mp

from zetafpga.apps.criticalline import CHUNK, Runner
from zetafpga.apps.zeros import LMFDB_ZEROS
from zetafpga.driver.golden_backend import GoldenBackend
from zetafpga.golden import mpfloat as mf
from zetafpga.golden import theta
from zetafpga.kernel.em_setup import mpf_value
from zetafpga.kernel.program import Program
from zetafpga.kernel.rs_setup import rs_entry, rs_n, z_from_powersum


def z_scan(
    runner: Runner,
    fmt: mf.Format,
    t_values: list[float],
    kmax: int = 4,
    onchip: bool = True,
) -> list[tuple[float, float]]:
    """Z(t) at each t (t > 2*pi), batched through COMPUTE_Z/COMPUTE_RS programs."""
    tcfg = theta.load_cfg(fmt.width)
    out: list[tuple[float, float]] = []
    for lo in range(0, len(t_values), CHUNK):
        ts = t_values[lo : lo + CHUNK]
        t_fxs = [int(mp.nint(mp.mpf(t) * (1 << 32))) for t in ts]
        ns = [rs_n(t_fx) for t_fx in t_fxs]
        # on-chip Z only above the theta validity floor and at the ROM'd KMAX
        use_z = [onchip and kmax == 4 and t_fx >= (tcfg.t_min << 32) for t_fx in t_fxs]
        # +1 margin: COMPUTE_Z derives N on chip (may exceed rs_n at a boundary)
        entries = [rs_entry(k, fmt, fmt.width + 64) for k in range(1, max(ns) + 2)]
        prg = Program(fmt)
        prg.write_rs_table(entries)
        for t_fx, n, uz in zip(t_fxs, ns, use_z, strict=True):
            if uz:
                prg.compute_z(t_fx)  # fully on-chip Z(t): 1/t and N derived
            else:
                prg.compute_rs(t_fx, n)  # main sum on chip, epilogue on host
        prg.readback()
        rb = runner(prg)
        assert not rb.err and len(rb.results) == len(ts)
        for t, t_fx, uz, r in zip(ts, t_fxs, use_z, rb.results, strict=True):
            if uz:
                z = float(mpf_value(mf.unpack(r.re_word, fmt), fmt))
            else:
                z = z_from_powersum(
                    t_fx, mf.unpack(r.re_word, fmt), mf.unpack(r.im_word, fmt), fmt, kmax=kmax
                )
            out.append((t, z))
    return out


def z_grid(
    runner: Runner, fmt: mf.Format, t0: float, dt: float, count: int
) -> list[tuple[float, float]]:
    """Z(t) on a uniform grid via single COMPUTE_ZGRID descriptors — the
    Odlyzko-Schoenhage evaluation pattern: per-point host work is zero (the
    engine derives 1/t and N itself); the host only writes the shared RS
    table sized to the grid maximum. Requires t0 >= t_min."""
    tcfg = theta.load_cfg(fmt.width)
    t0_fx = int(mp.nint(mp.mpf(t0) * (1 << 32)))
    dt_fx = int(mp.nint(mp.mpf(dt) * (1 << 32)))
    assert t0_fx >= (tcfg.t_min << 32) and dt_fx > 0 and count >= 1
    max_n = rs_n(t0_fx + (count - 1) * dt_fx) + 1  # +1 on-chip-N margin
    out: list[tuple[float, float]] = []
    for lo in range(0, count, CHUNK):
        points = min(CHUNK, count - lo)
        start_fx = t0_fx + lo * dt_fx
        prg = Program(fmt)
        prg.write_rs_table([rs_entry(k, fmt, fmt.width + 64) for k in range(1, max_n + 1)])
        prg.compute_zgrid(start_fx, dt_fx, points)
        prg.readback()
        rb = runner(prg)
        assert not rb.err and len(rb.results) == points
        for j, r in enumerate(rb.results):
            t = float(mp.mpf(start_fx + j * dt_fx) / (1 << 32))
            out.append((t, float(mpf_value(mf.unpack(r.re_word, fmt), fmt))))
    return out


def locate_zeros_z(
    runner: Runner,
    fmt: mf.Format,
    count: int = 10,
    t_min: float = 12.0,
    t_max: float = 51.0,
    coarse: float = 0.4,
    tol: float = 1e-8,
    kmax: int = 4,
    onchip: bool = True,
) -> list[float]:
    """Zeros of Z(t) by sign-change bracketing + bisection. The coarse scan
    rides COMPUTE_ZGRID (one descriptor per chunk) where the on-chip path is
    valid; the below-floor prefix and the bisection midpoints go through
    z_scan."""
    steps = int((t_max - t_min) / coarse)
    grid = [t_min + k * coarse for k in range(steps + 1)]
    t_floor = float(theta.load_cfg(fmt.width).t_min)
    if onchip and kmax == 4 and grid[-1] >= t_floor:
        hi0 = next(i for i, t in enumerate(grid) if t >= t_floor)
        zs = z_scan(runner, fmt, grid[:hi0], kmax=kmax, onchip=onchip)
        zs += z_grid(runner, fmt, grid[hi0], coarse, len(grid) - hi0)
    else:
        zs = z_scan(runner, fmt, grid, kmax=kmax, onchip=onchip)
    zeros: list[float] = []
    for (t1, z1), (t2, z2) in pairwise(zs):
        if z1 == 0.0:
            zeros.append(t1)
        elif z1 * z2 < 0:
            a, fa, b = t1, z1, t2
            while b - a > tol:
                m = (a + b) / 2
                fm = z_scan(runner, fmt, [m], kmax=kmax, onchip=onchip)[0][1]
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
    ap.add_argument("--kmax", type=int, default=4, help="RS correction order (0..4)")
    args = ap.parse_args()
    fmt = mf.Format(limbs=args.limbs)
    backend = GoldenBackend(fmt)
    found = locate_zeros_z(backend.run, fmt, count=args.count, kmax=args.kmax)
    for i, t in enumerate(found, 1):
        ref = LMFDB_ZEROS[i - 1] if i <= len(LMFDB_ZEROS) else float("nan")
        print(f"zero {i:2d}: t = {t:.9f}   (LMFDB {ref:.9f}, diff {abs(t - ref):.2e})")


if __name__ == "__main__":
    main()
