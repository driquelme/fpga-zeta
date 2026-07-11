"""Critical-line scan: |zeta(1/2 + it)| over a t range, through the kernel path.

Library API + CLI. The runner is any Backend-style program executor
(GoldenBackend.run today; the RTL-backed runners plug in behind the same
Program/ReadbackResult contract).

CLI:  uv run python -m zetafpga.apps.criticalline --t0 0 --t1 50 --steps 500
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import mpmath as mp

from zetafpga.driver.golden_backend import GoldenBackend
from zetafpga.golden import mpfloat as mf
from zetafpga.kernel import isa
from zetafpga.kernel.em_setup import build_program, mpf_from_real, mpf_value
from zetafpga.kernel.program import Program
from zetafpga.kernel.tables import lnn_entry

Runner = Callable[[Program], isa.ReadbackResult]
CHUNK = 96  # evals per program (engine result buffer is 128)


@dataclass(frozen=True)
class ScanPoint:
    t: float
    re: float
    im: float
    mag: float


def scan(
    runner: Runner,
    fmt: mf.Format,
    t_values: list[float],
    sigma_value: float = 0.5,
) -> list[ScanPoint]:
    mp.mp.prec = 2 * fmt.width + 80
    sigma = mpf_from_real(mp.mpf(sigma_value), fmt)
    points: list[ScanPoint] = []
    for lo in range(0, len(t_values), CHUNK):
        ts = t_values[lo : lo + CHUNK]
        progs = [build_program(sigma, int(mp.nint(mp.mpf(t) * (1 << 32))), fmt) for t in ts]
        max_n = max(p.n for p in progs)
        entries = [lnn_entry(n, fmt, fmt.width + 64) for n in range(1, max_n + 2)]
        prg = Program(fmt)
        prg.write_lnn_table(entries).write_bern_table(list(progs[0].bern))
        for p in progs:
            prg.compute_em(p)
        prg.readback()
        rb = runner(prg)
        assert not rb.err and len(rb.results) == len(ts)
        for t, r in zip(ts, rb.results, strict=True):
            re = float(mpf_value(mf.unpack(r.re_word, fmt), fmt))
            im = float(mpf_value(mf.unpack(r.im_word, fmt), fmt))
            points.append(ScanPoint(t, re, im, (re * re + im * im) ** 0.5))
    return points


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=50.0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--limbs", type=int, default=1)
    args = ap.parse_args()

    fmt = mf.Format(limbs=args.limbs)
    backend = GoldenBackend(fmt)
    ts = [args.t0 + k * (args.t1 - args.t0) / args.steps for k in range(args.steps + 1)]
    for p in scan(backend.run, fmt, ts):
        bar = "#" * min(int(p.mag * 12), 70)
        print(f"t={p.t:9.4f}  |zeta|={p.mag:10.6f}  {bar}")


if __name__ == "__main__":
    main()
