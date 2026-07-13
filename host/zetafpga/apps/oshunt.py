"""Zero hunting via Odlyzko-Schonhage multi-evaluation (M19).

One binned-FFT pass evaluates Z on the whole grid (O(N + M log M) per
segment instead of O(N * J)); sign changes are refined by bisection on
direct single-point evaluations.

CLI:  uv run python -m zetafpga.apps.oshunt --t0 10 --t1 51
"""

import argparse
import time

from zetafpga.apps.zeros import LMFDB_ZEROS
from zetafpga.kernel.os_multieval import hunt_zeros_os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t0", type=float, default=10.0)
    ap.add_argument("--t1", type=float, default=51.0)
    ap.add_argument("--dt", type=float, default=0.25)
    args = ap.parse_args()

    start = time.perf_counter()
    zeros = hunt_zeros_os(args.t0, args.t1, dt=args.dt)
    elapsed = time.perf_counter() - start

    for i, t in enumerate(zeros, 1):
        ref = next((z for z in LMFDB_ZEROS if abs(z - t) < 0.1), None)
        tag = f"   (LMFDB {ref:.9f}, diff {abs(t - ref):.2e})" if ref is not None else ""
        print(f"zero {i:3d}: t = {t:.9f}{tag}")
    print(f"{len(zeros)} zeros in [{args.t0}, {args.t1}] — {elapsed:.2f}s")


if __name__ == "__main__":
    main()
