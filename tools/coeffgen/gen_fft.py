"""Generate twiddle ROMs for fft_radix2 (M20).

One line per twiddle k < M/2: {im, re} packed, each CW bits signed at scale
2^(CW-2) (the sincos Q2.62 convention), kernel e^(+2 pi i k / M) — matching
the host O-S evaluator's FFT sign (kernel/os_multieval.py).

Outputs per size:  fft_mM.mem (M/2 lines), fft_mM.json {m, cw}.
"""

import argparse
import json
from pathlib import Path

import mpmath as mp

SIZES = [64, 256, 1024, 4096]
CW = 64


def generate(m: int, out_dir: Path) -> None:
    mp.mp.prec = 128
    scale = mp.mpf(2) ** (CW - 2)
    mask = (1 << CW) - 1
    lines = []
    for k in range(m // 2):
        ang = 2 * mp.pi * k / m
        wre = int(mp.nint(mp.cos(ang) * scale))
        wim = int(mp.nint(mp.sin(ang) * scale))
        word = (wre & mask) | ((wim & mask) << CW)
        lines.append(f"{word:0{2 * CW // 4}x}")
    (out_dir / f"fft_m{m}.mem").write_text("\n".join(lines) + "\n")
    (out_dir / f"fft_m{m}.json").write_text(json.dumps({"m": m, "cw": CW}, indent=2) + "\n")
    print(f"wrote fft_m{m}: cw={CW}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("rtl/common/fn/tables"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for m in SIZES:
        generate(m, args.out_dir)


if __name__ == "__main__":
    main()
