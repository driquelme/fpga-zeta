"""Generate constant ROMs for cexp_turns (full-precision e^(2*pi*i*phi)).

Decomposition: phi = phi_hi + phi_lo with phi_hi the top SEGW bits.
e^(2*pi*i*phi) = table[phi_hi] * e^(2*pi*i*phi_lo), where the table holds
cos/sin(2*pi*j/2^SEGW) at full working precision and the residual factor is a
short complex Taylor series in z = 2*pi*phi_lo (|z| < 2*pi*2^-SEGW).

ROM layout (one file per width): line 2j = cos, line 2j+1 = sin, plus a final
line holding 2*pi. All at scale 2^FG, CONSTW-bit two's complement (same
FG/CONSTW as the expln tables; the Taylor coefficients 1/k! are reused from
expln_w*_exp.mem). Deterministic.

Usage: uv run python tools/coeffgen/gen_cexp.py
"""

import argparse
import json
import math
from pathlib import Path

import mpmath as mp

WIDTHS = [64, 128, 256]
SEGW = 10


def _terms(fg: int) -> int:
    """Smallest T with |z|^T/T! below 2^-(FG+4) for |z| <= 2*pi*2^-SEGW."""
    logz = math.log2(2 * math.pi) - SEGW  # ~ -7.35 for SEGW=10
    t = 1
    while -logz * t + math.lgamma(t + 1) / math.log(2) < fg + 4:
        t += 1
    return t + 1


def _enc(val: mp.mpf, fg: int, constw: int) -> str:
    c = int(mp.nint(val * mp.mpf(2) ** fg))
    assert -(1 << (constw - 1)) <= c < (1 << (constw - 1))
    return f"{c & ((1 << constw) - 1):0{constw // 4}x}"


def generate(width: int, out_dir: Path) -> None:
    fg = width + 24
    constw = fg + 8
    terms = _terms(fg)
    mp.mp.dps = int(fg * 0.302) + 30

    lines = []
    for j in range(1 << SEGW):
        t = mp.mpf(j) / (1 << SEGW)
        lines.append(_enc(mp.cospi(2 * t), fg, constw))
        lines.append(_enc(mp.sinpi(2 * t), fg, constw))
    lines.append(_enc(2 * mp.pi, fg - 3, constw))  # 2*pi at scale 2^(FG-3)
    (out_dir / f"cexp_w{width}.mem").write_text("\n".join(lines) + "\n")

    meta = {"width": width, "fg": fg, "constw": constw, "segw": SEGW, "terms": terms}
    (out_dir / f"cexp_w{width}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote cexp_w{width}: fg={fg}, segw={SEGW}, terms={terms}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("rtl/common/fn/tables"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for width in WIDTHS:
        generate(width, args.out_dir)


if __name__ == "__main__":
    main()
