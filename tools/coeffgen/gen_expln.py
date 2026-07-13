"""Generate constant ROMs for exp_mpf and log_mpf, per mantissa width.

exp_mpf: range reduction y = k*ln2 + r (|r| <= ln2/2) then Taylor-Horner
e^r = sum r^k/k! with TERMS terms chosen so the truncated tail is below the
FG-bit working precision. Constants: ln2, 1/ln2, 1/k!.

log_mpf: multiplicative normalization ln(mu) = -sum ln(1+2^-i) over accepted i;
table of ln(1+2^-i) for i = 1..FG.

All constants are stored as CONSTW-bit two's complement at scale 2^FG, where
FG = width + 24 fractional working bits and CONSTW = FG + 8. Deterministic.

Usage: uv run python tools/coeffgen/gen_expln.py [--out-dir rtl/common/fn/tables]
"""

import argparse
import json
import math
from pathlib import Path

import mpmath as mp

WIDTHS = [64, 128, 192, 256, 320]  # 192/320 serve theta's W+64 internal log


def _terms(fg: int) -> int:
    """Smallest T with |r|^T/T! below 2^-(FG+4) for |r| <= ln2/2 (~2^-1.529)."""
    t = 1
    while 1.529 * t + math.lgamma(t + 1) / math.log(2) < fg + 4:
        t += 1
    return t + 1  # store invfact[0..t], i.e. t+1 coefficients


def _enc(val: mp.mpf, fg: int, constw: int) -> str:
    c = int(mp.nint(val * mp.mpf(2) ** fg))
    assert -(1 << (constw - 1)) <= c < (1 << (constw - 1))
    return f"{c & ((1 << constw) - 1):0{constw // 4}x}"


def generate(width: int, out_dir: Path) -> None:
    fg = width + 24
    constw = fg + 8
    terms = _terms(fg)
    mp.mp.dps = int(fg * 0.302) + 30

    exp_lines = [_enc(mp.ln(2), fg, constw), _enc(1 / mp.ln(2), fg, constw)]
    exp_lines += [_enc(1 / mp.factorial(k), fg, constw) for k in range(terms)]
    (out_dir / f"expln_w{width}_exp.mem").write_text("\n".join(exp_lines) + "\n")

    ln_lines = [_enc(mp.ln(1 + mp.mpf(2) ** -i), fg, constw) for i in range(1, fg + 1)]
    (out_dir / f"expln_w{width}_ln.mem").write_text("\n".join(ln_lines) + "\n")

    meta = {"width": width, "fg": fg, "constw": constw, "terms": terms}
    (out_dir / f"expln_w{width}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote expln_w{width}: fg={fg}, constw={constw}, terms={terms}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("rtl/common/fn/tables"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for width in WIDTHS:
        generate(width, args.out_dir)


if __name__ == "__main__":
    main()
