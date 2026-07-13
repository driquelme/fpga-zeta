"""Generate constant ROMs for theta_turns (on-chip Riemann-Siegel theta).

  theta(t)/2pi = (u/2)(ln u - 1) - 1/16 + (1/2pi) * sum_k c_k * t^(1-2k)
  u = t/(2pi),   c_k = (1 - 2^(1-2k)) * |B_2k| / (4k(2k-1))

(The c_k identity reproduces 1/48, 7/5760, 31/80640, ... — verified against
the classical expansion.) The unit works one limb wider than the target
format (W2 = width + 64) so the t*ln t magnitude does not eat the target's
mantissa; the asymptotic-series floor e^(-pi t) sets a validity threshold
t >= t_min(width), below which the host computes theta itself.

Outputs per target width:
  theta_wW_fx.mem  : one line, 1/2pi at scale 2^(FG2+32) (FG2 = W2+24)
  theta_wW_mpf.mem : K lines, c_k/2pi as packed MPF@W2 words
  theta_wW.json    : {width, w2, fg2, k, t_min}
"""

import argparse
import json
from pathlib import Path

import mpmath as mp

WIDTHS = [64, 128, 192, 256]


def _t_min(width: int) -> int:
    # e^(-pi t) below 2^-(width+16)
    return int(mp.ceil((width + 16) * mp.ln(2) / mp.pi))


def _k_terms(width: int, t_min: int) -> int:
    """Terms until c_k * t_min^(1-2k) drops below the asymptotic floor."""
    floor = mp.mpf(2) ** -(width + 24)
    k = 1
    while True:
        c = (1 - mp.mpf(2) ** (1 - 2 * k)) * abs(mp.bernoulli(2 * k)) / (4 * k * (2 * k - 1))
        if c * mp.mpf(t_min) ** (1 - 2 * k) < floor or k > 80:
            return k
        k += 1


def generate(width: int, out_dir: Path) -> None:
    w2 = width + 64
    fg2 = w2 + 24
    mp.mp.dps = int(fg2 * 0.302) + 40

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "host"))
    from zetafpga.golden import mpfloat as mf
    from zetafpga.kernel.em_setup import mpf_from_real

    fmt2 = mf.Format(limbs=w2 // 64)
    t_min = _t_min(width)
    k_terms = _k_terms(width, t_min)

    inv2pi = int(mp.nint(mp.mpf(2) ** (fg2 + 32) / (2 * mp.pi)))
    fxw = fg2 + 36  # storage width, multiple-of-4 padded
    fxw += (4 - fxw % 4) % 4
    (out_dir / f"theta_w{width}_fx.mem").write_text(f"{inv2pi:0{fxw // 4}x}\n")

    lines = []
    for k in range(1, k_terms + 1):
        c = (1 - mp.mpf(2) ** (1 - 2 * k)) * abs(mp.bernoulli(2 * k)) / (4 * k * (2 * k - 1))
        word = mf.pack(mpf_from_real(c / (2 * mp.pi), fmt2), fmt2)
        lines.append(f"{word:0{(fmt2.mpw + 3) // 4}x}")
    (out_dir / f"theta_w{width}_mpf.mem").write_text("\n".join(lines) + "\n")

    meta = {"width": width, "w2": w2, "fg2": fg2, "k": k_terms, "t_min": t_min}
    (out_dir / f"theta_w{width}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote theta_w{width}: w2={w2}, fg2={fg2}, k={k_terms}, t_min={t_min}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("rtl/common/fn/tables"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for width in WIDTHS:
        generate(width, args.out_dir)


if __name__ == "__main__":
    main()
