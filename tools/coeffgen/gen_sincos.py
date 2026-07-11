"""Generate piecewise-polynomial coefficient ROMs for sincos_turns.

For each of 2^SEGW segments of the quarter turn [0, 1/4), fits a degree-DEG
polynomial in the normalized in-segment offset z in [0, 1) to
sin(2*pi*t) / cos(2*pi*t), t = (seg + z) / (4 * 2^SEGW) turns, by Chebyshev-node
interpolation at high precision (near-minimax: error within a small factor of
the true minimax polynomial, with orders of magnitude of margin at these
segment widths). Deterministic: same parameters always produce the same tables.

Coefficients are emitted as two's-complement CW-bit hex, one per line,
segment-major (seg*(DEG+1) + k for coefficient of z^k), plus a JSON metadata
file consumed by the golden model and testbenches.

Usage: uv run python tools/coeffgen/gen_sincos.py [--out-dir rtl/common/fn/tables]
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mpmath as mp


@dataclass(frozen=True)
class SinCosCfg:
    """Parameters of a sincos_turns table set (mirrored in RTL parameters)."""

    segw: int = 10  # log2(segments per quarter turn)
    deg: int = 5  # polynomial degree
    zw: int = 64  # in-segment offset bits used by Horner
    ow: int = 64  # output width (signed Q2.(ow-2))
    cf: int = 72  # coefficient fractional bits
    cw: int = 80  # coefficient storage width (two's complement)

    @property
    def nseg(self) -> int:
        return 1 << self.segw

    @property
    def stem(self) -> str:
        return f"sincos_s{self.segw}d{self.deg}z{self.zw}o{self.ow}"


def _fit_segment(func: str, seg: int, cfg: SinCosCfg, nodes: list[mp.mpf]) -> list[int]:
    """Chebyshev-node interpolation of one segment; returns fixed-point coeffs."""
    scale = mp.mpf(1) / (4 * cfg.nseg)  # segment width in turns

    def f(z: mp.mpf) -> mp.mpf:
        t = (seg + z) * scale  # turns
        return mp.sinpi(2 * t) if func == "sin" else mp.cospi(2 * t)

    n = cfg.deg + 1
    rhs = mp.matrix([f(z) for z in nodes])
    van = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            van[i, j] = nodes[i] ** j
    coeffs = mp.lu_solve(van, rhs)

    out: list[int] = []
    for j in range(n):
        c = int(mp.nint(coeffs[j] * (1 << cfg.cf)))
        assert -(1 << (cfg.cw - 1)) <= c < (1 << (cfg.cw - 1)), (func, seg, j, c)
        out.append(c & ((1 << cfg.cw) - 1))  # two's complement encode
    return out


def generate(cfg: SinCosCfg, out_dir: Path) -> None:
    mp.mp.dps = 60
    n = cfg.deg + 1
    nodes = [(1 + mp.cos(mp.pi * (2 * i + 1) / (2 * n))) / 2 for i in range(n)]
    out_dir.mkdir(parents=True, exist_ok=True)
    for func in ("sin", "cos"):
        lines = []
        for seg in range(cfg.nseg):
            lines.extend(f"{c:0{cfg.cw // 4}x}" for c in _fit_segment(func, seg, cfg, nodes))
        path = out_dir / f"{cfg.stem}_{func}.mem"
        path.write_text("\n".join(lines) + "\n")
        print(f"wrote {path} ({cfg.nseg * n} coefficients)")
    meta = out_dir / f"{cfg.stem}.json"
    meta.write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    print(f"wrote {meta}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("rtl/common/fn/tables"))
    ap.add_argument("--segw", type=int, default=10)
    ap.add_argument("--deg", type=int, default=5)
    ap.add_argument("--zw", type=int, default=64)
    ap.add_argument("--ow", type=int, default=64)
    ap.add_argument("--cf", type=int, default=72)
    ap.add_argument("--cw", type=int, default=80)
    args = ap.parse_args()
    cfg = SinCosCfg(args.segw, args.deg, args.zw, args.ow, args.cf, args.cw)
    generate(cfg, args.out_dir)


if __name__ == "__main__":
    main()
