"""Bit-true golden model of fft_radix2: complex fixed-point radix-2 DIT (M20).

Kernel e^(+2 pi i jk/M) (the host O-S evaluator's convention). Values are
plain signed integers on whatever scale the caller chose; the butterfly is

  t = (w * b) >> (CW - 2)   (floor truncation, per component)
  a' = a + t,  b' = a - t   (exact)

so precision is one truncation per element per stage (~log2(M) * 1 lsb) plus
the 2^-(CW-2) twiddle quantization. There is NO per-stage growth to scale
away: every DIT intermediate is a DFT of a subset of the inputs, hence
bounded by the input l1 norm — the caller guarantees l1 < 2^(DW-1) (the O-S
bins have l1 <= ~2 sqrt(N)). Mirrors rtl/common/fn/fft_radix2.sv exactly.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

TABLES_DIR = Path(__file__).resolve().parents[3] / "rtl" / "common" / "fn" / "tables"


@dataclass(frozen=True)
class FftCfg:
    m: int
    cw: int

    @property
    def stem(self) -> str:
        return f"fft_m{self.m}"


def load_cfg(m: int) -> FftCfg:
    meta = json.loads((TABLES_DIR / f"fft_m{m}.json").read_text())
    return FftCfg(**meta)


@lru_cache(maxsize=8)
def _twiddles(stem: str, cw: int) -> tuple[tuple[int, int], ...]:
    out = []
    for line in (TABLES_DIR / f"{stem}.mem").read_text().splitlines():
        word = int(line, 16)
        re = word & ((1 << cw) - 1)
        im = (word >> cw) & ((1 << cw) - 1)
        if re >> (cw - 1):
            re -= 1 << cw
        if im >> (cw - 1):
            im -= 1 << cw
        out.append((re, im))
    return tuple(out)


def bitrev(i: int, bits: int) -> int:
    r = 0
    for _ in range(bits):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


def fft_fx(vec: list[tuple[int, int]], cfg: FftCfg) -> list[tuple[int, int]]:
    """DFT of `vec` (natural order in, natural order out), bit-true."""
    m = cfg.m
    assert len(vec) == m
    sh = cfg.cw - 2
    tw = _twiddles(cfg.stem, cfg.cw)
    bits = m.bit_length() - 1
    out = [vec[bitrev(i, bits)] for i in range(m)]
    span = 1
    while span < m:
        stride = m // (2 * span)
        for start in range(0, m, 2 * span):
            for k in range(span):
                wre, wim = tw[k * stride]
                bre, bim = out[start + k + span]
                tre = (bre * wre - bim * wim) >> sh
                tim = (bre * wim + bim * wre) >> sh
                are, aim = out[start + k]
                out[start + k] = (are + tre, aim + tim)
                out[start + k + span] = (are - tre, aim - tim)
        span *= 2
    return out
