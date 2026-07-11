"""Bit-true golden model of the elementary-function units (M4: sincos_turns).

Evaluates the same committed coefficient tables as the RTL, with the same
fixed-point truncation order, so RTL comparison is bit-exact. Mathematical
accuracy of the tables themselves is validated in tests/test_golden_fn.py
against mpmath.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

TABLES_DIR = Path(__file__).resolve().parents[3] / "rtl" / "common" / "fn" / "tables"


@dataclass(frozen=True)
class SinCosCfg:
    segw: int
    deg: int
    zw: int
    ow: int
    cf: int
    cw: int

    @property
    def stem(self) -> str:
        return f"sincos_s{self.segw}d{self.deg}z{self.zw}o{self.ow}"


def load_cfg(stem: str = "sincos_s10d5z64o64") -> SinCosCfg:
    meta = json.loads((TABLES_DIR / f"{stem}.json").read_text())
    return SinCosCfg(**meta)


@lru_cache(maxsize=8)
def _load_rom(path: str, cw: int) -> tuple[int, ...]:
    values = []
    for line in Path(path).read_text().splitlines():
        v = int(line, 16)
        if v >> (cw - 1):  # two's complement decode
            v -= 1 << cw
        values.append(v)
    return tuple(values)


def _rom(cfg: SinCosCfg, func: str) -> tuple[int, ...]:
    return _load_rom(str(TABLES_DIR / f"{cfg.stem}_{func}.mem"), cfg.cw)


def _horner(rom: tuple[int, ...], seg: int, z: int, cfg: SinCosCfg) -> int:
    base = seg * (cfg.deg + 1)
    acc = rom[base + cfg.deg]
    for k in range(cfg.deg - 1, -1, -1):
        acc = rom[base + k] + ((acc * z) >> cfg.zw)  # floor shift == RTL >>>
    return acc


def sincos_turns(phase: int, phw: int, cfg: SinCosCfg) -> tuple[int, int]:
    """sin/cos of phase (phw-bit turns) as signed Q2.(ow-2) integers."""
    assert 0 <= phase < (1 << phw)
    qw = phw - 2
    assert qw - cfg.segw >= cfg.zw, "PHW too small for table config"
    quadrant = phase >> qw
    u = phase & ((1 << qw) - 1)
    seg = u >> (qw - cfg.segw)
    z = (u >> (qw - cfg.segw - cfg.zw)) & ((1 << cfg.zw) - 1)

    shift = cfg.cf - (cfg.ow - 2)
    s0 = _horner(_rom(cfg, "sin"), seg, z, cfg) >> shift
    c0 = _horner(_rom(cfg, "cos"), seg, z, cfg) >> shift

    if quadrant == 0:
        return s0, c0
    if quadrant == 1:
        return c0, -s0
    if quadrant == 2:
        return -s0, -c0
    return -c0, s0
