"""Bit-true golden model of cexp_turns: full-precision e^(2*pi*i*phi) (M6).

Mirrors rtl/common/fn/cexp_turns.sv: table factor for the top SEGW phase bits
times a complex Taylor-Horner residual, all at FG fractional working bits with
floor truncations, then two MPF normalizations.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from zetafpga.golden import expln
from zetafpga.golden import mpfloat as mf

TABLES_DIR = Path(__file__).resolve().parents[3] / "rtl" / "common" / "fn" / "tables"


@dataclass(frozen=True)
class CexpCfg:
    width: int
    fg: int
    constw: int
    segw: int
    terms: int

    @property
    def stem(self) -> str:
        return f"cexp_w{self.width}"


def load_cfg(width: int) -> CexpCfg:
    meta = json.loads((TABLES_DIR / f"cexp_w{width}.json").read_text())
    return CexpCfg(**meta)


@lru_cache(maxsize=8)
def _load(path: str, constw: int) -> tuple[int, ...]:
    values = []
    for line in Path(path).read_text().splitlines():
        v = int(line, 16)
        if v >> (constw - 1):
            v -= 1 << constw
        values.append(v)
    return tuple(values)


def _table(cfg: CexpCfg) -> tuple[int, ...]:
    return _load(str(TABLES_DIR / f"{cfg.stem}.mem"), cfg.constw)


def _fix_to_mpf(a: int, fmt: mf.Format, fg: int) -> mf.MPF:
    """Normalize a signed FG-scale fixed-point value into an MPF (RNE-ish)."""
    if a == 0:
        return mf.zero(0)
    sign = 1 if a < 0 else 0
    mag = -a if sign else a
    p = mag.bit_length() - 1
    e = p - fg + 1
    w = fmt.width
    t = (mag >> (p - w)) if p >= w else (mag << (w - p))
    mant = (t + 1) >> 1
    if mant >> w:
        mant >>= 1
        e += 1
    assert fmt.emin <= e <= fmt.emax  # |value| <= 1 + eps: never saturates
    return mf.MPF(sign, e, mant)


def cexp_turns(phi: int, phw: int, fmt: mf.Format, cfg: CexpCfg) -> tuple[mf.MPF, mf.MPF]:
    """(cos, sin) of 2*pi*phi/2^phw as MPF values."""
    assert 0 <= phi < (1 << phw)
    fg = cfg.fg
    tbl = _table(cfg)
    invfact = expln._exp_consts(expln.load_cfg(cfg.width))[2:]

    hi = phi >> (phw - cfg.segw)
    lo = phi & ((1 << (phw - cfg.segw)) - 1)
    twopi = tbl[-1]  # scale 2^(fg-3)
    z = (lo * twopi) >> (phw - 3)  # scale 2^fg, z in [0, 2*pi*2^-segw)

    are, aim = invfact[cfg.terms - 1], 0
    for j in range(cfg.terms - 2, -1, -1):
        nre = invfact[j] - ((z * aim) >> fg)
        nim = (z * are) >> fg
        are, aim = nre, nim

    tc, ts = tbl[2 * hi], tbl[2 * hi + 1]
    ore = ((tc * are) - (ts * aim)) >> fg
    oim = ((tc * aim) + (ts * are)) >> fg
    return _fix_to_mpf(ore, fmt, fg), _fix_to_mpf(oim, fmt, fg)
