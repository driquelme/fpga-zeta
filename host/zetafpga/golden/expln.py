"""Bit-true golden model of exp_mpf and log_mpf (M5).

Mirrors rtl/common/fn/exp_mpf.sv and log_mpf.sv exactly: same fixed-point
scales (FG fractional working bits), same truncation (floor) at every step,
same rounding of the final mantissa. Constants come from the same committed
ROM files the RTL loads with $readmemh.

Accuracy contract (validated in tests/test_golden_expln.py, DESIGN.md budget):
- exp_mpf: <= 2 ulp for all representable y (saturating outside exponent range)
- log_mpf: <= 2 ulp for x with |ln x| >= 2^-8; near x = 1 the *absolute* error
  is <= 2^-(width+12) but relative ulp accuracy degrades (documented band;
  zeta-family callers use ln n, n >= 2, and Stirling arguments far from 1).
- Domain: log of zero/negative/special returns is_special.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from zetafpga.golden import mpfloat as mf

TABLES_DIR = Path(__file__).resolve().parents[3] / "rtl" / "common" / "fn" / "tables"


@dataclass(frozen=True)
class ExpLnCfg:
    width: int
    fg: int
    constw: int
    terms: int

    @property
    def stem(self) -> str:
        return f"expln_w{self.width}"


def load_cfg(width: int) -> ExpLnCfg:
    meta = json.loads((TABLES_DIR / f"expln_w{width}.json").read_text())
    return ExpLnCfg(**meta)


@lru_cache(maxsize=8)
def _load(path: str, constw: int) -> tuple[int, ...]:
    values = []
    for line in Path(path).read_text().splitlines():
        v = int(line, 16)
        if v >> (constw - 1):
            v -= 1 << constw
        values.append(v)
    return tuple(values)


def _exp_consts(cfg: ExpLnCfg) -> tuple[int, ...]:
    return _load(str(TABLES_DIR / f"{cfg.stem}_exp.mem"), cfg.constw)


def _ln_table(cfg: ExpLnCfg) -> tuple[int, ...]:
    return _load(str(TABLES_DIR / f"{cfg.stem}_ln.mem"), cfg.constw)


def _round_norm(sign: int, e: int, t: int, fmt: mf.Format) -> tuple[mf.MPF, bool, bool]:
    """Round a W+1-bit normalized candidate (top bit set) and saturate."""
    mant = (t + 1) >> 1
    if mant >> fmt.width:
        mant >>= 1
        e += 1
    if e > fmt.emax:
        return mf.special(sign), True, False
    if e < fmt.emin:
        return mf.zero(sign), False, True
    return mf.MPF(sign, e, mant), False, False


def exp_fx(yfx: int, fmt: mf.Format, cfg: ExpLnCfg) -> tuple[mf.MPF, bool, bool]:
    """e^y for y given directly as signed fixed point at scale 2^FG.

    This is the fused-path entry (npow_s_kernel): supplying y at FG working
    precision avoids the W-bit MPF quantization of intermediate products,
    whose 0.5 ulp would otherwise be amplified by |y| in the exponential.
    """
    w, fg = fmt.width, cfg.fg
    if yfx == 0:
        return mf.MPF(0, 1, 1 << (w - 1)), False, False  # e^0 = 1.0
    if yfx >= (1 << (fg + 21)):
        return mf.special(0), True, False
    if yfx <= -(1 << (fg + 21)):
        return mf.zero(0), False, True

    consts = _exp_consts(cfg)
    ln2c, invln2 = consts[0], consts[1]
    invfact = consts[2:]

    p = yfx * invln2  # scale 2^(2*fg)
    k = (p + (1 << (2 * fg - 1))) >> (2 * fg)
    r = yfx - k * ln2c  # scale 2^fg, |r| <= ln2/2 + eps

    acc = invfact[cfg.terms - 1]
    for j in range(cfg.terms - 2, -1, -1):
        acc = ((acc * r) >> fg) + invfact[j]
    # acc = e^r in [0.70, 1.42] at scale 2^fg

    if acc >= (1 << fg):
        return _round_norm(0, k + 1, acc >> (fg - w), fmt)
    return _round_norm(0, k, acc >> (fg - w - 1), fmt)


def exp_mpf(y: mf.MPF, fmt: mf.Format, cfg: ExpLnCfg) -> tuple[mf.MPF, bool, bool]:
    """e^y, following the RTL step for step."""
    w, fg = fmt.width, cfg.fg
    if y.is_special:
        return mf.special(y.sign), False, False
    if y.is_zero:
        return mf.MPF(0, 1, 1 << (w - 1)), False, False
    if y.exp > 21:  # |y| >= 2^20 > emax*ln2: saturate by sign
        if y.sign:
            return mf.zero(0), False, True
        return mf.special(0), True, False

    s = fg + y.exp - w
    yfx = (y.mant << s) if s >= 0 else (y.mant >> -s)
    if y.sign:
        yfx = -yfx
    return exp_fx(yfx, fmt, cfg)


def log_mpf(x: mf.MPF, fmt: mf.Format, cfg: ExpLnCfg) -> tuple[mf.MPF, bool, bool]:
    """ln x, following the RTL step for step."""
    w, fg = fmt.width, cfg.fg
    if x.is_special or x.is_zero or x.sign:
        return mf.special(x.sign), False, False
    if x.exp == 1 and x.mant == 1 << (w - 1):
        return mf.zero(0), False, False  # ln(1) = 0 exactly

    consts = _exp_consts(cfg)
    lntbl = _ln_table(cfg)
    one = 1 << fg

    v = x.mant << (fg - w)  # mu in [0.5, 1) at scale 2^fg
    acc = x.exp * consts[0]  # e * ln2 at scale 2^fg
    for i in range(1, fg + 1):
        trial = v + (v >> i)
        if trial <= one:
            v = trial
            acc -= lntbl[i - 1]
    acc -= one - v  # residual -(1 - v), v in (1 - 2^-fg, 1]

    if acc == 0:
        return mf.zero(0), False, False
    sign = 1 if acc < 0 else 0
    a = -acc if sign else acc
    p = a.bit_length() - 1
    e = p - fg + 1
    t = (a >> (p - w)) if p >= w else (a << (w - p))
    return _round_norm(sign, e, t, fmt)
