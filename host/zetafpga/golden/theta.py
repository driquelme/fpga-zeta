"""Bit-true golden model of theta_turns: on-chip theta(t)/2pi mod 1 (M14).

  theta/2pi = (u/2)(ln u - 1) - 1/16 + sum_k (c_k/2pi) t^(1-2k),  u = t/2pi

Works at W2 = width+64 internal precision (the t*ln t magnitude costs
~log2(t) <= 32 bits plus guard), all in the wide-fixed/mod-1 style of the
phase engine: the main term is formed in Q(38).(FG2) fixed point where the
mod-1 wrap is exact, then truncated to the target's PHW turns.

Validity: t >= t_min(width) (asymptotic floor e^(-pi t); see theta_w*.json).
Mirrors rtl/common/fn/theta_turns.sv step for step.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from zetafpga.golden import expln
from zetafpga.golden import mpfloat as mf

TABLES_DIR = Path(__file__).resolve().parents[3] / "rtl" / "common" / "fn" / "tables"


@dataclass(frozen=True)
class ThetaCfg:
    width: int
    w2: int
    fg2: int
    k: int
    t_min: int

    @property
    def stem(self) -> str:
        return f"theta_w{self.width}"


def load_cfg(width: int) -> ThetaCfg:
    meta = json.loads((TABLES_DIR / f"theta_w{width}.json").read_text())
    return ThetaCfg(**meta)


@lru_cache(maxsize=8)
def _consts(stem: str, w2: int) -> tuple[int, tuple[mf.MPF, ...]]:
    fmt2 = mf.Format(limbs=w2 // 64)
    inv2pi = int((TABLES_DIR / f"{stem}_fx.mem").read_text().strip(), 16)
    coeffs = tuple(
        mf.unpack(int(line, 16), fmt2)
        for line in (TABLES_DIR / f"{stem}_mpf.mem").read_text().splitlines()
    )
    return inv2pi, coeffs


def _ufx_to_mpf(u: int, fg2: int, fmt2: mf.Format) -> mf.MPF:
    """Normalize the u fixed-point value (scale 2^FG2, u >= 1) into MPF@W2."""
    assert u > 0
    p = u.bit_length() - 1
    e = p - fg2 + 1
    w = fmt2.width
    t = (u >> (p - w)) if p >= w else (u << (w - p))
    mant = (t + 1) >> 1
    if mant >> w:
        mant >>= 1
        e += 1
    return mf.MPF(0, e, mant)


def _mpf_to_fx(v: mf.MPF, fg2: int, fmt2: mf.Format) -> int:
    """MPF@W2 -> signed fixed at scale 2^FG2 (truncate toward zero)."""
    if v.is_zero:
        return 0
    assert not v.is_special
    sh = fg2 + v.exp - fmt2.width
    mag = (v.mant << sh) if sh >= 0 else (v.mant >> -sh)
    return -mag if v.sign else mag


def theta_turns(
    t_fx: int, inv_t: mf.MPF, fmt: mf.Format, phw: int, cfg: ThetaCfg
) -> tuple[int, mf.MPF]:
    """(theta(t)/2pi mod 1 truncated to phw bits, ln(t/2pi) as MPF@W2).

    inv_t = 1/t as MPF@W2. ln u is exported because the RS Z-epilogue (M15)
    needs ln(t/2pi) at W2 for its sqrt/power chain — reusing it avoids a
    second log unit.
    """
    fg2 = cfg.fg2
    fmt2 = mf.Format(limbs=cfg.w2 // 64)
    ecfg2 = expln.load_cfg(cfg.w2)
    inv2pi, coeffs = _consts(cfg.stem, cfg.w2)

    # u = t/(2pi) at scale 2^FG2 (t_fx is Q32.32)
    u = (t_fx * inv2pi) >> 64

    # ln u at W2 precision
    lnu, _, _ = expln.log_mpf(_ufx_to_mpf(u, fg2, fmt2), fmt2, ecfg2)
    one2 = mf.MPF(0, 1, 1 << (fmt2.width - 1))
    lm1, _, _ = mf.mpf_add(lnu, mf.MPF(1, one2.exp, one2.mant), fmt2)  # ln u - 1
    l_fx = _mpf_to_fx(lm1, fg2, fmt2)

    # main term (u/2)(ln u - 1) mod 4, at scale 2^FG2
    mask = (1 << (fg2 + 2)) - 1
    theta = ((u * l_fx) >> (fg2 + 1)) & mask

    # - 1/16
    theta = (theta - (1 << (fg2 - 4))) & mask

    # tail: inv_t * Horner(coeffs; v = inv_t^2), added at scale 2^FG2
    v2, _, _ = mf.mpf_mul(inv_t, inv_t, fmt2)
    s = coeffs[cfg.k - 1]
    for j in range(cfg.k - 2, -1, -1):
        sm, _, _ = mf.mpf_mul(s, v2, fmt2)
        s, _, _ = mf.mpf_add(coeffs[j], sm, fmt2)
    tail, _, _ = mf.mpf_mul(s, inv_t, fmt2)
    theta = (theta + _mpf_to_fx(tail, fg2, fmt2)) & mask

    # top PHW fractional bits of the mod-1 value
    return (theta & ((1 << fg2) - 1)) >> (fg2 - phw), lnu
