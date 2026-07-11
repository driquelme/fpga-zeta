"""Bit-true golden model of the MPF floating-point format and operators (M3).

Mirrors rtl/common/pkg/mp_pkg.sv (layout) and rtl/common/mp/mpf_{add,mul}.sv
exactly. See DESIGN.md for the format rationale.

Layout of a packed MPF word (LSB first), MPW = width + expw + 3 bits:

    [width-1        : 0]  mantissa, normalized (MSB = 1) unless zero/special
    [width+expw-1   : width]  exponent, two's complement
    [width+expw]          sign
    [width+expw+1]        is_zero
    [width+expw+2]        is_special (inf/nan collapsed)

Value = (-1)^sign * mant * 2^(exp - width), i.e. the mantissa is a fixed-point
fraction in [1/2, 1). Rounding is RNE. No subnormals: exponent overflow
saturates to special (with ovf flag), underflow to zero (with unf flag).

Algorithm notes (RTL is structured identically):
- mpf_add aligns into a 2*width+3-bit window, which holds every bit of the
  smaller operand for exponent differences up to width+2 — the sum/difference
  is exact, so a single final RNE rounding is correct by construction. For
  larger differences the small operand collapses to a 1-in-the-LSB sticky,
  which is below the guard position and rounds identically to the true value.
- mpf_mul rounds the exact 2*width-bit product.
"""

from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class Format:
    """An MPF format instance (LIMBS x 64-bit mantissa, EXPW-bit exponent)."""

    limbs: int
    expw: int = 20

    @cached_property
    def width(self) -> int:
        return self.limbs * 64

    @cached_property
    def mpw(self) -> int:
        return self.width + self.expw + 3

    @cached_property
    def emax(self) -> int:
        return (1 << (self.expw - 1)) - 1

    @cached_property
    def emin(self) -> int:
        return -(1 << (self.expw - 1))


@dataclass(frozen=True)
class MPF:
    sign: int
    exp: int
    mant: int
    is_zero: bool = False
    is_special: bool = False


def zero(sign: int = 0) -> MPF:
    return MPF(sign, 0, 0, is_zero=True)


def special(sign: int = 0) -> MPF:
    return MPF(sign, 0, 0, is_special=True)


def pack(v: MPF, fmt: Format) -> int:
    w = fmt.width
    word = v.mant
    word |= (v.exp & ((1 << fmt.expw) - 1)) << w
    word |= v.sign << (w + fmt.expw)
    word |= int(v.is_zero) << (w + fmt.expw + 1)
    word |= int(v.is_special) << (w + fmt.expw + 2)
    return word


def unpack(word: int, fmt: Format) -> MPF:
    w = fmt.width
    mant = word & ((1 << w) - 1)
    exp = (word >> w) & ((1 << fmt.expw) - 1)
    if exp >> (fmt.expw - 1):  # sign-extend two's complement
        exp -= 1 << fmt.expw
    sign = (word >> (w + fmt.expw)) & 1
    is_zero = bool((word >> (w + fmt.expw + 1)) & 1)
    is_special = bool((word >> (w + fmt.expw + 2)) & 1)
    return MPF(sign, exp, mant, is_zero, is_special)


def _saturate(sign: int, exp: int, mant: int, fmt: Format) -> tuple[MPF, bool, bool]:
    if exp > fmt.emax:
        return special(sign), True, False
    if exp < fmt.emin:
        return zero(sign), False, True
    return MPF(sign, exp, mant), False, False


def mpf_mul(x: MPF, y: MPF, fmt: Format) -> tuple[MPF, bool, bool]:
    """Multiply: returns (result, ovf, unf). Correctly rounded (RNE)."""
    sign = x.sign ^ y.sign
    if x.is_special or y.is_special:
        return special(sign), False, False
    if x.is_zero or y.is_zero:
        return zero(sign), False, False
    w = fmt.width
    p = x.mant * y.mant  # exact, 2w bits
    if (p >> (2 * w - 1)) & 1:
        e = x.exp + y.exp
    else:
        p <<= 1
        e = x.exp + y.exp - 1
    mant = p >> w
    g = (p >> (w - 1)) & 1
    s = (p & ((1 << (w - 1)) - 1)) != 0
    if g and (s or (mant & 1)):
        mant += 1
        if mant >> w:  # all-ones rounded up
            mant >>= 1
            e += 1
    return _saturate(sign, e, mant, fmt)


def mpf_add(x: MPF, y: MPF, fmt: Format) -> tuple[MPF, bool, bool]:
    """Add (signed): returns (result, ovf, unf). Correctly rounded (RNE)."""
    if x.is_special or y.is_special:
        return special(x.sign if x.is_special else y.sign), False, False
    if x.is_zero and y.is_zero:
        return zero(x.sign & y.sign), False, False
    if x.is_zero:
        return y, False, False
    if y.is_zero:
        return x, False, False

    w = fmt.width
    ew = 2 * w + 3
    bias = 1 << (fmt.expw - 1)
    kx = ((x.exp + bias) << w) | x.mant
    ky = ((y.exp + bias) << w) | y.mant
    big, sml = (x, y) if kx >= ky else (y, x)
    eff_sub = x.sign != y.sign
    d = big.exp - sml.exp

    mb = big.mant << (w + 2)
    ms = (sml.mant << (w + 2 - d)) if d <= w + 2 else 1
    r = mb - ms if eff_sub else mb + ms
    if r == 0:
        return zero(0), False, False

    lz = (ew - 1) - (r.bit_length() - 1)
    rn = r << lz
    mant = rn >> (w + 3)
    g = (rn >> (w + 2)) & 1
    s = (rn & ((1 << (w + 2)) - 1)) != 0
    e = big.exp + 1 - lz
    if g and (s or (mant & 1)):
        mant += 1
        if mant >> w:
            mant >>= 1
            e += 1
    sign = big.sign if eff_sub else x.sign
    return _saturate(sign, e, mant, fmt)
