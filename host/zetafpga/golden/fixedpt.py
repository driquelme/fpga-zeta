"""Bit-true golden model of the wide fixed-point phase operators (M4).

Mirrors rtl/common/fx/fx_mul_mod1.sv.
"""


def fx_mul_mod1(a: int, b: int, af: int, bw: int, phw: int) -> int:
    """frac(a * b) truncated to the top `phw` fractional bits.

    a: unsigned fixed point with `af` fractional bits (any width).
    b: unsigned fixed point with `bw` FRACTIONAL bits (integer bits allowed —
       needed when a has fractional bits, since frac(a*(K+f)) != frac(a*f)).
    Result: integer P with value P / 2^phw = frac(a * b) truncated.
    Requires af + bw >= phw.
    """
    assert af + bw >= phw
    assert a >= 0 and b >= 0
    fb = af + bw  # fractional bits of the raw product
    return ((a * b) & ((1 << fb) - 1)) >> (fb - phw)
