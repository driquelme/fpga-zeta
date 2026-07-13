"""Bit-true golden model of mpf_recip: Newton reciprocal, no divider (M16).

y = 1/x by Newton iteration on the mantissa in fixed point at F = width + 8
guard bits:  y_{n+1} = y_n * (2 - m * y_n),  seeded with the classical
minimax line y_0 = 48/17 - 32/17 * m  (|rel err| <= 1/17 ~ 2^-4.09, doubling
per iteration; NITER = clog2(width) over-converges past the 2^-(W+2) target).
All intermediate products floor-truncate; one final RNE to the mantissa.

Accuracy contract (tests/test_golden_recip.py): <= 2 ulp vs the exact value;
exact for powers of two. 1/0 saturates to special with ovf. Mirrors
rtl/common/mp/mpf_recip.sv step for step.
"""

from zetafpga.golden import mpfloat as mf


def _niter(width: int) -> int:
    return (width - 1).bit_length()  # $clog2(width)


def mpf_recip(x: mf.MPF, fmt: mf.Format) -> tuple[mf.MPF, bool, bool]:
    """1/x, following the RTL step for step."""
    w = fmt.width
    f = w + 8
    if x.is_special:
        return mf.special(x.sign), False, False
    if x.is_zero:
        return mf.special(x.sign), True, False

    k48 = (48 << f) // 17
    k32 = (32 << f) // 17
    m = x.mant  # value m/2^w in [0.5, 1)
    y = k48 - ((k32 * m) >> w)  # scale 2^f, y ~ 1/(m/2^w) in (1, 2]
    for _ in range(_niter(w)):
        e = (2 << f) - ((m * y) >> w)
        y = (y * e) >> f

    t = y >> (f - w)  # w+1 bits, top bit set
    mant = (t + 1) >> 1
    e_r = 1 - x.exp
    if mant >> w:
        mant >>= 1
        e_r += 1
    return mf._saturate(x.sign, e_r, mant, fmt)
