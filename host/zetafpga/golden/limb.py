"""Bit-true golden model of the limb integer core (M1).

Mirrors rtl/common/prim/lzc.sv, rtl/common/mp/limb_addsub.sv and
rtl/common/mp/limb_shift.sv exactly. RTL is verified bit-for-bit against these
functions; mathematical correctness of these functions is trivially Python int
arithmetic.

Semantics notes (must match RTL):
- ``addsub``: add computes a + b + cin; sub computes a - b - cin (cin acts as a
  borrow-in). ``carry`` is carry-out for add, borrow-out for sub.
- ``shift``: logical shift by ``amount``; amounts > width behave as a full shift
  (result 0). ``lost`` is the OR of all bits shifted out of the window.
- ``lzc``: number of leading zeros; equals ``width`` when value == 0.
"""


def mask(width: int) -> int:
    """All-ones mask of `width` bits."""
    return (1 << width) - 1


def lzc(value: int, width: int) -> int:
    """Leading-zero count of a `width`-bit value (== width for value 0)."""
    assert 0 <= value <= mask(width)
    if value == 0:
        return width
    return width - value.bit_length()


def addsub(a: int, b: int, width: int, *, sub: bool = False, cin: bool = False) -> tuple[int, bool]:
    """Add/subtract with carry/borrow. Returns (result mod 2^width, carry_or_borrow_out)."""
    assert 0 <= a <= mask(width) and 0 <= b <= mask(width)
    if sub:
        full = a - b - int(cin)
        borrow = full < 0
        return full & mask(width), borrow
    full = a + b + int(cin)
    return full & mask(width), full > mask(width)


def mul(a: int, b: int, width: int) -> int:
    """Full product of two `width`-bit values (2*width bits, exact)."""
    assert 0 <= a <= mask(width) and 0 <= b <= mask(width)
    return a * b


def shift(a: int, amount: int, width: int, *, left: bool = False) -> tuple[int, bool]:
    """Logical shift with lost-bits detection. Returns (result, lost).

    `lost` is True iff any 1-bit was shifted out of the `width`-bit window.
    Amounts >= width yield (0, a != 0) for a nonzero shift of a nonzero value.
    """
    assert 0 <= a <= mask(width)
    assert amount >= 0
    if amount == 0:
        return a, False
    if amount >= width:
        return 0, a != 0
    if left:
        result = (a << amount) & mask(width)
        lost = (a >> (width - amount)) != 0
    else:
        result = a >> amount
        lost = (a & mask(amount)) != 0
    return result, lost
