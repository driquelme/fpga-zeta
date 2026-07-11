"""Host-side generation of the ln(n) tables streamed to the engine.

These are runtime WRITE_TABLE payloads (per the overlay model), not committed
ROMs. Each entry provides ln(n) in the two forms the npow kernel consumes:

- lnn_fx: ln(n) as unsigned fixed point at scale 2^FG (Q8.FG, FG = width+24).
  Full working precision — supplying ln(n) at only W bits would let the
  exponential amplify the table rounding by |sigma| (see golden/npow.py).
- lnn2pi: the FULL value ln(n)/2pi as Q8.BW fixed point, BW = PHW + 32.
  The integer part must be included: for non-integer t, frac(t*(K+f)) is not
  frac(t*f) — the integer part K beats against t's fractional bits. (Storing
  only the fractional part is valid only for integer t; this was caught by
  the true-value npow test.) The 32 guard bits absorb t up to 2^32.
"""

from functools import lru_cache

import mpmath as mp

from zetafpga.golden import mpfloat as mf


@lru_cache(maxsize=1 << 20)
def lnn_entry(n: int, fmt: mf.Format, bw: int) -> tuple[int, int]:
    """(ln n at scale 2^FG, ln(n)/2pi as Q8.bw — integer part included)."""
    assert n >= 1
    if n == 1:
        return 0, 0
    fg = fmt.width + 24
    with mp.workprec(fg + bw + 64):
        x = mp.ln(n)
        lnn_fx = int(mp.nint(x * mp.mpf(2) ** fg))
        assert lnn_fx < (1 << (fg + 8))
        lnn2pi = int(mp.floor(x / (2 * mp.pi) * mp.mpf(2) ** bw))
        assert lnn2pi < (1 << (bw + 8))
    return lnn_fx, lnn2pi


def lnn_table(n_max: int, fmt: mf.Format, bw: int) -> list[tuple[int, int]]:
    return [lnn_entry(n, fmt, bw) for n in range(1, n_max + 1)]
