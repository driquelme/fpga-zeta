"""M15 acceptance: fully on-chip Z(t) (golden power sum + golden Z epilogue).

Z vs mp.siegelz within the RS truncation budget (the K=4 remainder
~t^(-11/4) dominates at small t; the engine's own error floor ~2^-(W-16)
matters only at large t where the remainder has fallen below it).
"""

import mpmath as mp
import pytest

from zetafpga.golden import mpfloat as mf
from zetafpga.golden import rs_pipe, rs_z, theta
from zetafpga.kernel.rs_setup import rs_entry, rs_n

FMTS = [mf.Format(limbs=1), mf.Format(limbs=2)]


@pytest.fixture(autouse=True)
def _prec() -> None:
    mp.mp.prec = 400


def _budget(t: float, width: int) -> float:
    return float(0.1 * t ** (-11 / 4) + mp.mpf(2) ** -(width - 16))


def _golden_z(t_fx: int, fmt: mf.Format) -> float:
    phw = fmt.width + 32
    bw = phw + 32
    prep = rs_z.z_prep(t_fx, fmt)
    assert prep.n == rs_n(t_fx)  # on-chip floor(sqrt(t/2pi)) matches the host rule
    entries = [rs_entry(k, fmt, bw) for k in range(1, prep.n + 1)]
    s_re, s_im = rs_pipe.rs_power_sum(t_fx, entries, fmt, phw, bw)
    z = rs_z.z_post(prep, s_re, s_im, fmt)
    return float(mp.mpf(z.mant) / (1 << fmt.width) * mp.mpf(2) ** z.exp * (-1 if z.sign else 1))


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_z_onchip_vs_siegelz(fmt: mf.Format) -> None:
    t_min = theta.load_cfg(fmt.width).t_min
    ts = [float(t_min), 50.0, 100.0, 500.0, 1000.0, 10_000.0, 100_000.0]
    for tv in ts:
        if tv < t_min:
            continue
        t_fx = int(mp.nint(mp.mpf(tv) * (1 << 32)))
        got = _golden_z(t_fx, fmt)
        ref = float(mp.siegelz(mp.mpf(t_fx) / (1 << 32)))
        assert abs(got - ref) <= _budget(tv, fmt.width), (
            f"t={tv}: Z={got!r} vs siegelz={ref!r} (budget {_budget(tv, fmt.width):.3e})"
        )
