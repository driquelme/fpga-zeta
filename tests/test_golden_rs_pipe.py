"""Golden pipelined RS power sum: accuracy vs direct mpmath, and consistency
with the sequential COMPUTE_PS path."""

import mpmath as mp
import pytest

from zetafpga.golden import mpfloat as mf
from zetafpga.golden import rs_pipe, zeta_em
from zetafpga.kernel.em_setup import mpf_value
from zetafpga.kernel.rs_setup import rs_entry, rs_program

FMTS = [mf.Format(limbs=1), mf.Format(limbs=2)]


@pytest.fixture(autouse=True)
def _prec() -> None:
    mp.mp.prec = 400


def _phw_bw(fmt: mf.Format) -> tuple[int, int]:
    return fmt.width + 32, fmt.width + 64


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_vs_direct_sum(fmt: mf.Format) -> None:
    phw, bw = _phw_bw(fmt)
    for tv, n in [(100.0, 3), (1000.0, 12), (100_000.0, 126)]:
        t_fx = int(mp.nint(mp.mpf(tv) * (1 << 32)))
        entries = [rs_entry(k, fmt, bw) for k in range(1, n + 1)]
        re, im = rs_pipe.rs_power_sum(t_fx, entries, fmt, phw, bw)
        t = mp.mpf(t_fx) / (1 << 32)
        ref = sum(mp.exp(mp.mpc(0, -t * mp.ln(k))) / mp.sqrt(k) for k in range(1, n + 1))
        got = mp.mpc(mpf_value(re, fmt), mpf_value(im, fmt))
        # per-term ~2 ulp + FXA truncation; absolute budget vs term count
        budget = n * (mp.mpf(2) ** (2 - fmt.width) + mp.mpf(2) ** -(fmt.width + 15))
        assert abs(got - ref) <= budget, f"t={tv}, N={n}: err {mp.nstr(abs(got - ref), 4)}"


def test_matches_sequential_ps_path() -> None:
    """Pipelined engine == sequential COMPUTE_PS engine within accumulator noise."""
    fmt = mf.Format(limbs=1)
    phw, bw = _phw_bw(fmt)
    t_fx = int(mp.nint(mp.mpf(5000.0) * (1 << 32)))
    prog = rs_program(t_fx, fmt)  # sequential path program (sigma = 1/2)
    zr, zi, _, _ = zeta_em.zeta_em(prog)
    entries = [rs_entry(k, fmt, bw) for k in range(1, prog.n + 1)]
    re, im = rs_pipe.rs_power_sum(t_fx, entries, fmt, phw, bw)
    seq = mp.mpc(mpf_value(zr, fmt), mpf_value(zi, fmt))
    pipe = mp.mpc(mpf_value(re, fmt), mpf_value(im, fmt))
    # Different amplitude paths (on-chip exp vs host table) and accumulators:
    # agreement within a few ulp * N
    assert abs(seq - pipe) <= prog.n * mp.mpf(2) ** (3 - fmt.width)
