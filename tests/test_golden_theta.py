"""Golden theta_turns vs mp.siegeltheta: absolute accuracy in turns (mod 1)."""

import mpmath as mp
import pytest

from zetafpga.golden import mpfloat as mf
from zetafpga.golden import theta
from zetafpga.kernel.em_setup import mpf_from_real

FMTS = [mf.Format(limbs=1), mf.Format(limbs=2)]


@pytest.fixture(autouse=True)
def _prec() -> None:
    mp.mp.prec = 600


def _inv_t(t_fx: int, w2: int) -> mf.MPF:
    fmt2 = mf.Format(limbs=w2 // 64)
    return mpf_from_real(mp.mpf(2) ** 32 / t_fx, fmt2)


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_theta_vs_siegeltheta(fmt: mf.Format) -> None:
    cfg = theta.load_cfg(fmt.width)
    phw = fmt.width + 32
    # accuracy target: comfortably below the target format's Z needs
    bound = mp.mpf(2) ** -(fmt.width + 8)
    ts = [float(cfg.t_min), cfg.t_min + 0.5, 50.0, 100.0, 1234.5678, 1e5, 1e7, 4e9]
    for tv in ts:
        if tv < cfg.t_min:
            continue
        t_fx = int(mp.nint(mp.mpf(tv) * (1 << 32)))
        got, _ = theta.theta_turns(t_fx, _inv_t(t_fx, cfg.w2), fmt, phw, cfg)
        t = mp.mpf(t_fx) / (1 << 32)
        ref = mp.frac(mp.siegeltheta(t) / (2 * mp.pi))
        gv = mp.mpf(got) / mp.mpf(2) ** phw
        err = abs(gv - ref)
        err = min(err, 1 - err)  # circular distance
        assert err <= bound + mp.mpf(2) ** -phw, (
            f"t={tv}: theta err {mp.nstr(err, 4)} turns (bound {mp.nstr(bound, 3)})"
        )


def test_below_threshold_documented() -> None:
    cfg = theta.load_cfg(64)
    assert cfg.t_min == 18  # host computes theta below this (documented contract)
