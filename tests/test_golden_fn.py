"""Golden sincos_turns vs mpmath: accuracy within the DESIGN.md budget (2 ulp).

ZETA_TEST_N scales the random-phase count (default 20k; the M4 acceptance run
uses 1e6).
"""

import os
import random

import mpmath as mp
import pytest

from zetafpga.golden import fn

PHW = 96
CFG = fn.load_cfg()
N = int(os.environ.get("ZETA_TEST_N", "20000"))
ULP = 2  # budget, in ulps of 2^-(ow-2)


def _check(phase: int) -> None:
    s, c = fn.sincos_turns(phase, PHW, CFG)
    scale = mp.mpf(2) ** (CFG.ow - 2)
    t = mp.mpf(phase) / mp.mpf(2) ** PHW
    ref_s = mp.sinpi(2 * t) * scale
    ref_c = mp.cospi(2 * t) * scale
    err_s = abs(mp.mpf(s) - ref_s)
    err_c = abs(mp.mpf(c) - ref_c)
    assert err_s <= ULP and err_c <= ULP, (
        f"phase={phase:#x}: sin err={mp.nstr(err_s, 4)} ulp, cos err={mp.nstr(err_c, 4)} ulp"
    )


@pytest.fixture(autouse=True)
def _prec() -> None:
    mp.mp.prec = 160


def test_boundaries() -> None:
    qw = PHW - 2
    for base in [0, 1 << qw, 2 << qw, 3 << qw]:
        for off in [0, 1, 2, (1 << qw) - 2, (1 << qw) - 1]:
            _check((base + off) % (1 << PHW))
    # segment boundaries within the first quadrant
    seg_step = 1 << (qw - CFG.segw)
    for seg in [0, 1, 511, 512, 1023]:
        for off in [0, 1, seg_step - 1]:
            _check(seg * seg_step + off)
    # phase wrap
    _check((1 << PHW) - 1)


def test_random_phases() -> None:
    rng = random.Random(1234)
    for _ in range(N):
        _check(rng.getrandbits(PHW))


def test_exact_cardinal_points() -> None:
    # 0, 1/4, 1/2, 3/4 turns must be exact: sin/cos in {0, ±1}.
    one = 1 << (CFG.ow - 2)
    qw = PHW - 2
    assert fn.sincos_turns(0, PHW, CFG) == (0, one)
    assert fn.sincos_turns(1 << qw, PHW, CFG) == (one, 0)
    assert fn.sincos_turns(2 << qw, PHW, CFG) == (0, -one)
    assert fn.sincos_turns(3 << qw, PHW, CFG) == (-one, 0)
