"""fx_mul_mod1: RTL vs golden, directed + random."""

import os
import random

import cocotb
from cocotb.triggers import Timer

from zetafpga.golden.fixedpt import fx_mul_mod1

AW = int(os.environ.get("ZETA_AW", "64"))
AF = int(os.environ.get("ZETA_AF", "16"))
BW = int(os.environ.get("ZETA_BW", "128"))
BI = int(os.environ.get("ZETA_BI", "0"))
PHW = int(os.environ.get("ZETA_PHW", "96"))
N_RANDOM = int(os.environ.get("ZETA_TEST_N", "10000"))


async def _check(dut, a: int, b: int) -> None:
    dut.a.value = a
    dut.b.value = b
    await Timer(1, "ns")
    expected = fx_mul_mod1(a, b, AF, BW, PHW)
    got = int(dut.frac.value)
    assert got == expected, f"a={a:#x}, b={b:#x}: got {got:#x}, expected {expected:#x}"


@cocotb.test()
async def directed(dut) -> None:
    ma, mb = (1 << AW) - 1, (1 << (BI + BW)) - 1
    for a in [0, 1, ma, 1 << (AW - 1), 1 << AF]:
        for b in [0, 1, mb, 1 << (BW - 1), mb >> 1]:
            await _check(dut, a, b)
    # products straddling the integer boundary (wrap cases)
    await _check(dut, 1 << AF, mb)  # a = 1.0 exactly
    await _check(dut, (1 << AF) + 1, mb)


@cocotb.test()
async def random_vectors(dut) -> None:
    rng = random.Random(AW * 1000 + PHW)
    for _ in range(N_RANDOM):
        await _check(dut, rng.getrandbits(AW), rng.getrandbits(BI + BW))


def test_fx_mul_mod1() -> None:
    import pytest
    from common.runner import run_block

    for aw, af, bw, bi, phw in [
        (64, 16, 128, 0, 96),
        (80, 32, 192, 8, 160),
        (64, 32, 160, 8, 96),  # npow phase configuration (Z64)
        (32, 0, 96, 0, 96),
    ]:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("ZETA_AW", str(aw))
            mp.setenv("ZETA_AF", str(af))
            mp.setenv("ZETA_BW", str(bw))
            mp.setenv("ZETA_BI", str(bi))
            mp.setenv("ZETA_PHW", str(phw))
            run_block(
                "rtl/common/fx/fx_mul_mod1.sv",
                "fx_mul_mod1",
                __file__,
                parameters={"AW": aw, "AF": af, "BW": bw, "BI": bi, "PHW": phw},
            )
