"""lzc: RTL vs golden model, directed + random."""

import os
import random

import cocotb
from cocotb.triggers import Timer

from zetafpga.golden import limb

WIDTH = int(os.environ.get("ZETA_WIDTH", "64"))
N_RANDOM = int(os.environ.get("ZETA_TEST_N", "10000"))


async def _check(dut, value: int) -> None:
    dut.data.value = value
    await Timer(1, "ns")
    expected = limb.lzc(value, WIDTH)
    got = int(dut.count.value)
    assert got == expected, f"lzc({value:#x}) = {got}, expected {expected}"
    assert int(dut.all_zero.value) == (value == 0)


@cocotb.test()
async def directed(dut) -> None:
    await _check(dut, 0)
    await _check(dut, limb.mask(WIDTH))
    for i in range(WIDTH):
        await _check(dut, 1 << i)
        await _check(dut, limb.mask(WIDTH) >> i)


@cocotb.test()
async def random_vectors(dut) -> None:
    rng = random.Random(WIDTH)
    for _ in range(N_RANDOM):
        # Bias toward small values so high leading-zero counts are exercised.
        bits = rng.randrange(1, WIDTH + 1)
        await _check(dut, rng.getrandbits(bits))


def test_lzc(width: int) -> None:
    from common.runner import run_block

    run_block("rtl/common/prim/lzc.sv", "lzc", __file__, parameters={"WIDTH": width})
