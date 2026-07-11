"""limb_shift: RTL vs golden model, directed + random."""

import os
import random

import cocotb
from cocotb.triggers import Timer

from zetafpga.golden import limb

LIMBS = int(os.environ.get("ZETA_LIMBS", "2"))
WIDTH = LIMBS * 64
N_RANDOM = int(os.environ.get("ZETA_TEST_N", "10000"))


def _amt_max() -> int:
    """Largest value expressible in the RTL's AMTW = $clog2(WIDTH+1) bits."""
    amtw = (WIDTH).bit_length()  # clog2(WIDTH+1) for WIDTH a power of two multiple
    return (1 << amtw) - 1


async def _check(dut, a: int, amount: int, left: bool) -> None:
    dut.a.value = a
    dut.amount.value = amount
    dut.left.value = int(left)
    await Timer(1, "ns")
    expected, exp_lost = limb.shift(a, amount, WIDTH, left=left)
    got = int(dut.result.value)
    got_lost = int(dut.lost.value)
    assert got == expected and got_lost == int(exp_lost), (
        f"shift(a={a:#x}, amt={amount}, left={left}) = ({got:#x}, {got_lost}), "
        f"expected ({expected:#x}, {int(exp_lost)})"
    )


@cocotb.test()
async def directed(dut) -> None:
    m = limb.mask(WIDTH)
    for left in (False, True):
        for amount in [0, 1, 63, 64, 65, WIDTH - 1, WIDTH, _amt_max()]:
            if amount > _amt_max():
                continue
            await _check(dut, m, amount, left)
            await _check(dut, 0, amount, left)
            await _check(dut, 1, amount, left)
            await _check(dut, 1 << (WIDTH - 1), amount, left)


@cocotb.test()
async def single_bit_walk(dut) -> None:
    for i in range(WIDTH):
        await _check(dut, 1 << i, i, False)
        await _check(dut, 1 << i, WIDTH - 1 - i, True)


@cocotb.test()
async def random_vectors(dut) -> None:
    rng = random.Random(2000 + LIMBS)
    for _ in range(N_RANDOM):
        a = rng.getrandbits(WIDTH)
        amount = rng.randrange(_amt_max() + 1)
        await _check(dut, a, amount, rng.random() < 0.5)


def test_limb_shift(limbs: int) -> None:
    from common.runner import run_block

    run_block(
        "rtl/common/mp/limb_shift.sv",
        "limb_shift",
        __file__,
        parameters={"LIMBS": limbs},
    )
