"""limb_addsub: RTL vs golden model, directed + random."""

import os
import random

import cocotb
from cocotb.triggers import Timer

from zetafpga.golden import limb

LIMBS = int(os.environ.get("ZETA_LIMBS", "2"))
WIDTH = LIMBS * 64
N_RANDOM = int(os.environ.get("ZETA_TEST_N", "10000"))


async def _check(dut, a: int, b: int, sub: bool, cin: bool) -> None:
    dut.a.value = a
    dut.b.value = b
    dut.sub.value = int(sub)
    dut.cin.value = int(cin)
    await Timer(1, "ns")
    expected, exp_carry = limb.addsub(a, b, WIDTH, sub=sub, cin=cin)
    got = int(dut.result.value)
    got_carry = int(dut.carry.value)
    assert got == expected and got_carry == int(exp_carry), (
        f"{'sub' if sub else 'add'}(a={a:#x}, b={b:#x}, cin={cin}) = "
        f"({got:#x}, {got_carry}), expected ({expected:#x}, {int(exp_carry)})"
    )


@cocotb.test()
async def directed(dut) -> None:
    m = limb.mask(WIDTH)
    for sub in (False, True):
        for cin in (False, True):
            await _check(dut, 0, 0, sub, cin)
            await _check(dut, m, m, sub, cin)
            await _check(dut, m, 0, sub, cin)
            await _check(dut, 0, m, sub, cin)
            await _check(dut, m, 1, sub, cin)
            await _check(dut, 1, m, sub, cin)
    # carry/borrow propagation across every limb boundary
    for k in range(1, LIMBS + 1):
        boundary = (1 << (64 * k)) - 1  # all-ones through limb k
        await _check(dut, boundary & limb.mask(WIDTH), 1, False, False)
        await _check(dut, (boundary + 1) & limb.mask(WIDTH), 1, True, False)


@cocotb.test()
async def random_vectors(dut) -> None:
    rng = random.Random(LIMBS)
    for _ in range(N_RANDOM):
        a = rng.getrandbits(WIDTH)
        b = rng.getrandbits(WIDTH)
        await _check(dut, a, b, rng.random() < 0.5, rng.random() < 0.5)


@cocotb.test()
async def random_near_boundaries(dut) -> None:
    """a and b close in value: exercises long borrow chains and result ~0."""
    rng = random.Random(1000 + LIMBS)
    for _ in range(N_RANDOM // 4):
        a = rng.getrandbits(WIDTH)
        b = (a + rng.randrange(-4, 5)) & limb.mask(WIDTH)
        await _check(dut, a, b, True, rng.random() < 0.5)


def test_limb_addsub(limbs: int) -> None:
    from common.runner import run_block

    run_block(
        "rtl/common/mp/limb_addsub.sv",
        "limb_addsub",
        __file__,
        parameters={"LIMBS": limbs},
    )
