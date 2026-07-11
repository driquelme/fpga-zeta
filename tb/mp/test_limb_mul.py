"""limb_mul: RTL vs golden (Python int), streaming II=1 verification."""

import os
import random

import cocotb
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from zetafpga.golden import limb

LIMBS = int(os.environ.get("ZETA_LIMBS", "2"))
WIDTH = LIMBS * 64
N_RANDOM = int(os.environ.get("ZETA_TEST_N", "10000"))


async def _clock(clk) -> None:
    while True:
        clk.value = 0
        await Timer(5, "ns")
        clk.value = 1
        await Timer(5, "ns")


async def _reset(dut) -> None:
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.a.value = 0
    dut.b.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def _drive(dut, pairs: list[tuple[int, int]], p_valid: float, rng: random.Random) -> None:
    for a, b in pairs:
        while rng.random() > p_valid:
            dut.in_valid.value = 0
            await RisingEdge(dut.clk)
        dut.in_valid.value = 1
        dut.a.value = a
        dut.b.value = b
        await RisingEdge(dut.clk)
    dut.in_valid.value = 0


async def _collect(dut, count: int) -> list[tuple[int, int]]:
    """Returns (cycle_index, product) samples taken on falling edges."""
    out: list[tuple[int, int]] = []
    cycle = 0
    while len(out) < count:
        await FallingEdge(dut.clk)
        cycle += 1
        if dut.out_valid.value == 1:
            out.append((cycle, int(dut.product.value)))
    return out


async def _run(
    dut, pairs: list[tuple[int, int]], p_valid: float, seed: int
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    cocotb.start_soon(_clock(dut.clk))
    await _reset(dut)
    cocotb.start_soon(_drive(dut, pairs, p_valid, rng))
    got = await _collect(dut, len(pairs))
    expected = [limb.mul(a, b, WIDTH) for a, b in pairs]
    values = [v for _, v in got]
    assert values == expected, (
        f"first mismatch at index "
        f"{next(i for i, (g, e) in enumerate(zip(values, expected, strict=True)) if g != e)}"
    )
    return got


@cocotb.test()
async def directed(dut) -> None:
    m = limb.mask(WIDTH)
    pairs = [(0, 0), (1, 1), (m, m), (m, 1), (1, m), (m, 0)]
    pairs += [(1 << i, 1 << (WIDTH - 1 - i)) for i in range(WIDTH)]
    # all-ones times single-bit: exercises every carry column
    pairs += [(m, 1 << i) for i in range(0, WIDTH, 7)]
    await _run(dut, pairs, p_valid=1.0, seed=10)


@cocotb.test()
async def back_to_back_ii1(dut) -> None:
    """Full-rate streaming: out_valid must be contiguous once results start."""
    rng = random.Random(20 + LIMBS)
    pairs = [(rng.getrandbits(WIDTH), rng.getrandbits(WIDTH)) for _ in range(200)]
    got = await _run(dut, pairs, p_valid=1.0, seed=20)
    first = got[0][0]
    cycles = [c for c, _ in got]
    assert cycles == list(range(first, first + len(pairs))), (
        "out_valid not contiguous at full input rate — pipeline is not II=1"
    )


@cocotb.test()
async def random_gapped(dut) -> None:
    rng = random.Random(30 + LIMBS)
    pairs = [(rng.getrandbits(WIDTH), rng.getrandbits(WIDTH)) for _ in range(N_RANDOM)]
    await _run(dut, pairs, p_valid=0.7, seed=30)


def test_limb_mul(limbs: int) -> None:
    from common.runner import run_block

    run_block(
        ["rtl/arch/generic-sim/mul_tile.sv", "rtl/common/mp/limb_mul.sv"],
        "limb_mul",
        __file__,
        parameters={"LIMBS": limbs},
    )
