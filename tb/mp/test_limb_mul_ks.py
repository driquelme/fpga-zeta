"""limb_mul_ks (Karatsuba): equivalence with the exact product at LIMBS=4.

Schoolbook limb_mul is verified against Python int in test_limb_mul; verifying
Karatsuba against the same exact product establishes ks ≡ schoolbook.
"""

import os
import random

import cocotb
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from zetafpga.golden import limb

LIMBS = int(os.environ.get("ZETA_LIMBS", "4"))
WIDTH = LIMBS * 64
N_RANDOM = int(os.environ.get("ZETA_TEST_N", "10000"))


async def _clock(clk) -> None:
    while True:
        clk.value = 0
        await Timer(5, "ns")
        clk.value = 1
        await Timer(5, "ns")


async def _check_stream(dut, pairs: list[tuple[int, int]]) -> None:
    cocotb.start_soon(_clock(dut.clk))
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.a.value = 0
    dut.b.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    async def drive() -> None:
        for a, b in pairs:
            dut.in_valid.value = 1
            dut.a.value = a
            dut.b.value = b
            await RisingEdge(dut.clk)
        dut.in_valid.value = 0

    cocotb.start_soon(drive())
    got: list[int] = []
    while len(got) < len(pairs):
        await FallingEdge(dut.clk)
        if dut.out_valid.value == 1:
            got.append(int(dut.product.value))
    expected = [limb.mul(a, b, WIDTH) for a, b in pairs]
    mismatches = [i for i, (g, e) in enumerate(zip(got, expected, strict=True)) if g != e]
    assert not mismatches, f"karatsuba != exact product, first mismatch at {mismatches[0]}"


@cocotb.test()
async def directed(dut) -> None:
    m = limb.mask(WIDTH)
    h = limb.mask(WIDTH // 2)
    # Corners chosen to force the presum carries (sa, sb) on and off.
    pairs = [
        (0, 0),
        (1, 1),
        (m, m),
        (m, 1),
        (h, h),  # no presum carry
        (h << (WIDTH // 2), h),  # halves maximally asymmetric
        ((h << (WIDTH // 2)) | h, m),  # both presums carry
        (1 << (WIDTH - 1), 1 << (WIDTH - 1)),
    ]
    await _check_stream(dut, pairs)


@cocotb.test()
async def random_vectors(dut) -> None:
    rng = random.Random(40)
    pairs = [(rng.getrandbits(WIDTH), rng.getrandbits(WIDTH)) for _ in range(N_RANDOM)]
    await _check_stream(dut, pairs)


def test_limb_mul_ks() -> None:
    from common.runner import run_block

    run_block(
        [
            "rtl/arch/generic-sim/mul_tile.sv",
            "rtl/common/mp/limb_mul.sv",
            "rtl/common/mp/limb_mul_ks.sv",
        ],
        "limb_mul_ks",
        __file__,
        parameters={"LIMBS": 4},
    )
