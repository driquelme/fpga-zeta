"""mpf_add: RTL vs golden model (bus-exact, including ovf/unf flags)."""

import os
import random

import cocotb
from cocotb.triggers import FallingEdge, RisingEdge, Timer
from common.mpf_stim import cancellation_pair, random_mpf

from zetafpga.golden import mpfloat as mf

FMT = mf.Format(limbs=int(os.environ.get("ZETA_LIMBS", "2")))
N_RANDOM = int(os.environ.get("ZETA_TEST_N", "10000"))


async def _clock(clk) -> None:
    while True:
        clk.value = 0
        await Timer(5, "ns")
        clk.value = 1
        await Timer(5, "ns")


async def _run(dut, pairs: list[tuple[mf.MPF, mf.MPF]]) -> None:
    cocotb.start_soon(_clock(dut.clk))
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.x.value = 0
    dut.y.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    async def drive() -> None:
        for x, y in pairs:
            dut.in_valid.value = 1
            dut.x.value = mf.pack(x, FMT)
            dut.y.value = mf.pack(y, FMT)
            await RisingEdge(dut.clk)
        dut.in_valid.value = 0

    cocotb.start_soon(drive())
    got: list[tuple[int, int, int]] = []
    while len(got) < len(pairs):
        await FallingEdge(dut.clk)
        if dut.out_valid.value == 1:
            got.append((int(dut.result.value), int(dut.ovf.value), int(dut.unf.value)))

    for i, ((x, y), (rw, rovf, runf)) in enumerate(zip(pairs, got, strict=True)):
        exp, eovf, eunf = mf.mpf_add(x, y, FMT)
        ew = mf.pack(exp, FMT)
        assert (rw, rovf, runf) == (ew, int(eovf), int(eunf)), (
            f"vector {i}: x={x}, y={y}: got ({rw:#x},{rovf},{runf}), "
            f"expected ({ew:#x},{int(eovf)},{int(eunf)})"
        )


@cocotb.test()
async def directed(dut) -> None:
    w = FMT.width
    one = mf.MPF(0, 1, 1 << (w - 1))
    m_one = mf.MPF(1, 1, 1 << (w - 1))
    tie = mf.MPF(0, 1 - w, 1 << (w - 1))  # ulp/2 of one
    pairs = [
        (one, one),
        (one, m_one),  # exact cancellation -> +0
        (one, tie),  # tie to even
        (mf.zero(0), one),
        (one, mf.zero(1)),
        (mf.zero(1), mf.zero(1)),
        (mf.zero(0), mf.zero(1)),
        (mf.special(1), one),
        (one, mf.special(0)),
        # far-apart exponents: sticky-only alignment
        (one, mf.MPF(0, 1 - w - 5, (1 << (w - 1)) | 1)),
        (one, mf.MPF(1, 1 - w - 5, (1 << (w - 1)) | 1)),
        # max exponent + round-up -> overflow
        (mf.MPF(0, FMT.emax, (1 << w) - 1), mf.MPF(0, FMT.emax - w, 1 << (w - 1))),
    ]
    await _run(dut, pairs)


@cocotb.test()
async def random_vectors(dut) -> None:
    rng = random.Random(400 + FMT.limbs)
    pairs = [(random_mpf(FMT, rng), random_mpf(FMT, rng)) for _ in range(N_RANDOM)]
    await _run(dut, pairs)


@cocotb.test()
async def random_cancellation(dut) -> None:
    rng = random.Random(500 + FMT.limbs)
    pairs = [cancellation_pair(FMT, rng) for _ in range(N_RANDOM // 2)]
    await _run(dut, pairs)


def test_mpf_add(limbs: int) -> None:
    from common.runner import run_block

    run_block(
        [
            "rtl/common/prim/lzc.sv",
            "rtl/common/mp/limb_shift.sv",
            "rtl/common/mp/limb_addsub.sv",
            "rtl/common/mp/mpf_add.sv",
        ],
        "mpf_add",
        __file__,
        parameters={"LIMBS": limbs},
    )
