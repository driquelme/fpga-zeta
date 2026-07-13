"""mpf_recip: RTL vs golden model, bit-exact (accuracy proven at golden level)."""

import os
import random

import cocotb
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from zetafpga.golden import mpfloat as mf
from zetafpga.golden import recip

FMT = mf.Format(limbs=int(os.environ.get("ZETA_LIMBS", "1")))
N_VECTORS = min(int(os.environ.get("ZETA_TEST_N", "10000")), 2000)


def _vectors() -> list[mf.MPF]:
    rng = random.Random(f"mpf_recip{FMT.limbs}")
    w = FMT.width
    out = [
        mf.MPF(0, 1, 1 << (w - 1)),  # 1.0 (exact)
        mf.MPF(0, 9, 1 << (w - 1)),  # power of two
        mf.MPF(1, -7, (1 << (w - 1)) + 1),
        mf.MPF(0, 0, (1 << w) - 1),
        mf.zero(0),
        mf.special(1),
    ]
    for _ in range(N_VECTORS):
        mant = (1 << (w - 1)) | rng.getrandbits(w - 1)
        out.append(mf.MPF(rng.getrandbits(1), rng.randrange(-100, 100), mant))
    return out


VECTORS = _vectors()


async def _clock(clk) -> None:
    while True:
        clk.value = 0
        await Timer(5, "ns")
        clk.value = 1
        await Timer(5, "ns")


@cocotb.test()
async def recip_vectors(dut) -> None:
    cocotb.start_soon(_clock(dut.clk))
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.x.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    for x in VECTORS:
        ev, eovf, eunf = recip.mpf_recip(x, FMT)
        expected = (mf.pack(ev, FMT), int(eovf), int(eunf))
        while dut.in_ready.value == 0:
            await RisingEdge(dut.clk)
        dut.in_valid.value = 1
        dut.x.value = mf.pack(x, FMT)
        await RisingEdge(dut.clk)
        dut.in_valid.value = 0
        for _ in range(200):
            await FallingEdge(dut.clk)
            if dut.out_valid.value == 1:
                got = (int(dut.result.value), int(dut.ovf.value), int(dut.unf.value))
                assert got == expected, f"x={x}: got {got}, expected {expected}"
                break
        else:
            raise AssertionError(f"timeout for x={x}")


def test_mpf_recip(limbs: int) -> None:
    from common.runner import run_block

    run_block(
        ["rtl/common/mp/mpf_recip.sv"],
        "mpf_recip",
        __file__,
        parameters={"LIMBS": limbs},
    )
