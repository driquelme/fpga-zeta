"""log_mpf: RTL vs golden model, bit-exact (accuracy proven at golden level)."""

import os
import random

import cocotb
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from zetafpga.golden import expln
from zetafpga.golden import mpfloat as mf

FMT = mf.Format(limbs=int(os.environ.get("ZETA_LIMBS", "2")))
CFG = expln.load_cfg(FMT.width)
N_RANDOM = int(os.environ.get("ZETA_TEST_N", "10000"))
N_OPS = min(N_RANDOM, 150)  # sequential unit: ~FG+5 cycles per op


async def _clock(clk) -> None:
    while True:
        clk.value = 0
        await Timer(5, "ns")
        clk.value = 1
        await Timer(5, "ns")


async def _reset(dut) -> None:
    cocotb.start_soon(_clock(dut.clk))
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.x.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def _op(dut, v: mf.MPF) -> tuple[int, int, int]:
    while dut.in_ready.value == 0:
        await RisingEdge(dut.clk)
    dut.in_valid.value = 1
    dut.x.value = mf.pack(v, FMT)
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    for _ in range(CFG.fg + 40):
        await FallingEdge(dut.clk)
        if dut.out_valid.value == 1:
            return int(dut.result.value), int(dut.ovf.value), int(dut.unf.value)
    raise AssertionError(f"timeout waiting for log_mpf result, input {v}")


async def _check(dut, v: mf.MPF) -> None:
    got = await _op(dut, v)
    exp, eovf, eunf = expln.log_mpf(v, FMT, CFG)
    expected = (mf.pack(exp, FMT), int(eovf), int(eunf))
    assert got == expected, f"x={v}: got {got}, expected {expected}"


@cocotb.test()
async def directed(dut) -> None:
    await _reset(dut)
    w = FMT.width
    one = mf.MPF(0, 1, 1 << (w - 1))
    for v in [
        one,  # ln(1) = 0 exactly
        mf.MPF(0, 2, 1 << (w - 1)),  # ln(2)
        mf.MPF(0, 1, (1 << w) - 1),  # just below 2
        mf.MPF(0, 1, (1 << (w - 1)) | 1),  # just above 1 (near-1 band)
        mf.zero(0),  # domain error
        mf.MPF(1, 1, 1 << (w - 1)),  # negative: domain error
        mf.special(0),
        mf.MPF(0, 1 << 18, 1 << (w - 1)),  # huge x
        mf.MPF(0, -(1 << 18), 1 << (w - 1)),  # tiny x
    ]:
        await _check(dut, v)


@cocotb.test()
async def random_vectors(dut) -> None:
    await _reset(dut)
    rng = random.Random(700 + FMT.limbs)
    for _ in range(N_OPS):
        mant = (1 << (FMT.width - 1)) | rng.getrandbits(FMT.width - 1)
        exp = rng.randrange(-(1 << 18), 1 << 18)
        await _check(dut, mf.MPF(0, exp, mant))


def test_log_mpf(limbs: int) -> None:
    from common.runner import run_block

    fmt = mf.Format(limbs=limbs)
    cfg = expln.load_cfg(fmt.width)
    run_block(
        ["rtl/common/prim/lzc.sv", "rtl/common/fn/log_mpf.sv"],
        "log_mpf",
        __file__,
        parameters={
            "LIMBS": limbs,
            "FG": cfg.fg,
            "CONSTW": cfg.constw,
            "CONST_LINES": cfg.terms + 2,
            "CONSTS_ROM": f'"{expln.TABLES_DIR / (cfg.stem + "_exp.mem")}"',
            "LN_ROM": f'"{expln.TABLES_DIR / (cfg.stem + "_ln.mem")}"',
        },
    )
