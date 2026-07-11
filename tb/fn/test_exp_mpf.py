"""exp_mpf: RTL vs golden model, bit-exact (accuracy proven at golden level)."""

import os
import random

import cocotb
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from zetafpga.golden import expln
from zetafpga.golden import mpfloat as mf

FMT = mf.Format(limbs=int(os.environ.get("ZETA_LIMBS", "2")))
CFG = expln.load_cfg(FMT.width)
N_RANDOM = int(os.environ.get("ZETA_TEST_N", "10000"))
N_OPS = min(N_RANDOM, 400)  # sequential unit: ~TERMS+6 cycles per op


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
    dut.fx_mode.value = 0
    dut.yfx_in.value = 0
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
    for _ in range(CFG.terms + 40):
        await FallingEdge(dut.clk)
        if dut.out_valid.value == 1:
            return int(dut.result.value), int(dut.ovf.value), int(dut.unf.value)
    raise AssertionError(f"timeout waiting for exp_mpf result, input {v}")


async def _check(dut, v: mf.MPF) -> None:
    got = await _op(dut, v)
    exp, eovf, eunf = expln.exp_mpf(v, FMT, CFG)
    expected = (mf.pack(exp, FMT), int(eovf), int(eunf))
    assert got == expected, f"y={v}: got {got}, expected {expected}"


@cocotb.test()
async def directed(dut) -> None:
    await _reset(dut)
    w = FMT.width
    for v in [
        mf.zero(0),
        mf.special(1),
        mf.MPF(0, 1, 1 << (w - 1)),  # e^1
        mf.MPF(1, 1, 1 << (w - 1)),  # e^-1
        mf.MPF(0, 22, 1 << (w - 1)),  # overflow saturate
        mf.MPF(1, 22, 1 << (w - 1)),  # underflow saturate
        mf.MPF(0, -w - 40, 1 << (w - 1)),  # tiny y -> 1.0
        mf.MPF(0, 19, (1 << w) - 1),  # near exponent limit
        mf.MPF(1, 19, (1 << w) - 1),
    ]:
        await _check(dut, v)


@cocotb.test()
async def random_vectors(dut) -> None:
    await _reset(dut)
    rng = random.Random(600 + FMT.limbs)
    for _ in range(N_OPS):
        mant = (1 << (FMT.width - 1)) | rng.getrandbits(FMT.width - 1)
        exp = rng.randrange(-FMT.width - 10, 20)
        await _check(dut, mf.MPF(rng.getrandbits(1), exp, mant))


def test_exp_mpf(limbs: int) -> None:
    from common.runner import run_block

    fmt = mf.Format(limbs=limbs)
    cfg = expln.load_cfg(fmt.width)
    run_block(
        "rtl/common/fn/exp_mpf.sv",
        "exp_mpf",
        __file__,
        parameters={
            "LIMBS": limbs,
            "FG": cfg.fg,
            "CONSTW": cfg.constw,
            "TERMS": cfg.terms,
            "CONSTS_ROM": f'"{expln.TABLES_DIR / (cfg.stem + "_exp.mem")}"',
        },
    )
