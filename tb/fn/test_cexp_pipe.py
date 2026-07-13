"""cexp_pipe: bit-exact vs the golden cexp model, at full rate (II=1)."""

import os
import random

import cocotb
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from zetafpga.golden import cexp
from zetafpga.golden import mpfloat as mf

FMT = mf.Format(limbs=int(os.environ.get("ZETA_LIMBS", "1")))
CCFG = cexp.load_cfg(FMT.width)
PHW = FMT.width + 32
N_RANDOM = int(os.environ.get("ZETA_TEST_N", "10000"))


async def _clock(clk) -> None:
    while True:
        clk.value = 0
        await Timer(5, "ns")
        clk.value = 1
        await Timer(5, "ns")


async def _run(dut, phis: list[int], p_valid: float, seed: int) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    cocotb.start_soon(_clock(dut.clk))
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.phi.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    async def drive() -> None:
        for p in phis:
            while rng.random() > p_valid:
                dut.in_valid.value = 0
                await RisingEdge(dut.clk)
            dut.in_valid.value = 1
            dut.phi.value = p
            await RisingEdge(dut.clk)
        dut.in_valid.value = 0

    cocotb.start_soon(drive())
    got: list[tuple[int, int, int]] = []
    cycle = 0
    while len(got) < len(phis):
        await FallingEdge(dut.clk)
        cycle += 1
        if dut.out_valid.value == 1:
            got.append((cycle, int(dut.cos_o.value), int(dut.sin_o.value)))

    for phi, (_, gc, gs) in zip(phis, got, strict=True):
        ec, es = cexp.cexp_turns(phi, PHW, FMT, CCFG)
        expected = (mf.pack(ec, FMT), mf.pack(es, FMT))
        assert (gc, gs) == expected, f"phi={phi:#x}: got {(gc, gs)}, expected {expected}"
    return got


@cocotb.test()
async def full_rate_ii1(dut) -> None:
    rng = random.Random(11)
    phis = [rng.getrandbits(PHW) for _ in range(300)]
    got = await _run(dut, phis, p_valid=1.0, seed=11)
    cycles = [c for c, _, _ in got]
    assert cycles == list(range(cycles[0], cycles[0] + len(phis))), (
        "out_valid not contiguous at full rate — pipeline is not II=1"
    )


@cocotb.test()
async def gapped_random(dut) -> None:
    rng = random.Random(12)
    phis = [rng.getrandbits(PHW) for _ in range(min(N_RANDOM, 2000))]
    phis += [0, 1, (1 << PHW) - 1, 1 << (PHW - 1)]
    await _run(dut, phis, p_valid=0.7, seed=12)


def test_cexp_pipe(limbs: int) -> None:
    import pytest

    if limbs == 4:
        pytest.skip("Z256 pipelined cexp runs in the nightly tier")
    from common.runner import run_block

    from zetafpga.golden import expln

    fmt = mf.Format(limbs=limbs)
    ecfg = expln.load_cfg(fmt.width)
    ccfg = cexp.load_cfg(fmt.width)
    run_block(
        ["rtl/common/prim/lzc.sv", "rtl/common/fn/cexp_pipe.sv"],
        "cexp_pipe",
        __file__,
        parameters={
            "LIMBS": limbs,
            "PHW": fmt.width + 32,
            "FG": ecfg.fg,
            "CONSTW": ecfg.constw,
            "TERMS": ccfg.terms,
            "CONST_LINES": ecfg.terms + 2,
            "CEXP_ROM": f'"{cexp.TABLES_DIR / (ccfg.stem + ".mem")}"',
            "EXP_ROM": f'"{expln.TABLES_DIR / (ecfg.stem + "_exp.mem")}"',
        },
    )
