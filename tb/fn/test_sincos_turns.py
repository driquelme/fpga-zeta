"""sincos_turns: RTL vs golden model, bit-exact (accuracy is proven at the
golden level in tests/test_golden_fn.py)."""

import os
import random

import cocotb
from cocotb.triggers import Timer

from zetafpga.golden import fn

PHW = int(os.environ.get("ZETA_PHW", "96"))
CFG = fn.load_cfg()
N_RANDOM = int(os.environ.get("ZETA_TEST_N", "10000"))


def _signed(v: int, width: int) -> int:
    return v - (1 << width) if v >> (width - 1) else v


async def _check(dut, phase: int) -> None:
    dut.phase.value = phase
    await Timer(1, "ns")
    exp_s, exp_c = fn.sincos_turns(phase, PHW, CFG)
    got_s = _signed(int(dut.sin_o.value), CFG.ow)
    got_c = _signed(int(dut.cos_o.value), CFG.ow)
    assert (got_s, got_c) == (exp_s, exp_c), (
        f"phase={phase:#x}: got ({got_s}, {got_c}), expected ({exp_s}, {exp_c})"
    )


@cocotb.test()
async def boundaries(dut) -> None:
    qw = PHW - 2
    for base in [0, 1 << qw, 2 << qw, 3 << qw]:
        for off in [0, 1, (1 << qw) - 1]:
            await _check(dut, (base + off) % (1 << PHW))
    seg_step = 1 << (qw - CFG.segw)
    for seg in [0, 1, (1 << CFG.segw) - 1]:
        for off in [0, 1, seg_step - 1]:
            await _check(dut, seg * seg_step + off)
    await _check(dut, (1 << PHW) - 1)


@cocotb.test()
async def random_phases(dut) -> None:
    rng = random.Random(4321)
    for _ in range(N_RANDOM):
        await _check(dut, rng.getrandbits(PHW))


def test_sincos_turns() -> None:
    from common.runner import run_block

    tables = fn.TABLES_DIR
    run_block(
        "rtl/common/fn/sincos_turns.sv",
        "sincos_turns",
        __file__,
        parameters={
            "PHW": PHW,
            "SIN_ROM": f'"{tables / (CFG.stem + "_sin.mem")}"',
            "COS_ROM": f'"{tables / (CFG.stem + "_cos.mem")}"',
        },
    )
