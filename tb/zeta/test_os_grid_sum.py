"""os_grid_sum: RTL vs golden model, bit-exact (accuracy proven at golden level)."""

import os

import cocotb
import mpmath as mp
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from zetafpga.golden import cexp, expln, fft, os_pipe
from zetafpga.golden import mpfloat as mf
from zetafpga.kernel.rs_setup import rs_entry, rs_n

FMT = mf.Format(limbs=int(os.environ.get("ZETA_LIMBS", "1")))
PHW = FMT.width + 32
BW = PHW + 32
M = 128
COUNT = 16


def _vector() -> tuple[int, int, int, list[tuple[int, mf.MPF]], list[tuple[int, int]]]:
    mp.mp.prec = 2 * FMT.width + 80
    t0_fx = int(mp.nint(mp.mpf(5000.0) * (1 << 32)))
    dt_fx = int(mp.nint(mp.mpf(0.25) * (1 << 32)))
    n = rs_n(t0_fx + (COUNT - 1) * dt_fx)
    assert n == rs_n(t0_fx)
    entries = [rs_entry(k, FMT, BW) for k in range(1, n + 1)]
    expected = os_pipe.os_grid_sum(t0_fx, dt_fx, n, COUNT, entries, FMT, M)
    exp_words = [
        (mf.pack(os_pipe.fx54_to_mpf(re, FMT), FMT), mf.pack(os_pipe.fx54_to_mpf(im, FMT), FMT))
        for re, im in expected
    ]
    return t0_fx, dt_fx, n, entries, exp_words


T0_FX, DT_FX, N, ENTRIES, EXPECTED = _vector()


async def _clock(clk) -> None:
    while True:
        clk.value = 0
        await Timer(5, "ns")
        clk.value = 1
        await Timer(5, "ns")


@cocotb.test()
async def grid_bit_exact(dut) -> None:
    cocotb.start_soon(_clock(dut.clk))
    dut.rst_n.value = 0
    dut.start_valid.value = 0
    dut.entry_valid.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    while dut.start_ready.value == 0:
        await RisingEdge(dut.clk)
    dut.start_valid.value = 1
    dut.t0_fx.value = T0_FX
    dut.dt_fx.value = DT_FX
    dut.n_in.value = N
    dut.j_in.value = COUNT
    await RisingEdge(dut.clk)
    dut.start_valid.value = 0

    async def feed() -> None:
        for lnn2pi, amp in ENTRIES:
            dut.entry_valid.value = 1
            dut.lnn2pi.value = lnn2pi
            dut.amp.value = mf.pack(amp, FMT)
            while True:
                await FallingEdge(dut.clk)
                if dut.entry_ready.value == 1:
                    await RisingEdge(dut.clk)
                    break
        dut.entry_valid.value = 0

    cocotb.start_soon(feed())

    got: list[tuple[int, int]] = []
    limit = 15 * M + N * 120 + 15 * (M * 3 + M // 2 * (M.bit_length() + 2) * 4) + COUNT * 60 + 2000
    for _ in range(limit):
        await FallingEdge(dut.clk)
        if dut.point_valid.value == 1:
            got.append((int(dut.s_re.value), int(dut.s_im.value)))
        if dut.done.value == 1:
            break
    else:
        raise AssertionError(f"timeout: {len(got)}/{COUNT} points")
    assert len(got) == COUNT
    assert got == EXPECTED, "RTL grid sum != golden"


def test_os_grid_sum(limbs: int) -> None:
    import pytest

    if limbs >= 3:
        pytest.skip("OS sweep runs Z64/Z128; wide configs in the nightly tier")
    from common.runner import run_block

    fmt = mf.Format(limbs=limbs)
    ecfg = expln.load_cfg(fmt.width)
    ccfg = cexp.load_cfg(fmt.width)
    tables = fft.TABLES_DIR
    run_block(
        [
            "rtl/common/prim/lzc.sv",
            "rtl/arch/generic-sim/mul_tile.sv",
            "rtl/common/mp/limb_mul.sv",
            "rtl/common/mp/mpf_mul.sv",
            "rtl/common/fx/fx_mul_mod1.sv",
            "rtl/common/fn/cexp_turns.sv",
            "rtl/common/fn/fft_radix2.sv",
            "rtl/common/zeta/rs_power_sum.sv",
            "rtl/common/zeta/os_grid_sum.sv",
        ],
        "os_grid_sum",
        __file__,
        parameters={
            "LIMBS": limbs,
            "PHW": fmt.width + 32,
            "FG": ecfg.fg,
            "CONSTW": ecfg.constw,
            "CTERMS": ccfg.terms,
            "EXP_TERMS": ecfg.terms,
            "CEXP_ROM": f'"{tables / (ccfg.stem + ".mem")}"',
            "EXP_ROM": f'"{tables / (ecfg.stem + "_exp.mem")}"',
            "M": M,
            "FFT_ROM": f'"{tables / f"fft_m{M}.mem"}"',
            "OS_ROM": f'"{tables / "fft_os.mem"}"',
        },
    )
