"""rs_power_sum: bit-exact vs golden rs_pipe, II=1, throughput recorded."""

import os

import cocotb
import mpmath as mp
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from zetafpga.golden import cexp, expln, rs_pipe
from zetafpga.golden import mpfloat as mf
from zetafpga.kernel.rs_setup import rs_entry

FMT = mf.Format(limbs=int(os.environ.get("ZETA_LIMBS", "1")))
ECFG = expln.load_cfg(FMT.width)
CCFG = cexp.load_cfg(FMT.width)
PHW = FMT.width + 32
BW = PHW + 32


def _vectors() -> list[tuple[int, list[tuple[int, mf.MPF]]]]:
    mp.mp.prec = 2 * FMT.width + 80
    out = []
    for tv, n in [(5000.0, 28), (100_000.0, 126), (50.0, 2)]:
        t_fx = int(mp.nint(mp.mpf(tv) * (1 << 32)))
        out.append((t_fx, [rs_entry(k, FMT, BW) for k in range(1, n + 1)]))
    return out


VECTORS = _vectors()


async def _clock(clk) -> None:
    while True:
        clk.value = 0
        await Timer(5, "ns")
        clk.value = 1
        await Timer(5, "ns")


@cocotb.test()
async def sums_bit_exact_and_ii1(dut) -> None:
    cocotb.start_soon(_clock(dut.clk))
    dut.rst_n.value = 0
    dut.start_valid.value = 0
    dut.entry_valid.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    for t_fx, entries in VECTORS:
        exp_re, exp_im = rs_pipe.rs_power_sum(t_fx, entries, FMT, PHW, BW)

        while dut.start_ready.value == 0:
            await RisingEdge(dut.clk)
        dut.start_valid.value = 1
        dut.t_fx.value = t_fx
        dut.n_in.value = len(entries)
        await RisingEdge(dut.clk)
        dut.start_valid.value = 0

        async def feed(entries=entries) -> None:
            for lnn2pi, amp in entries:
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
        cycles = 0
        limit = len(entries) + CCFG.terms + 200
        while True:
            await FallingEdge(dut.clk)
            cycles += 1
            assert cycles < limit, f"timeout: N={len(entries)} took >{limit} cycles"
            if dut.out_valid.value == 1:
                got = (int(dut.s_re.value), int(dut.s_im.value))
                expected = (mf.pack(exp_re, FMT), mf.pack(exp_im, FMT))
                assert got == expected, (
                    f"t_fx={t_fx:#x} N={len(entries)}: got {got}, expected {expected}"
                )
                # II=1: total cycles ~ N + pipeline drain, far below 2N
                assert cycles <= len(entries) + CCFG.terms + 60, (
                    f"N={len(entries)}: {cycles} cycles — not II=1"
                )
                cocotb.log.info(f"N={len(entries)}: {cycles} cycles")
                break


def test_rs_power_sum(limbs: int) -> None:
    import pytest

    if limbs == 4:
        pytest.skip("Z256 runs in the nightly tier")
    from common.runner import run_block

    fmt = mf.Format(limbs=limbs)
    ecfg = expln.load_cfg(fmt.width)
    ccfg = cexp.load_cfg(fmt.width)
    run_block(
        [
            "rtl/common/prim/lzc.sv",
            "rtl/arch/generic-sim/mul_tile.sv",
            "rtl/common/mp/limb_mul.sv",
            "rtl/common/mp/mpf_mul.sv",
            "rtl/common/fx/fx_mul_mod1.sv",
            "rtl/common/fn/cexp_pipe.sv",
            "rtl/common/zeta/rs_power_sum.sv",
        ],
        "rs_power_sum",
        __file__,
        parameters={
            "LIMBS": limbs,
            "PHW": fmt.width + 32,
            "FG": ecfg.fg,
            "CONSTW": ecfg.constw,
            "CTERMS": ccfg.terms,
            "EXP_TERMS": ecfg.terms,
            "CEXP_ROM": f'"{cexp.TABLES_DIR / (ccfg.stem + ".mem")}"',
            "EXP_ROM": f'"{expln.TABLES_DIR / (ecfg.stem + "_exp.mem")}"',
        },
    )
