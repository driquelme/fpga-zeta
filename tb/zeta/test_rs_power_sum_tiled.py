"""rs_power_sum_tiled: bit-exact vs the (unchanged) golden model at LANES
parallel lanes, and the lane speedup measured."""

import os

import cocotb
import mpmath as mp
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from zetafpga.golden import cexp, expln, rs_pipe
from zetafpga.golden import mpfloat as mf
from zetafpga.kernel.rs_setup import rs_entry

FMT = mf.Format(limbs=int(os.environ.get("ZETA_LIMBS", "1")))
LANES = int(os.environ.get("ZETA_RS_LANES", "4"))
CCFG = cexp.load_cfg(FMT.width)
PHW = FMT.width + 32
BW = PHW + 32


def _vectors() -> list[tuple[int, list[tuple[int, mf.MPF]]]]:
    mp.mp.prec = 2 * FMT.width + 80
    out = []
    # N chosen around lane-count corners: multiples, off-by-one, N < LANES
    for tv, n in [(100_000.0, 126), (5000.0, 28), (300.0, LANES + 1), (50.0, 2)]:
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
async def tiled_bit_exact_and_fast(dut) -> None:
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
            # stripe: lane l gets entries l, l+LANES, ...
            queues = [entries[lane::LANES] for lane in range(LANES)]
            idx = [0] * LANES
            while any(idx[lane] < len(queues[lane]) for lane in range(LANES)):
                vmask = 0
                lnn_flat = 0
                amp_flat = 0
                for lane in range(LANES):
                    if idx[lane] < len(queues[lane]):
                        lnn2pi, amp = queues[lane][idx[lane]]
                        vmask |= 1 << lane
                        lnn_flat |= lnn2pi << (lane * (BW + 8))
                        amp_flat |= mf.pack(amp, FMT) << (lane * FMT.mpw)
                dut.entry_valid.value = vmask
                dut.lnn2pi.value = lnn_flat
                dut.amp.value = amp_flat
                await FallingEdge(dut.clk)
                ready = int(dut.entry_ready.value)
                await RisingEdge(dut.clk)
                for lane in range(LANES):
                    if (vmask & (1 << lane)) and (ready & (1 << lane)):
                        idx[lane] += 1
            dut.entry_valid.value = 0

        cocotb.start_soon(feed())
        n = len(entries)
        cycles = 0
        limit = (n + LANES - 1) // LANES + CCFG.terms + 200
        while True:
            await FallingEdge(dut.clk)
            cycles += 1
            assert cycles < limit, f"timeout: N={n} took >{limit} cycles"
            if dut.out_valid.value == 1:
                got = (int(dut.s_re.value), int(dut.s_im.value))
                expected = (mf.pack(exp_re, FMT), mf.pack(exp_im, FMT))
                assert got == expected, f"t_fx={t_fx:#x} N={n}: got {got}, expected {expected}"
                assert cycles <= (n + LANES - 1) // LANES + CCFG.terms + 80, (
                    f"N={n}, LANES={LANES}: {cycles} cycles — lanes not parallel"
                )
                cocotb.log.info(f"N={n} LANES={LANES}: {cycles} cycles")
                break


def test_rs_power_sum_tiled(limbs: int) -> None:
    import pytest

    if limbs >= 3:
        pytest.skip("tiled sweep runs Z64/Z128; wide configs in the nightly tier")
    from common.runner import run_block

    fmt = mf.Format(limbs=limbs)
    ecfg = expln.load_cfg(fmt.width)
    ccfg = cexp.load_cfg(fmt.width)
    lanes = 4 if limbs == 1 else 2
    os.environ["ZETA_RS_LANES"] = str(lanes)
    run_block(
        [
            "rtl/common/prim/lzc.sv",
            "rtl/arch/generic-sim/mul_tile.sv",
            "rtl/common/mp/limb_mul.sv",
            "rtl/common/mp/mpf_mul.sv",
            "rtl/common/fx/fx_mul_mod1.sv",
            "rtl/common/fn/cexp_pipe.sv",
            "rtl/common/zeta/rs_power_sum.sv",
            "rtl/common/zeta/rs_power_sum_tiled.sv",
        ],
        "rs_power_sum_tiled",
        __file__,
        parameters={
            "LANES": lanes,
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
