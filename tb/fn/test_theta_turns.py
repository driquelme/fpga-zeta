"""theta_turns: RTL vs golden model, bit-exact (accuracy proven at golden level)."""

import os

import cocotb
import mpmath as mp
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from zetafpga.golden import expln, theta
from zetafpga.golden import mpfloat as mf
from zetafpga.kernel.em_setup import mpf_from_real

FMT = mf.Format(limbs=int(os.environ.get("ZETA_LIMBS", "1")))
CFG = theta.load_cfg(FMT.width)
FMT2 = mf.Format(limbs=CFG.w2 // 64)
ECFG2 = expln.load_cfg(CFG.w2)
PHW = FMT.width + 32


def _vectors() -> list[tuple[int, mf.MPF]]:
    mp.mp.prec = 2 * CFG.w2 + 80
    out = []
    for tv in [float(CFG.t_min), 50.0, 100.0, 1234.5678, 1e5, 1e7, 4e9]:
        t_fx = int(mp.nint(mp.mpf(tv) * (1 << 32)))
        out.append((t_fx, mpf_from_real(mp.mpf(2) ** 32 / t_fx, FMT2)))
    return out


VECTORS = _vectors()


async def _clock(clk) -> None:
    while True:
        clk.value = 0
        await Timer(5, "ns")
        clk.value = 1
        await Timer(5, "ns")


@cocotb.test()
async def theta_vectors(dut) -> None:
    cocotb.start_soon(_clock(dut.clk))
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.t_fx.value = 0
    dut.inv_t.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    timeout = CFG.fg2 + CFG.k * 40 + 200
    for t_fx, inv_t in VECTORS:
        expected, exp_lnu = theta.theta_turns(t_fx, inv_t, FMT, PHW, CFG)
        while dut.in_ready.value == 0:
            await RisingEdge(dut.clk)
        dut.in_valid.value = 1
        dut.t_fx.value = t_fx
        dut.inv_t.value = mf.pack(inv_t, FMT2)
        await RisingEdge(dut.clk)
        dut.in_valid.value = 0
        for _ in range(timeout):
            await FallingEdge(dut.clk)
            if dut.out_valid.value == 1:
                got = int(dut.theta_o.value)
                assert got == expected, f"t_fx={t_fx:#x}: got {got:#x}, expected {expected:#x}"
                got_lnu = int(dut.lnu_o.value)
                assert got_lnu == mf.pack(exp_lnu, FMT2), f"t_fx={t_fx:#x}: lnu mismatch"
                break
        else:
            raise AssertionError(f"timeout for t_fx={t_fx:#x}")


def test_theta_turns(limbs: int) -> None:
    import pytest

    if limbs == 4:
        pytest.skip("Z256 theta runs in the nightly tier")
    from common.runner import run_block

    fmt = mf.Format(limbs=limbs)
    cfg = theta.load_cfg(fmt.width)
    ecfg2 = expln.load_cfg(cfg.w2)
    tables = theta.TABLES_DIR
    run_block(
        [
            "rtl/common/prim/lzc.sv",
            "rtl/arch/generic-sim/mul_tile.sv",
            "rtl/common/mp/limb_mul.sv",
            "rtl/common/mp/limb_shift.sv",
            "rtl/common/mp/limb_addsub.sv",
            "rtl/common/mp/mpf_mul.sv",
            "rtl/common/mp/mpf_add.sv",
            "rtl/common/fn/log_mpf.sv",
            "rtl/common/fn/theta_turns.sv",
        ],
        "theta_turns",
        __file__,
        parameters={
            "LIMBS": limbs,
            "PHW": fmt.width + 32,
            "KTERMS": cfg.k,
            "LOG_TERMS": ecfg2.terms,
            "THETA_FX_ROM": f'"{tables / (cfg.stem + "_fx.mem")}"',
            "THETA_MPF_ROM": f'"{tables / (cfg.stem + "_mpf.mem")}"',
            "EXP_ROM2": f'"{tables / (ecfg2.stem + "_exp.mem")}"',
            "LN_ROM2": f'"{tables / (ecfg2.stem + "_ln.mem")}"',
        },
    )
