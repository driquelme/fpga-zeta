"""rs_z_unit: RTL vs golden model, bit-exact, two-phase protocol
(prep -> N check -> power sum injected -> Z)."""

import os

import cocotb
import mpmath as mp
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from zetafpga.golden import cexp, expln, rs_pipe, rs_z, theta
from zetafpga.golden import mpfloat as mf
from zetafpga.kernel.rs_setup import rs_entry

FMT = mf.Format(limbs=int(os.environ.get("ZETA_LIMBS", "1")))
TCFG = theta.load_cfg(FMT.width)
RCFG = rs_z.load_cfg(FMT.width)
PHW = FMT.width + 32
BW = PHW + 32


def _vectors() -> list[tuple[int, rs_z.ZPrep, mf.MPF, mf.MPF, int]]:
    """(t_fx, prep, s_re, s_im, expected_z_word) — the sum uses prep.n."""
    mp.mp.prec = 2 * TCFG.w2 + 80
    out = []
    for tv in [float(TCFG.t_min), 1234.5678, 1e7]:
        t_fx = int(mp.nint(mp.mpf(tv) * (1 << 32)))
        prep = rs_z.z_prep(t_fx, FMT)
        entries = [rs_entry(k, FMT, BW) for k in range(1, prep.n + 1)]
        s_re, s_im = rs_pipe.rs_power_sum(t_fx, entries, FMT, PHW, BW)
        z_word = mf.pack(rs_z.z_post(prep, s_re, s_im, FMT), FMT)
        out.append((t_fx, prep, s_re, s_im, z_word))
    return out


VECTORS = _vectors()


async def _clock(clk) -> None:
    while True:
        clk.value = 0
        await Timer(5, "ns")
        clk.value = 1
        await Timer(5, "ns")


@cocotb.test()
async def z_two_phase(dut) -> None:
    cocotb.start_soon(_clock(dut.clk))
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.sum_valid.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    timeout = TCFG.fg2 + TCFG.k * 40 + 5 * RCFG.nc * 60 + 4000
    for t_fx, prep, s_re, s_im, z_word in VECTORS:
        while dut.in_ready.value == 0:
            await RisingEdge(dut.clk)
        dut.in_valid.value = 1
        dut.t_fx.value = t_fx
        await RisingEdge(dut.clk)
        dut.in_valid.value = 0

        for _ in range(timeout):
            await FallingEdge(dut.clk)
            if dut.prep_valid.value == 1:
                break
        else:
            raise AssertionError(f"prep timeout for t_fx={t_fx:#x}")
        got_n = int(dut.n_out.value)
        assert got_n == prep.n, f"t_fx={t_fx:#x}: N={got_n}, expected {prep.n}"

        await RisingEdge(dut.clk)
        dut.sum_valid.value = 1
        dut.s_re.value = mf.pack(s_re, FMT)
        dut.s_im.value = mf.pack(s_im, FMT)
        await RisingEdge(dut.clk)
        dut.sum_valid.value = 0

        for _ in range(timeout):
            await FallingEdge(dut.clk)
            if dut.out_valid.value == 1:
                got = int(dut.z_o.value)
                assert got == z_word, f"t_fx={t_fx:#x}: got {got:#x}, expected {z_word:#x}"
                break
        else:
            raise AssertionError(f"post timeout for t_fx={t_fx:#x}")


def test_rs_z_unit(limbs: int) -> None:
    import pytest

    if limbs == 4:
        pytest.skip("Z256 runs in the nightly tier")
    from common.runner import run_block

    fmt = mf.Format(limbs=limbs)
    tcfg = theta.load_cfg(fmt.width)
    rcfg = rs_z.load_cfg(fmt.width)
    ecfg = expln.load_cfg(fmt.width)
    ecfg2 = expln.load_cfg(tcfg.w2)
    ccfg = cexp.load_cfg(fmt.width)
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
            "rtl/common/mp/mpf_recip.sv",
            "rtl/common/fn/log_mpf.sv",
            "rtl/common/fn/exp_mpf.sv",
            "rtl/common/fn/cexp_turns.sv",
            "rtl/common/fn/theta_turns.sv",
            "rtl/common/zeta/rs_z_unit.sv",
        ],
        "rs_z_unit",
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
            "KTERMS": tcfg.k,
            "LOG_TERMS": ecfg2.terms,
            "THETA_FX_ROM": f'"{tables / (tcfg.stem + "_fx.mem")}"',
            "THETA_MPF_ROM": f'"{tables / (tcfg.stem + "_mpf.mem")}"',
            "EXP_ROM2": f'"{tables / (ecfg2.stem + "_exp.mem")}"',
            "LN_ROM2": f'"{tables / (ecfg2.stem + "_ln.mem")}"',
            "NC": rcfg.nc,
            "KMAX": rcfg.kmax,
            "RSCK_ROM": f'"{tables / (rcfg.stem + ".mem")}"',
        },
    )
