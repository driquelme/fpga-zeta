"""euler_maclaurin_top: RTL vs golden zeta_em, bit-exact end to end.

Accuracy vs mpmath is proven at the golden level (tests/test_golden_zeta_em.py);
this test proves the RTL executes the identical computation.
"""

import os

import cocotb
import mpmath as mp
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from zetafpga.golden import cexp, expln, zeta_em
from zetafpga.golden import mpfloat as mf
from zetafpga.kernel.em_setup import EmProgram, build_program, mpf_from_real

FMT = mf.Format(limbs=int(os.environ.get("ZETA_LIMBS", "1")))
ECFG = expln.load_cfg(FMT.width)
CCFG = cexp.load_cfg(FMT.width)

# Vectors kept small-N (sequential engine): s = 2, first zero, negative sigma.
VECTOR_SPECS = [
    (2.0, 0.0),
    (0.5, 14.134725141734695),
    (-3.0, 10.0),
]


def _programs() -> list[EmProgram]:
    mp.mp.prec = 2 * FMT.width + 80
    progs = []
    for sv, tv in VECTOR_SPECS:
        sigma = mf.zero(0) if sv == 0.0 else mpf_from_real(mp.mpf(sv), FMT)
        t_fx = int(mp.nint(mp.mpf(tv) * (1 << 32)))
        progs.append(build_program(sigma, t_fx, FMT))
    return progs


PROGS = _programs()


async def _clock(clk) -> None:
    while True:
        clk.value = 0
        await Timer(5, "ns")
        clk.value = 1
        await Timer(5, "ns")


async def _beat(dut, ready_name: str) -> None:
    """Complete exactly one valid/ready beat (data already driven)."""
    while True:
        await FallingEdge(dut.clk)
        if getattr(dut, ready_name).value == 1:
            await RisingEdge(dut.clk)
            return


async def _feed_entries(dut, prog: EmProgram) -> None:
    entries = [*list(prog.entries), prog.entries[-1]]
    for lnn_fx, lnn2pi in entries:
        dut.entry_valid.value = 1
        dut.entry_lnn_fx.value = lnn_fx
        dut.entry_lnn2pi.value = lnn2pi
        await _beat(dut, "entry_ready")
    dut.entry_valid.value = 0


async def _feed_bern(dut, prog: EmProgram) -> None:
    for b in prog.bern:
        dut.bern_valid.value = 1
        dut.bern_data.value = mf.pack(b, FMT)
        await _beat(dut, "bern_ready")
    dut.bern_valid.value = 0


@cocotb.test()
async def zeta_vectors(dut) -> None:
    cocotb.start_soon(_clock(dut.clk))
    dut.rst_n.value = 0
    for sig in ("start_valid", "entry_valid", "bern_valid"):
        getattr(dut, sig).value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    for prog in PROGS:
        exp_re, exp_im, exp_ovf, exp_unf = zeta_em.zeta_em(prog)

        while dut.start_ready.value == 0:
            await RisingEdge(dut.clk)
        dut.start_valid.value = 1
        dut.ps_only.value = int(prog.ps_only)
        dut.sigma.value = mf.pack(prog.sigma, FMT)
        dut.t_mpf.value = mf.pack(prog.t_mpf, FMT)
        dut.t_fx.value = prog.t_fx
        dut.n_in.value = prog.n
        dut.m_in.value = prog.m
        dut.inv_sm1_re.value = mf.pack(prog.inv_sm1_re, FMT)
        dut.inv_sm1_im.value = mf.pack(prog.inv_sm1_im, FMT)
        dut.inv_np2.value = mf.pack(prog.inv_np2, FMT)
        await RisingEdge(dut.clk)
        dut.start_valid.value = 0

        cocotb.start_soon(_feed_entries(dut, prog))
        cocotb.start_soon(_feed_bern(dut, prog))

        # Generous timeout: N npow evaluations + M tail iterations.
        limit = (prog.n + 4) * (ECFG.fg + ECFG.terms + CCFG.terms + 120) + prog.m * 400
        for _ in range(limit):
            await FallingEdge(dut.clk)
            if dut.out_valid.value == 1:
                got = (
                    int(dut.z_re.value),
                    int(dut.z_im.value),
                    int(dut.ovf.value),
                    int(dut.unf.value),
                )
                expected = (
                    mf.pack(exp_re, FMT),
                    mf.pack(exp_im, FMT),
                    int(exp_ovf),
                    int(exp_unf),
                )
                assert got == expected, f"N={prog.n} M={prog.m}: got {got}, expected {expected}"
                break
        else:
            raise AssertionError(f"timeout after {limit} cycles (N={prog.n}, M={prog.m})")


def test_euler_maclaurin(limbs: int) -> None:
    if limbs == 4:
        import pytest

        pytest.skip("Z256 end-to-end runs in the nightly tier")
    from common.runner import run_block

    fmt = mf.Format(limbs=limbs)
    ecfg = expln.load_cfg(fmt.width)
    ccfg = cexp.load_cfg(fmt.width)
    run_block(
        [
            "rtl/common/prim/lzc.sv",
            "rtl/common/prim/skid_buffer.sv",
            "rtl/arch/generic-sim/mul_tile.sv",
            "rtl/common/mp/limb_mul.sv",
            "rtl/common/mp/limb_shift.sv",
            "rtl/common/mp/limb_addsub.sv",
            "rtl/common/mp/mpf_mul.sv",
            "rtl/common/mp/mpf_add.sv",
            "rtl/common/fx/fx_mul_mod1.sv",
            "rtl/common/fn/exp_mpf.sv",
            "rtl/common/fn/cexp_turns.sv",
            "rtl/common/zeta/npow_s_kernel.sv",
            "rtl/common/zeta/euler_maclaurin_top.sv",
        ],
        "euler_maclaurin_top",
        __file__,
        parameters={
            "LIMBS": limbs,
            "PHW": fmt.width + 32,
            "FG": ecfg.fg,
            "CONSTW": ecfg.constw,
            "TERMS": ecfg.terms,
            "CTERMS": ccfg.terms,
            "EXP_ROM": f'"{expln.TABLES_DIR / (ecfg.stem + "_exp.mem")}"',
            "CEXP_ROM": f'"{cexp.TABLES_DIR / (ccfg.stem + ".mem")}"',
        },
    )
