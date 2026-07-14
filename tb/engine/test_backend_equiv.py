"""M9 acceptance: identical program bytes through the RTL engine (SimBackend)
and the GoldenBackend reference interpreter produce identical readback bytes."""

import os

import cocotb
import mpmath as mp
from cocotb.triggers import RisingEdge, Timer

from zetafpga.driver.golden_backend import GoldenBackend
from zetafpga.driver.sim_backend import SimBackend
from zetafpga.golden import cexp, expln, rs_z, theta
from zetafpga.golden import mpfloat as mf
from zetafpga.kernel import isa
from zetafpga.kernel.em_setup import build_program, mpf_from_real
from zetafpga.kernel.program import Program
from zetafpga.kernel.rs_setup import rs_entry, rs_n
from zetafpga.kernel.tables import lnn_entry

FMT = mf.Format(limbs=int(os.environ.get("ZETA_LIMBS", "1")))
ECFG = expln.load_cfg(FMT.width)
CCFG = cexp.load_cfg(FMT.width)


def _build() -> tuple[Program, int]:
    """A representative app program: critical-line mini-scan, mixing full
    COMPUTE_EM evaluations with COMPUTE_PS (Riemann-Siegel main-sum) ones."""
    from dataclasses import replace

    mp.mp.prec = 2 * FMT.width + 80
    sigma = mpf_from_real(mp.mpf(0.5), FMT)
    progs = [
        build_program(sigma, int(mp.nint(mp.mpf(tv) * (1 << 32))), FMT)
        for tv in [10.0, 12.0, 14.0, 14.134725, 16.0, 18.0]
    ]
    # every other evaluation runs as a power-sum-only descriptor
    progs = [replace(p, ps_only=(i % 2 == 1)) for i, p in enumerate(progs)]
    max_n = max(p.n for p in progs)
    entries = [lnn_entry(n, FMT, FMT.width + 64) for n in range(1, max_n + 2)]
    prg = Program(FMT)
    prg.write_lnn_table(entries).write_bern_table(list(progs[0].bern)).barrier()
    for p in progs:
        prg.compute_em(p)
    # pipelined-RS descriptors (COMPUTE_RS) in the same program
    rs_tfx = [int(mp.nint(mp.mpf(tv) * (1 << 32))) for tv in [100.0, 5000.0]]
    max_rs_n = max(rs_n(t) for t in rs_tfx)
    # +1 margin: COMPUTE_Z/ZGRID derive N on chip (may exceed rs_n by 1 at a boundary)
    prg.write_rs_table([rs_entry(k, FMT, FMT.width + 64) for k in range(1, max_rs_n + 2)])
    for t_fx in rs_tfx:
        prg.compute_rs(t_fx, rs_n(t_fx))
    # fully on-chip Z(t) (COMPUTE_Z) on the same RS table (both t >= t_min)
    for t_fx in rs_tfx:
        prg.compute_z(t_fx)
    # batched grid multi-eval (COMPUTE_ZGRID): 3 points below t=100
    prg.compute_zgrid(int(98.5 * (1 << 32)), int(0.5 * (1 << 32)), 3)
    # O-S binned-FFT grid main sum (COMPUTE_OS): 8 points at t=5000
    prg.compute_os(int(5000.0 * (1 << 32)), int(0.25 * (1 << 32)), rs_n(int(5000.0 * (1 << 32))), 8)
    prg.readback()
    return prg, max_n


@cocotb.test()
async def same_bytes_same_results(dut) -> None:
    async def clock() -> None:
        while True:
            dut.clk.value = 0
            await Timer(5, "ns")
            dut.clk.value = 1
            await Timer(5, "ns")

    cocotb.start_soon(clock())
    dut.rst_n.value = 0
    sim = SimBackend(dut, FMT)
    await sim.reset()
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    prg, max_n = _build()
    program_bytes = prg.to_bytes()
    expected_words = isa.readback_words(prg.expected_evals, FMT)

    gold = GoldenBackend(FMT)
    gold.submit(program_bytes)
    gold_bytes = gold.read(0, 8 * expected_words)

    timeout = 120_000 + prg.expected_evals * (max_n + 4) * 400
    await sim.submit(program_bytes, expected_words, timeout)
    sim_bytes = sim.read(0, 8 * expected_words)

    assert sim_bytes == gold_bytes, "RTL and reference backends disagree on identical bytes"


def test_backend_equiv(limbs: int) -> None:
    if limbs == 4:
        import pytest

        pytest.skip("Z256 equivalence runs in the nightly tier")
    import os as _os

    _os.environ["ZETA_LIMBS"] = str(limbs)
    fmt = mf.Format(limbs=limbs)
    ecfg = expln.load_cfg(fmt.width)
    ccfg = cexp.load_cfg(fmt.width)
    tcfg = theta.load_cfg(fmt.width)
    rcfg = rs_z.load_cfg(fmt.width)
    ecfg2 = expln.load_cfg(tcfg.w2)
    tables = expln.TABLES_DIR
    from common.runner import run_block

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
            "rtl/common/mp/mpf_recip.sv",
            "rtl/common/fx/fx_mul_mod1.sv",
            "rtl/common/fn/exp_mpf.sv",
            "rtl/common/fn/log_mpf.sv",
            "rtl/common/fn/cexp_turns.sv",
            "rtl/common/zeta/npow_s_kernel.sv",
            "rtl/common/zeta/euler_maclaurin_top.sv",
            "rtl/common/fn/cexp_pipe.sv",
            "rtl/common/zeta/rs_power_sum.sv",
            "rtl/common/zeta/rs_power_sum_tiled.sv",
            "rtl/common/fn/theta_turns.sv",
            "rtl/common/zeta/rs_z_unit.sv",
            "rtl/common/fn/fft_radix2.sv",
            "rtl/common/zeta/os_grid_sum.sv",
            "rtl/common/engine/zeta_engine.sv",
        ],
        "zeta_engine",
        __file__,
        parameters={
            "LIMBS": fmt.limbs,
            "RS_LANES": 2,  # multi-lane RS path must stay byte-identical (M17)
            "PHW": fmt.width + 32,
            "FG": ecfg.fg,
            "CONSTW": ecfg.constw,
            "TERMS": ecfg.terms,
            "CTERMS": ccfg.terms,
            "EXP_ROM": f'"{tables / (ecfg.stem + "_exp.mem")}"',
            "CEXP_ROM": f'"{tables / (ccfg.stem + ".mem")}"',
            "KTERMS": tcfg.k,
            "LOG_TERMS": ecfg2.terms,
            "THETA_FX_ROM": f'"{tables / (tcfg.stem + "_fx.mem")}"',
            "THETA_MPF_ROM": f'"{tables / (tcfg.stem + "_mpf.mem")}"',
            "EXP_ROM2": f'"{tables / (ecfg2.stem + "_exp.mem")}"',
            "LN_ROM2": f'"{tables / (ecfg2.stem + "_ln.mem")}"',
            "NC": rcfg.nc,
            "KMAX": rcfg.kmax,
            "RSCK_ROM": f'"{tables / (rcfg.stem + ".mem")}"',
            "FFT_ROM": f'"{tables / "fft_m128.mem"}"',
            "OS_ROM": f'"{tables / "fft_os.mem"}"',
        },
    )
