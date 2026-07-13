"""zeta_engine: end-to-end kernel programs through the descriptor path.

The M8 acceptance: a Python-assembled program (tables -> barrier -> batch of
COMPUTE_EM -> READBACK) produces M7-quality (bit-exact vs golden) results,
and malformed descriptors flag err without derailing subsequent work.
"""

import os

import cocotb
import mpmath as mp
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from zetafpga.golden import cexp, expln, zeta_em
from zetafpga.golden import mpfloat as mf
from zetafpga.kernel import isa
from zetafpga.kernel.em_setup import EmProgram, build_program, mpf_from_real
from zetafpga.kernel.program import Program
from zetafpga.kernel.tables import lnn_entry

FMT = mf.Format(limbs=int(os.environ.get("ZETA_LIMBS", "1")))
ECFG = expln.load_cfg(FMT.width)
CCFG = cexp.load_cfg(FMT.width)
N_EVALS = int(os.environ.get("ZETA_EM_EVALS", "12"))


def _programs() -> list[EmProgram]:
    mp.mp.prec = 2 * FMT.width + 80
    progs = []
    for k in range(N_EVALS):
        sv = 2.0 + k / 8.0  # sigma grid
        tv = [0.0, 1.5, 14.134725141734695][k % 3]
        sigma = mpf_from_real(mp.mpf(sv), FMT)
        t_fx = int(mp.nint(mp.mpf(tv) * (1 << 32)))
        progs.append(build_program(sigma, t_fx, FMT))
    return progs


PROGS = _programs()
MAX_N = max(p.n for p in PROGS)
MAX_M = max(p.m for p in PROGS)


def _shared_tables() -> tuple[list[tuple[int, int]], list[mf.MPF]]:
    bw = FMT.width + 64
    entries = [lnn_entry(n, FMT, bw) for n in range(1, MAX_N + 2)]
    bern = list(max(PROGS, key=lambda p: p.m).bern)
    return entries, bern


ENTRIES, BERN = _shared_tables()


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
    dut.in_data.value = 0
    dut.out_ready.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def _submit(dut, words: list[int]) -> None:
    for w in words:
        dut.in_valid.value = 1
        dut.in_data.value = w
        while True:
            await FallingEdge(dut.clk)
            if dut.in_ready.value == 1:
                await RisingEdge(dut.clk)
                break
    dut.in_valid.value = 0


async def _collect(dut, count: int, timeout: int) -> list[int]:
    out: list[int] = []
    dut.out_ready.value = 1
    for _ in range(timeout):
        await FallingEdge(dut.clk)
        if dut.out_valid.value == 1:
            out.append(int(dut.out_data.value))
            if len(out) == count:
                break
    dut.out_ready.value = 0
    assert len(out) == count, f"collected {len(out)}/{count} readback words"
    return out


def _expected() -> list[tuple[int, int, int, int]]:
    exp = []
    for prog in PROGS:
        re, im, ovf, unf = zeta_em.zeta_em(prog)
        exp.append((mf.pack(re, FMT), mf.pack(im, FMT), int(ovf), int(unf)))
    return exp


@cocotb.test()
async def batch_program(dut) -> None:
    """Tables -> barrier -> batch COMPUTE_EM -> READBACK, bit-exact."""
    await _reset(dut)
    prg = Program(FMT)
    prg.write_lnn_table(ENTRIES).write_bern_table(BERN).barrier()
    for p in PROGS:
        prg.compute_em(p)
    prg.readback()

    timeout = 40_000 + N_EVALS * (MAX_N + 4) * 400
    cocotb.start_soon(_submit(dut, prg.words()))
    words = await _collect(dut, isa.readback_words(N_EVALS, FMT), timeout)

    rb = isa.parse_readback(words, FMT)
    assert not rb.err, "err flag set on a well-formed program"
    assert len(rb.results) == N_EVALS
    for got, (ere, eim, eovf, eunf) in zip(rb.results, _expected(), strict=True):
        assert (got.re_word, got.im_word, int(got.ovf), int(got.unf)) == (
            ere,
            eim,
            eovf,
            eunf,
        ), "engine result != golden zeta_em"


@cocotb.test()
async def malformed_then_recover(dut) -> None:
    """Unknown opcode sets err; the engine still executes what follows."""
    await _reset(dut)
    prg = Program(FMT)
    prg.write_lnn_table(ENTRIES).write_bern_table(BERN)
    prg.raw([0x3F, 0, 0, 0])  # malformed descriptor (no payload)
    prg.compute_em(PROGS[0]).readback()

    timeout = 40_000 + (MAX_N + 4) * 400
    cocotb.start_soon(_submit(dut, prg.words()))
    words = await _collect(dut, isa.readback_words(1, FMT), timeout)
    rb = isa.parse_readback(words, FMT)
    assert rb.err, "malformed opcode did not set err"
    ere, eim, eovf, eunf = _expected()[0]
    got = rb.results[0]
    assert (got.re_word, got.im_word, int(got.ovf), int(got.unf)) == (ere, eim, eovf, eunf)


def test_zeta_engine() -> None:
    from common.runner import run_block

    from zetafpga.golden import rs_z, theta

    fmt = FMT
    ecfg = ECFG
    ccfg = CCFG
    tcfg = theta.load_cfg(fmt.width)
    rcfg = rs_z.load_cfg(fmt.width)
    ecfg2 = expln.load_cfg(tcfg.w2)
    tables = expln.TABLES_DIR
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
            "rtl/common/engine/zeta_engine.sv",
        ],
        "zeta_engine",
        __file__,
        parameters={
            "LIMBS": fmt.limbs,
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
        },
    )
