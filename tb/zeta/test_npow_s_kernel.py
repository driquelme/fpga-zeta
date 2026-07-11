"""npow_s_kernel: RTL vs golden model, bit-exact (accuracy proven at golden
level in tests/test_golden_npow.py)."""

import os
import random

import cocotb
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from zetafpga.golden import cexp, expln, npow
from zetafpga.golden import mpfloat as mf
from zetafpga.kernel.tables import lnn_entry

FMT = mf.Format(limbs=int(os.environ.get("ZETA_LIMBS", "2")))
ECFG = expln.load_cfg(FMT.width)
CCFG = cexp.load_cfg(FMT.width)
PHW = FMT.width + 32
BW = PHW + 32
N_RANDOM = int(os.environ.get("ZETA_TEST_N", "10000"))
N_OPS = min(max(N_RANDOM // 100, 20), 120)  # FSM: ~FG-scale cycles per op
TIMEOUT = ECFG.fg + CCFG.terms + ECFG.terms + 80

# Precompute table entries for a small n set (mpmath per entry is slow).
N_SET = [1, 2, 3, 7, 10, 100, 4096, 99967, 10**6]
ENTRIES = {n: lnn_entry(n, FMT, BW) for n in N_SET}


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
    dut.sigma.value = 0
    dut.lnn_fx.value = 0
    dut.lnn2pi.value = 0
    dut.t_fx.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def _check(dut, sigma: mf.MPF, n: int, t_fx: int) -> None:
    lnn_fx, lnn2pi = ENTRIES[n]
    while dut.in_ready.value == 0:
        await RisingEdge(dut.clk)
    dut.in_valid.value = 1
    dut.sigma.value = mf.pack(sigma, FMT)
    dut.lnn_fx.value = lnn_fx
    dut.lnn2pi.value = lnn2pi
    dut.t_fx.value = t_fx
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    for _ in range(TIMEOUT):
        await FallingEdge(dut.clk)
        if dut.out_valid.value == 1:
            got = (
                int(dut.re_o.value),
                int(dut.im_o.value),
                int(dut.ovf.value),
                int(dut.unf.value),
            )
            re, im, ovf, unf = npow.npow_s(sigma, lnn_fx, lnn2pi, t_fx, FMT, PHW, BW)
            expected = (mf.pack(re, FMT), mf.pack(im, FMT), int(ovf), int(unf))
            assert got == expected, (
                f"sigma={sigma}, n={n}, t_fx={t_fx:#x}: got {got}, expected {expected}"
            )
            return
    raise AssertionError(f"timeout: sigma={sigma}, n={n}, t_fx={t_fx:#x}")


@cocotb.test()
async def directed(dut) -> None:
    await _reset(dut)
    w = FMT.width
    one = mf.MPF(0, 1, 1 << (w - 1))
    three = mf.MPF(0, 2, 3 << (w - 2))
    for sigma, n, t_fx in [
        (three, 7, 5 << 32),  # 7^-(3+5i)
        (one, 2, 0),  # 2^-1, t=0
        (npow.neg(one), 100, 1 << 31),  # n^{+1}, fractional t
        (mf.zero(0), 10, 123 << 30),  # sigma = 0: pure phase
        (three, 1, 99 << 32),  # n = 1 -> exactly 1
        (mf.special(1), 7, 1 << 32),  # special propagates
        (mf.MPF(0, 40, 1 << (w - 1)), 3, 1 << 32),  # huge sigma: unf
        (mf.MPF(1, 40, 1 << (w - 1)), 3, 1 << 32),  # huge negative sigma: ovf
    ]:
        await _check(dut, sigma, n, t_fx)


@cocotb.test()
async def random_vectors(dut) -> None:
    await _reset(dut)
    rng = random.Random(800 + FMT.limbs)
    for _ in range(N_OPS):
        n = rng.choice(N_SET)
        mant = (1 << (FMT.width - 1)) | rng.getrandbits(FMT.width - 1)
        sigma = mf.MPF(rng.getrandbits(1), rng.randrange(-3, 5), mant)
        await _check(dut, sigma, n, rng.getrandbits(52))


def test_npow_s_kernel(limbs: int) -> None:
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
            "rtl/common/fn/exp_mpf.sv",
            "rtl/common/fn/cexp_turns.sv",
            "rtl/common/zeta/npow_s_kernel.sv",
        ],
        "npow_s_kernel",
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
