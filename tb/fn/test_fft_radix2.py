"""fft_radix2: RTL vs golden model, bit-exact (accuracy proven at golden level)."""

import cmath
import math
import os
import random

import cocotb
import pytest
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from zetafpga.golden import fft

M = int(os.environ.get("ZETA_FFT_M", "64"))
DW = 64
MASK = (1 << DW) - 1
FRAC = 54


def _vectors() -> list[list[tuple[int, int]]]:
    rng = random.Random(f"fft_rtl{M}")
    vecs = []
    # impulse, DC, and two O-S-like sparse bin loads
    imp = [(0, 0)] * M
    imp[0] = (1 << FRAC, 0)
    vecs.append(imp)
    vecs.append([(1 << (FRAC - 6), 1 << (FRAC - 8))] * M)
    for _ in range(2):
        v = [0j] * M
        for _ in range(M // 4):
            v[rng.randrange(M)] += cmath.rect(rng.uniform(0, 0.25), rng.uniform(0, 2 * math.pi))
        vecs.append([(round(c.real * 2**FRAC), round(c.imag * 2**FRAC)) for c in v])
    return vecs


VECTORS = _vectors()


async def _clock(clk) -> None:
    while True:
        clk.value = 0
        await Timer(5, "ns")
        clk.value = 1
        await Timer(5, "ns")


@cocotb.test()
async def fft_bit_exact(dut) -> None:
    cfg = fft.load_cfg(M)
    cocotb.start_soon(_clock(dut.clk))
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.out_ready.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    for vec in VECTORS:
        expected = fft.fft_fx(vec, cfg)

        for re, im in vec:
            dut.in_valid.value = 1
            dut.in_data.value = ((im & MASK) << DW) | (re & MASK)
            while True:
                await FallingEdge(dut.clk)
                if dut.in_ready.value == 1:
                    await RisingEdge(dut.clk)
                    break
        dut.in_valid.value = 0

        dut.out_ready.value = 1
        got: list[tuple[int, int]] = []
        limit = 6 * M * (M.bit_length() + 2) + 200
        for _ in range(limit):
            await FallingEdge(dut.clk)
            if dut.out_valid.value == 1:
                word = int(dut.out_data.value)
                re = word & MASK
                im = (word >> DW) & MASK
                got.append(
                    (
                        re - (1 << DW) if re >> (DW - 1) else re,
                        im - (1 << DW) if im >> (DW - 1) else im,
                    )
                )
                if len(got) == M:
                    # complete the final ready/valid transfer before revoking
                    await RisingEdge(dut.clk)
                    break
        else:
            raise AssertionError(f"timeout: collected {len(got)}/{M}")
        dut.out_ready.value = 0
        assert got == expected, "RTL FFT != golden"
        await RisingEdge(dut.clk)


@pytest.mark.parametrize("m", [64, 256])
def test_fft_radix2(m: int) -> None:
    os.environ["ZETA_FFT_M"] = str(m)
    from common.runner import run_block

    run_block(
        ["rtl/common/fn/fft_radix2.sv"],
        "fft_radix2",
        __file__,
        parameters={
            "M": m,
            "ROM": f'"{fft.TABLES_DIR / f"fft_m{m}.mem"}"',
        },
    )
