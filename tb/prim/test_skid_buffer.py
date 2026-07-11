"""cocotb tests for skid_buffer + pytest entry point building/running Verilator."""

import os
import random
from pathlib import Path

import cocotb
from cocotb.triggers import RisingEdge, Timer

WIDTH = 32


async def _clock(clk) -> None:
    while True:
        clk.value = 0
        await Timer(5, "ns")
        clk.value = 1
        await Timer(5, "ns")


async def _reset(dut) -> None:
    dut.rst_n.value = 0
    dut.s_valid.value = 0
    dut.s_data.value = 0
    dut.m_ready.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def _drive(dut, words: list[int], p_valid: float, rng: random.Random) -> None:
    for word in words:
        while rng.random() > p_valid:
            dut.s_valid.value = 0
            await RisingEdge(dut.clk)
        dut.s_valid.value = 1
        dut.s_data.value = word
        await RisingEdge(dut.clk)
        while dut.s_ready.value == 0:
            await RisingEdge(dut.clk)
    dut.s_valid.value = 0


async def _sink(dut, count: int, p_ready: float, rng: random.Random) -> list[int]:
    received: list[int] = []
    while len(received) < count:
        dut.m_ready.value = 1 if rng.random() <= p_ready else 0
        await RisingEdge(dut.clk)
        if dut.m_ready.value == 1 and dut.m_valid.value == 1:
            received.append(int(dut.m_data.value))
    dut.m_ready.value = 0
    return received


async def _run_stream(dut, n_words: int, p_valid: float, p_ready: float, seed: int) -> None:
    rng = random.Random(seed)
    cocotb.start_soon(_clock(dut.clk))
    await _reset(dut)
    words = [rng.getrandbits(WIDTH) for _ in range(n_words)]
    cocotb.start_soon(_drive(dut, words, p_valid, rng))
    received = await _sink(dut, n_words, p_ready, rng)
    mismatches = [i for i, (a, b) in enumerate(zip(received, words, strict=True)) if a != b]
    assert not mismatches, f"data corrupted/reordered, first mismatch at index {mismatches[0]}"


@cocotb.test()
async def stream_full_throughput(dut) -> None:
    """No stalls on either side: data must flow one word per cycle, in order."""
    await _run_stream(dut, n_words=500, p_valid=1.0, p_ready=1.0, seed=1)


@cocotb.test()
async def stream_random_stalls(dut) -> None:
    """Random gaps upstream and backpressure downstream."""
    await _run_stream(dut, n_words=500, p_valid=0.6, p_ready=0.5, seed=2)


@cocotb.test()
async def stream_heavy_backpressure(dut) -> None:
    """Fast producer, slow consumer — exercises the skid register constantly."""
    await _run_stream(dut, n_words=300, p_valid=1.0, p_ready=0.2, seed=3)


def test_skid_buffer() -> None:
    """pytest entry point: build with Verilator and run the cocotb tests above."""
    try:
        from cocotb_tools.runner import get_runner
    except ImportError:  # cocotb < 2.0
        from cocotb.runner import get_runner

    repo = Path(__file__).resolve().parents[2]
    build_dir = repo / "sim_build" / "skid_buffer"
    os.environ["PYTHONPATH"] = (
        f"{Path(__file__).resolve().parent}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    )
    runner = get_runner("verilator")
    runner.build(
        sources=[repo / "rtl" / "common" / "prim" / "skid_buffer.sv"],
        hdl_toplevel="skid_buffer",
        build_dir=str(build_dir),
        build_args=(["--trace-fst"] if os.getenv("WAVES") else []),
        parameters={"WIDTH": WIDTH},
    )
    runner.test(
        hdl_toplevel="skid_buffer",
        test_module="test_skid_buffer",
        build_dir=str(build_dir),
    )
