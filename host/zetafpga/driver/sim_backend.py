"""SimBackend: the Backend contract over the zeta_engine word streams, for
use inside cocotb tests (async twin of the synchronous protocol; the
synchronous wrapper arrives with the Phase-2 socket harness).

Word-level semantics are identical to GoldenBackend: submit program words in,
readback words accumulate into a host-side buffer served by read().
"""

from typing import Any

from cocotb.triggers import FallingEdge, RisingEdge

from zetafpga.golden import mpfloat as mf
from zetafpga.kernel import isa
from zetafpga.kernel.program import Program


class SimBackend:
    def __init__(self, dut: Any, fmt: mf.Format) -> None:
        self.dut: Any = dut
        self.fmt = fmt
        self._out = bytearray()

    async def reset(self) -> None:
        self.dut.in_valid.value = 0
        self.dut.in_data.value = 0
        self.dut.out_ready.value = 0

    def read(self, addr: int, length: int) -> bytes:
        return bytes(self._out[addr : addr + length])

    async def submit(self, program: bytes, expected_out_words: int, timeout: int) -> None:
        dut = self.dut
        words = isa.from_bytes(program)
        collected: list[int] = []

        # Interleave: drive the program while draining the output stream, so
        # multi-READBACK programs cannot deadlock on a full result path.
        dut.out_ready.value = 1
        idx = 0
        driving = False
        for _ in range(timeout):
            if idx < len(words) and not driving:
                dut.in_valid.value = 1
                dut.in_data.value = words[idx]
                driving = True
            await FallingEdge(dut.clk)
            in_beat = driving and dut.in_ready.value == 1
            if dut.out_valid.value == 1:
                collected.append(int(dut.out_data.value))
            await RisingEdge(dut.clk)
            if in_beat:
                idx += 1
                driving = False
                if idx == len(words):
                    dut.in_valid.value = 0
            if idx == len(words) and len(collected) >= expected_out_words:
                break
        dut.out_ready.value = 0
        assert idx == len(words), f"program stalled at word {idx}/{len(words)}"
        assert len(collected) >= expected_out_words, (
            f"collected {len(collected)}/{expected_out_words} output words"
        )
        self._out += isa.to_bytes(collected)

    async def run(self, prg: Program, timeout: int) -> isa.ReadbackResult:
        start = len(self._out)
        expected = isa.readback_words(prg.expected_evals, self.fmt)
        await self.submit(prg.to_bytes(), expected, timeout)
        words = isa.from_bytes(self.read(start, len(self._out) - start))
        return isa.parse_readback(words, self.fmt)
