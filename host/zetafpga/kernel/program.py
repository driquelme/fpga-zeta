"""Program: assembles zeta-engine kernel programs from EmPrograms.

The "host generates execution kernels" of the project description, made
concrete: tables + descriptors in, a DMA-able byte stream out.
"""

from zetafpga.golden import mpfloat as mf
from zetafpga.kernel import isa
from zetafpga.kernel.em_setup import EmProgram


class Program:
    def __init__(self, fmt: mf.Format) -> None:
        self.fmt = fmt
        self._words: list[int] = []
        self._expected_evals = 0

    def raw(self, words: list[int]) -> "Program":
        self._words += words
        return self

    def nop(self) -> "Program":
        return self.raw(isa.descriptor(isa.Op.NOP))

    def barrier(self) -> "Program":
        return self.raw(isa.descriptor(isa.Op.BARRIER))

    def write_lnn_table(self, entries: list[tuple[int, int]]) -> "Program":
        self.raw(isa.descriptor(isa.Op.WRITE_TABLE, table_id=isa.TBL_LNN, count=len(entries)))
        for lnn_fx, lnn2pi in entries:
            self.raw(isa.pack_lnn_entry(lnn_fx, lnn2pi, self.fmt))
        return self

    def write_bern_table(self, bern: list[mf.MPF]) -> "Program":
        self.raw(isa.descriptor(isa.Op.WRITE_TABLE, table_id=isa.TBL_BERN, count=len(bern)))
        k = isa.mpf_words(self.fmt)
        for b in bern:
            self.raw(isa._split(mf.pack(b, self.fmt), k))
        return self

    def compute_em(self, prog: EmProgram) -> "Program":
        assert prog.fmt == self.fmt
        self._expected_evals += 1
        return self.raw(isa.pack_compute_em(prog))

    def write_rs_table(self, entries: list[tuple[int, mf.MPF]]) -> "Program":
        self.raw(isa.descriptor(isa.Op.WRITE_TABLE, table_id=isa.TBL_RS, count=len(entries)))
        for lnn2pi, amp in entries:
            self.raw(isa.pack_rs_entry(lnn2pi, amp, self.fmt))
        return self

    def compute_rs(self, t_fx: int, n: int) -> "Program":
        self._expected_evals += 1
        return self.raw(isa.pack_compute_rs(t_fx, n))

    def compute_z(self, t_fx: int) -> "Program":
        self._expected_evals += 1
        return self.raw(isa.pack_compute_z(t_fx))

    def compute_zgrid(self, t0_fx: int, dt_fx: int, count: int) -> "Program":
        self._expected_evals += count
        return self.raw(isa.pack_compute_zgrid(t0_fx, dt_fx, count))

    def readback(self) -> "Program":
        return self.raw(isa.descriptor(isa.Op.READBACK))

    @property
    def expected_evals(self) -> int:
        return self._expected_evals

    def to_bytes(self) -> bytes:
        return isa.to_bytes(self._words)

    def words(self) -> list[int]:
        return list(self._words)
