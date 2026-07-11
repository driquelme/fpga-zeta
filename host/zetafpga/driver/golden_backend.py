"""GoldenBackend: a bit-true reference implementation of the Backend contract.

Interprets kernel-program bytes exactly as zeta_engine.sv does — same table
model, same result buffering, same readback format — but executes COMPUTE_EM
with the golden models. Because it consumes the same bytes and produces the
same words, it independently validates the ISA contract: the RTL engine and
this interpreter must produce identical readback streams for identical
programs (proven in tb/engine/test_backend_equiv.py).

It is also the Phase-1 runner for the applications; the Verilator socket
harness and PCIe backends slot in behind the same interface in Phase 2.
"""

from zetafpga.golden import mpfloat as mf
from zetafpga.golden import zeta_em
from zetafpga.kernel import isa
from zetafpga.kernel.em_setup import EmProgram
from zetafpga.kernel.program import Program


class GoldenBackend:
    def __init__(self, fmt: mf.Format) -> None:
        self.fmt = fmt
        self._lnn: list[tuple[int, int]] = []
        self._bern: list[mf.MPF] = []
        self._results: list[isa.EmResult] = []
        self._ovf = False
        self._unf = False
        self._err = False
        self._out = bytearray()

    # ---- Backend protocol -----------------------------------------------------
    def write(self, addr: int, data: bytes) -> None:
        raise NotImplementedError("v0: no register space; use submit()")

    def read(self, addr: int, length: int) -> bytes:
        return bytes(self._out[addr : addr + length])

    def submit(self, program: bytes) -> None:
        self._execute(isa.from_bytes(program))

    def wait_idle(self, timeout_s: float) -> None:
        return  # synchronous execution

    # ---- convenience ------------------------------------------------------------
    def run(self, prg: Program) -> isa.ReadbackResult:
        start = len(self._out)
        self.submit(prg.to_bytes())
        words = isa.from_bytes(self.read(start, len(self._out) - start))
        return isa.parse_readback(words, self.fmt)

    # ---- interpreter --------------------------------------------------------------
    def _execute(self, words: list[int]) -> None:
        fmt = self.fmt
        k = isa.mpf_words(fmt)
        ew = isa.entry_words(fmt)
        lnw = fmt.width + 24 + 8
        pos = 0

        def take(n: int) -> list[int]:
            nonlocal pos
            chunk = words[pos : pos + n]
            assert len(chunk) == n, "truncated program"
            pos += n
            return chunk

        def join(ws: list[int]) -> int:
            return sum(w << (64 * i) for i, w in enumerate(ws))

        while pos < len(words):
            d = take(4)
            op = d[0] & 0xFF
            table_id = (d[0] >> 8) & 0xFF
            count = (d[0] >> 16) & 0xFFFFFF
            m = (d[0] >> 40) & 0xFFF
            if op in (isa.Op.NOP, isa.Op.SET_FORMAT, isa.Op.BARRIER):
                continue
            if op == isa.Op.WRITE_TABLE:
                if table_id == isa.TBL_LNN:
                    self._lnn = []
                    for _ in range(count):
                        v = join(take(ew))
                        self._lnn.append((v & ((1 << lnw) - 1), v >> lnw))
                else:
                    self._bern = [
                        mf.unpack(join(take(k)) & ((1 << fmt.mpw) - 1), fmt) for _ in range(count)
                    ]
            elif op in (isa.Op.COMPUTE_EM, isa.Op.COMPUTE_PS):
                t_fx = take(1)[0]
                fields = [mf.unpack(join(take(k)) & ((1 << fmt.mpw) - 1), fmt) for _ in range(5)]
                prog = EmProgram(
                    fmt=fmt,
                    phw=fmt.width + 32,
                    bw=fmt.width + 64,
                    sigma=fields[0],
                    t_mpf=fields[1],
                    t_fx=t_fx,
                    n=count,
                    m=m,
                    inv_sm1_re=fields[2],
                    inv_sm1_im=fields[3],
                    inv_np2=fields[4],
                    bern=tuple(self._bern[:m]),
                    entries=tuple(self._lnn[: count + 1]),
                    ps_only=(op == isa.Op.COMPUTE_PS),
                )
                re, im, ovf, unf = zeta_em.zeta_em(prog)
                self._results.append(isa.EmResult(mf.pack(re, fmt), mf.pack(im, fmt), ovf, unf))
                self._ovf |= ovf
                self._unf |= unf
            elif op == isa.Op.READBACK:
                out = [
                    (len(self._results) & 0xFFFFFF)
                    | (int(self._ovf) << 24)
                    | (int(self._unf) << 25)
                    | (int(self._err) << 26)
                ]
                for r in self._results:
                    out += isa._split(r.re_word, k)
                    out += isa._split(r.im_word, k)
                    out += [int(r.ovf) | (int(r.unf) << 1)]
                self._out += isa.to_bytes(out)
                self._results = []
                self._ovf = self._unf = self._err = False
            else:
                self._err = True  # malformed opcode, no payload by contract
