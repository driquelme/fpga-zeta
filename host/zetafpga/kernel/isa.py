"""The zeta-engine descriptor ISA (v0) — the host/engine kernel contract.

A kernel program is a stream of 64-bit little-endian words: 4-word (256-bit)
descriptors, each optionally followed by payload words.

Descriptor word 0 (words 1-3 reserved in v0):
    [7:0]   opcode
    [15:8]  table_id      (WRITE_TABLE)
    [39:16] count         (WRITE_TABLE: entries; COMPUTE_EM: N; READBACK: 0=all)
    [51:40] m             (COMPUTE_EM: Bernoulli terms M)
    [63:52] reserved

Opcodes:
    NOP=0  SET_FORMAT=1 (accepted, format is fixed per build)  WRITE_TABLE=2
    COMPUTE_EM=3  READBACK=4  BARRIER=5 (fence; engine is in-order)
Unknown opcodes set the sticky err flag and are skipped (4 words consumed).

Payloads (multi-word fields little-endian, lowest word first):
    WRITE_TABLE lnn (id 0):  count * entry_words words, entry = {lnn2pi, lnn_fx}
    WRITE_TABLE bern (id 1): count * mpf_words words (MPF words)
    COMPUTE_EM: t_fx (1 word) + sigma, t_mpf, inv_sm1_re, inv_sm1_im, inv_np2
                (mpf_words each)
    READBACK output: header {[23:0] eval_count, [24] ovf, [25] unf, [26] err}
                then per eval: z_re, z_im (mpf_words each) + 1 flag word
                {bit0 ovf, bit1 unf}. Clears the result buffer and flags.
"""

import struct
from dataclasses import dataclass
from enum import IntEnum

from zetafpga.golden import mpfloat as mf
from zetafpga.kernel.em_setup import EmProgram


class Op(IntEnum):
    NOP = 0
    SET_FORMAT = 1
    WRITE_TABLE = 2
    COMPUTE_EM = 3
    READBACK = 4
    BARRIER = 5
    COMPUTE_PS = 6  # power sum only (Riemann-Siegel main sum); same payload as COMPUTE_EM


TBL_LNN = 0
TBL_BERN = 1


def mpf_words(fmt: mf.Format) -> int:
    return (fmt.mpw + 63) // 64


def entry_words(fmt: mf.Format) -> int:
    phw = fmt.width + 32
    bw = phw + 32
    lnw = fmt.width + 24 + 8
    return (lnw + bw + 8 + 63) // 64


def _split(value: int, words: int) -> list[int]:
    return [(value >> (64 * i)) & ((1 << 64) - 1) for i in range(words)]


def desc_word0(op: Op, table_id: int = 0, count: int = 0, m: int = 0) -> int:
    assert 0 <= count < (1 << 24) and 0 <= m < (1 << 12)
    return int(op) | (table_id << 8) | (count << 16) | (m << 40)


def descriptor(op: Op, table_id: int = 0, count: int = 0, m: int = 0) -> list[int]:
    return [desc_word0(op, table_id, count, m), 0, 0, 0]


def pack_lnn_entry(lnn_fx: int, lnn2pi: int, fmt: mf.Format) -> list[int]:
    lnw = fmt.width + 24 + 8
    return _split(lnn_fx | (lnn2pi << lnw), entry_words(fmt))


def pack_compute_em(prog: EmProgram) -> list[int]:
    fmt = prog.fmt
    k = mpf_words(fmt)
    op = Op.COMPUTE_PS if prog.ps_only else Op.COMPUTE_EM
    words = descriptor(op, count=prog.n, m=prog.m)
    words += [prog.t_fx]
    for v in (prog.sigma, prog.t_mpf, prog.inv_sm1_re, prog.inv_sm1_im, prog.inv_np2):
        words += _split(mf.pack(v, fmt), k)
    return words


@dataclass(frozen=True)
class EmResult:
    re_word: int
    im_word: int
    ovf: bool
    unf: bool


@dataclass(frozen=True)
class ReadbackResult:
    results: tuple[EmResult, ...]
    ovf: bool
    unf: bool
    err: bool


def parse_readback(words: list[int], fmt: mf.Format) -> ReadbackResult:
    k = mpf_words(fmt)
    hdr = words[0]
    count = hdr & 0xFFFFFF
    mask = (1 << fmt.mpw) - 1
    results = []
    pos = 1
    for _ in range(count):
        re_w = sum(words[pos + i] << (64 * i) for i in range(k)) & mask
        im_w = sum(words[pos + k + i] << (64 * i) for i in range(k)) & mask
        flags = words[pos + 2 * k]
        results.append(EmResult(re_w, im_w, bool(flags & 1), bool(flags & 2)))
        pos += 2 * k + 1
    return ReadbackResult(
        tuple(results), bool(hdr & (1 << 24)), bool(hdr & (1 << 25)), bool(hdr & (1 << 26))
    )


def readback_words(count: int, fmt: mf.Format) -> int:
    """Number of output words a READBACK of `count` evals produces."""
    return 1 + count * (2 * mpf_words(fmt) + 1)


def to_bytes(words: list[int]) -> bytes:
    return b"".join(struct.pack("<Q", w) for w in words)


def from_bytes(data: bytes) -> list[int]:
    assert len(data) % 8 == 0
    return [struct.unpack_from("<Q", data, i)[0] for i in range(0, len(data), 8)]
