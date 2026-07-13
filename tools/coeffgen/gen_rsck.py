"""Generate Chebyshev coefficient ROMs for the on-chip RS correction (M15).

The Riemann-Siegel correction terms C_0..C_4 are entire functions of
p = frac(sqrt(t/2pi)); on chip they are evaluated by Clenshaw over Chebyshev
series in z = 2p - 1 (division-free — the classical Psi(p) quotient form
needs a divider the engine does not have, and monomial Horner at degree ~100
is catastrophically ill-conditioned at runtime precision).

Method: one high-degree Chebyshev interpolation of Psi(p) (pointwise evals
only — no mp.diff), then Psi', .., Psi^(12) by the exact Chebyshev
differentiation recurrence (d/dp = 2 d/dz), combined per Gabcke's formulas
(the same combinations as zetafpga.kernel.rs_setup._c_k, which this
generator cross-checks against).

Outputs per target width:
  rsck_wW.mem  : (kmax+1)*nc lines, Chebyshev coefficients of C_k as packed
                 MPF@W words, k-major (row k*nc+j = coefficient j of C_k)
  rsck_wW.json : {width, nc, kmax}
"""

import argparse
import json
import sys
from pathlib import Path

import mpmath as mp

WIDTHS = [64, 128, 192, 256]
KMAX = 4
DEGREE = 256  # interpolation degree for Psi (tail-checked below)


def _psi_of_z(z: mp.mpf) -> mp.mpf:
    p = (z + 1) / 2
    return mp.cos(2 * mp.pi * (p * p - p - mp.mpf(1) / 16)) / mp.cos(2 * mp.pi * p)


def cheb_interp(m: int) -> list[mp.mpf]:
    """Chebyshev coefficients of Psi on z in [-1,1] (c[0] in halved convention:
    f = c0/2 + sum_{j>=1} c_j T_j)."""
    xs = [mp.cos(mp.pi * (i + mp.mpf(1) / 2) / m) for i in range(m)]
    fs = [_psi_of_z(x) for x in xs]
    return [
        2 * mp.fsum(fs[i] * mp.cos(mp.pi * j * (i + mp.mpf(1) / 2) / m) for i in range(m)) / m
        for j in range(m)
    ]


def cheb_diff_p(c: list[mp.mpf]) -> list[mp.mpf]:
    """Coefficients of df/dp given f's (halved-c0 convention); dz/dp = 2."""
    n = len(c)
    d = [mp.mpf(0)] * n
    for j in range(n - 2, -1, -1):
        d[j] = (d[j + 2] if j + 2 < n else mp.mpf(0)) + 2 * (j + 1) * c[j + 1]
    return [2 * v for v in d]


def clenshaw(c: list[mp.mpf], z: mp.mpf) -> mp.mpf:
    """Evaluate with plain c0 (NOT halved) — the runtime convention."""
    b1 = b2 = mp.mpf(0)
    for j in range(len(c) - 1, 0, -1):
        b1, b2 = c[j] + 2 * z * b1 - b2, b1
    return c[0] + z * b1 - b2


def c_k_series(width: int) -> list[list[mp.mpf]]:
    """Chebyshev series (plain-c0 convention) of C_0..C_KMAX."""
    psi = cheb_interp(DEGREE)
    tail = max(abs(v) for v in psi[-DEGREE // 8 :])
    assert tail < mp.mpf(2) ** -(width + 96), f"raise DEGREE: tail {mp.nstr(tail, 3)}"

    d = [psi]
    for _ in range(12):
        d.append(cheb_diff_p(d[-1]))

    pi2 = mp.pi**2

    def combo(*terms: tuple[int, mp.mpf]) -> list[mp.mpf]:
        out = [mp.mpf(0)] * DEGREE
        for order, scale in terms:
            for j in range(DEGREE):
                out[j] += d[order][j] * scale
        return out

    one = mp.mpf(1)
    series = [
        combo((0, one)),
        combo((3, -1 / (96 * pi2))),
        combo((2, 1 / (64 * pi2)), (6, 1 / (18432 * pi2**2))),
        combo((1, -1 / (64 * pi2)), (5, -1 / (3840 * pi2**2)), (9, -1 / (5308416 * pi2**3))),
        combo(
            (0, 1 / (128 * pi2)),
            (4, 19 / (24576 * pi2**2)),
            (8, 11 / (5898240 * pi2**3)),
            (12, one / 2038431744 / pi2**4),
        ),
    ]
    # halved-c0 -> plain-c0 runtime convention
    return [[c[0] / 2, *c[1:]] for c in series]


def _self_check(series: list[list[mp.mpf]], width: int) -> None:
    """Cross-check the series against the mp.diff-based rs_setup._c_k."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "host"))
    from zetafpga.kernel.rs_setup import _c_k

    tol = mp.mpf(2) ** -(width + 8)
    for k, c in enumerate(series):
        for pv in (mp.mpf(1) / 7, mp.mpf(1) / 2, mp.mpf(9) / 10):
            got = clenshaw(c, 2 * pv - 1)
            ref = _c_k(k, pv)
            assert abs(got - ref) < tol, f"C_{k}({pv}): {mp.nstr(abs(got - ref), 3)}"


def generate(width: int, out_dir: Path) -> None:
    mp.mp.dps = int((width + 64) * 0.302) + 80

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "host"))
    from zetafpga.golden import mpfloat as mf
    from zetafpga.kernel.em_setup import mpf_from_real

    fmt = mf.Format(limbs=width // 64)
    series = c_k_series(width)
    _self_check(series, width)

    # truncate: sum of dropped |c_j| below the runtime evaluation floor
    floor = mp.mpf(2) ** -(width + 16)
    nc = 1
    for c in series:
        j = len(c)
        while j > 1 and mp.fsum(abs(v) for v in c[j - 1 :]) < floor:
            j -= 1
        nc = max(nc, j)

    lines = []
    for c in series:
        for j in range(nc):
            v = c[j] if j < len(c) else mp.mpf(0)
            word = mf.pack(mpf_from_real(v, fmt), fmt)
            lines.append(f"{word:0{(fmt.mpw + 3) // 4}x}")
    (out_dir / f"rsck_w{width}.mem").write_text("\n".join(lines) + "\n")

    meta = {"width": width, "nc": nc, "kmax": KMAX}
    (out_dir / f"rsck_w{width}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote rsck_w{width}: nc={nc}, kmax={KMAX}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("rtl/common/fn/tables"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for width in WIDTHS:
        generate(width, args.out_dir)


if __name__ == "__main__":
    main()
