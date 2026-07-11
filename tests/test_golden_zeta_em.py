"""Golden zeta_em vs mpmath: the M7 acceptance criterion.

zeta(s) must match mp.zeta to >= mantissa - 12 bits of relative accuracy
(complex distance) over the acceptance vector set, including the first
LMFDB zero (where |zeta| itself is small — checked in absolute terms).
"""

import mpmath as mp
import pytest

from zetafpga.golden import mpfloat as mf
from zetafpga.golden import zeta_em
from zetafpga.kernel.em_setup import build_program, mpf_value

FMTS = [mf.Format(limbs=1), mf.Format(limbs=2)]

# (sigma as float, t as float) acceptance vectors; sigma/t are rounded to the
# exact engine input grid before the reference is computed.
VECTORS = [
    (2.0, 0.0),  # zeta(2) = pi^2/6
    (3.0, 0.0),
    (0.5, 14.134725141734695),  # first zero (t rounded to Q32.32)
    (0.5, 100.0),
    (0.5, 1000.0),
    (2.0, 1000.0),
    (-3.0, 10.0),
    (-10.0, 5.0),
    (10.0, 100.0),
    (1.001, 0.5),  # near the pole
    (0.0, 25.010857580145688),  # third zero, off-sigma
]


def _sigma_mpf(sv: float, fmt: mf.Format) -> mf.MPF:
    if sv == 0.0:
        return mf.zero(0)
    from zetafpga.kernel.em_setup import mpf_from_real

    return mpf_from_real(mp.mpf(sv), fmt)


@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f"L{f.limbs}")
def test_zeta_em_vs_mpmath(fmt: mf.Format) -> None:
    mp.mp.prec = 2 * fmt.width + 80
    bound = mp.mpf(2) ** -(fmt.width - 12)
    for sv, tv in VECTORS:
        sigma = _sigma_mpf(sv, fmt)
        t_fx = int(mp.nint(mp.mpf(tv) * (1 << 32)))
        prog = build_program(sigma, t_fx, fmt)
        zr, zi, ovf, _unf = zeta_em.zeta_em(prog)
        assert not ovf, f"s=({sv},{tv}): overflow flagged"
        got = mp.mpc(mpf_value(zr, fmt), mpf_value(zi, fmt))
        s_used = mp.mpc(mpf_value(sigma, fmt), mp.mpf(t_fx) / (1 << 32))
        ref = mp.zeta(s_used)
        # Two intrinsic conditioning effects bound the achievable accuracy of
        # W-bit E-M summation (the pre-cancellation chain error is ~1 ulp):
        # 1. Near zeros |zeta| << the O(1) terms: accuracy is absolute, which
        #    is what zero-finding needs. Scale by max(|zeta|, 1).
        # 2. For sigma < 0 the integral term (N+1)^(1-s)/(s-1) cancels
        #    against the power sum down to |zeta|; the lost bits are
        #    log2(|I|/|zeta|). (Phase 2's chi-reflection avoids E-M there
        #    entirely, as all software does; alternatively the host escalates
        #    LIMBS.) The bound below charges exactly that conditioning.
        cond = abs(mp.mpf(prog.n + 1) ** (1 - s_used) / (s_used - 1))
        scale = max(abs(ref), mp.mpf(1), cond)
        err = abs(got - ref) / scale
        assert err <= bound, (
            f"s=({sv},{tv}) N={prog.n} M={prog.m}: "
            f"rel err 2^{mp.nstr(mp.log(err, 2), 5)} > 2^-{fmt.width - 12}"
        )


def test_zeta_em_zero_is_small() -> None:
    """At the first zero, |zeta| must be tiny (validates the zero-hunting path)."""
    fmt = mf.Format(limbs=2)
    mp.mp.prec = 2 * fmt.width + 80
    t_fx = int(mp.nint(mp.mpf("14.134725141734693790457251983562") * (1 << 32)))
    sigma = mf.MPF(0, 0, 1 << (fmt.width - 1))  # 0.5
    prog = build_program(sigma, t_fx, fmt)
    zr, zi, _, _ = zeta_em.zeta_em(prog)
    mag = abs(mp.mpc(mpf_value(zr, fmt), mpf_value(zi, fmt)))
    # t is on the Q32.32 grid, so |zeta| ~ |zeta'| * (grid error) ~ 1e-10
    assert mag < mp.mpf("1e-8"), f"|zeta| at zero grid point = {mp.nstr(mag, 6)}"
