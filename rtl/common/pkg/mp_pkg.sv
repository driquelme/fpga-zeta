// mp_pkg: the MPF format contract.
//
// SystemVerilog has no parameterized package types, so the format is a
// *layout convention* that every mp/ module implements with identical
// localparams (and the golden model mirrors in host/zetafpga/golden/mpfloat.py).
//
// Packed MPF word, MPW = WIDTH + EXPW + 3 bits, WIDTH = LIMBS*64:
//
//   [WIDTH-1        : 0]      mantissa, normalized (MSB=1) unless zero/special
//   [WIDTH+EXPW-1   : WIDTH]  exponent, two's complement
//   [WIDTH+EXPW]              sign
//   [WIDTH+EXPW+1]            is_zero
//   [WIDTH+EXPW+2]            is_special (inf/nan collapsed)
//
//   value = (-1)^sign * mant * 2^(exp - WIDTH)     (mantissa in [1/2, 1))
//
// Canonical zero/special words have mant = 0, exp = 0. Rounding is RNE.
// No subnormals; exponent overflow saturates to special (+ovf sticky flag),
// underflow to zero (+unf sticky flag). See DESIGN.md for rationale.
//
// The standard per-module localparam block (copy verbatim):
//
//   localparam int unsigned WIDTH  = LIMBS * LIMBW;
//   localparam int unsigned MPW    = WIDTH + EXPW + 3;
//   localparam int unsigned EXP_LO = WIDTH;
//   localparam int unsigned SIGN_B = WIDTH + EXPW;
//   localparam int unsigned ZERO_B = WIDTH + EXPW + 1;
//   localparam int unsigned SPEC_B = WIDTH + EXPW + 2;
package mp_pkg;
  // Exponent saturation bounds as functions of EXPW.
  function automatic int emax(input int unsigned expw);
    return (1 << (expw - 1)) - 1;
  endfunction
  function automatic int emin(input int unsigned expw);
    return -(1 << (expw - 1));
  endfunction
endpackage
