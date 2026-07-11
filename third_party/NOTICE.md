# Third-party policy and study references

**Policy: no vendored RTL by default.** External designs are *study-only* references;
nothing is copied into this repository. Coefficient tables and RTL are generated from
scratch by `tools/`. Anything ever copied or ported must be recorded here with its
license, origin commit, and the files it touched.

## Study-only references (nothing copied)

| Project | License | What we learn from it |
|---|---|---|
| [APFP](https://github.com/spcl/apfp) | BSD-3 | Pipelined 512/1024-bit FP decomposition, Karatsuba over DSPs |
| [fpnew / CVFPU](https://github.com/openhwgroup/cvfpu) | Solderpad 0.51 | Package-driven SV FP format parameterization |
| [en_cl_fix](https://github.com/enclustra/en_cl_fix) | MIT | Fixed-point format algebra + bit-true Python golden-model pattern |
| [FloPoCo](https://flopoco.org/) | AGPL (generator) / LGPL (output) | Algorithm oracle: piecewise-poly evaluators, exp/log/sincos range reduction, DSP tiling. Used as coefficient oracle only — no generated VHDL is included |
| [ZipCPU cordic](https://github.com/ZipCPU/cordic) | GPLv3 gen / LGPLv3 output | CORDIC reference for low-precision cross-checks — no generated RTL included |
| [BaseJump STL](https://github.com/bespoke-silicon-group/basejump_stl) | BSD-3 | Latency-insensitive interface conventions |
| [verilog-pcie / cocotbext-pcie](https://github.com/alexforencich) | MIT | PCIe DMA substrate + simulated PCIe system (Phase 2) |
| [Arb/FLINT](https://arblib.org/), [mpmath](https://mpmath.org/) | LGPL / BSD | Mathematical oracles via Python bindings (runtime deps, not vendored) |
