# fpga-zeta

Hardware-optimized, parametric SystemVerilog libraries to compute Riemann-zeta-related
functions on FPGAs, driven by a host CPU over PCIe.

- **Phase 1**: full pipeline computing ζ(s) for any complex s (Euler–Maclaurin engine),
  developed simulation-first (Verilator + cocotb) with bit-true Python golden models
  verified against mpmath/Arb.
- **Phase 2**: Riemann–Siegel Z(t) engine, FFT multi-evaluation, PCIe bring-up, and
  massively parallel multi-FPGA operation (zero hunting, real-time visualization).

## Layout

| Path | Purpose |
|---|---|
| `rtl/common/` | Portable SystemVerilog (no vendor primitives) |
| `rtl/arch/` | Architecture-specific tiles (`generic-sim`, `xilinx-usplus`, …) |
| `rtl/filelists/` | Build filelists selecting the tile implementations |
| `host/zetafpga/` | Typed Python package: golden models, oracles, kernel builder, drivers, apps |
| `tb/` | cocotb testbenches mirroring `rtl/common/` |
| `tools/` | Coefficient/ROM generators (Sollya or mpmath-Remez) |
| `docs` | `ARCHITECTURE.md`, `DESIGN.md` — living documents |

## Quick start

```sh
uv sync            # install Python deps (needs uv + Python >= 3.11)
make lint          # Verilator lint + ruff + mypy
make test          # cocotb tests under Verilator + Python unit tests
```

Requires [Verilator](https://verilator.org) >= 5.0 on PATH.

## Status

M0 (scaffold & toolchain) — see `ARCHITECTURE.md` for the milestone plan.
