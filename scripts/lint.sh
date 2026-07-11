#!/usr/bin/env bash
# Lint everything: RTL (verilator, verible if present) + Python (ruff, mypy).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== verilator --lint-only =="
# -Wno-MULTITOP: the filelist is a library of independent blocks, many tops by design.
verilator --lint-only -Wall -Wno-MULTITOP -f rtl/filelists/generic-sim.f

if command -v verible-verilog-lint >/dev/null 2>&1; then
    echo "== verible-verilog-lint =="
    find rtl -name '*.sv' -print0 | xargs -0 verible-verilog-lint
else
    echo "== verible not installed — skipping (optional) =="
fi

echo "== ruff check =="
uv run ruff check .
echo "== ruff format --check =="
uv run ruff format --check .
echo "== mypy =="
uv run mypy

echo "lint OK"
