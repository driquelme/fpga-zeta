"""Shared pytest→cocotb→Verilator runner helper.

Every block test calls run_block() from a plain pytest function; the cocotb
coroutines in the same file are executed inside the simulator. RTL parameters
are forwarded to Verilator and exported to the cocotb process as ZETA_<name>
environment variables so the test coroutines can mirror the configuration.
"""

import os
from pathlib import Path


def run_block(
    source: str | list[str],
    toplevel: str,
    test_file: str,
    parameters: dict[str, int | str] | None = None,
) -> None:
    try:
        from cocotb_tools.runner import get_runner
    except ImportError:  # cocotb < 2.0
        from cocotb.runner import get_runner

    params = parameters or {}
    repo = Path(__file__).resolve().parents[2]
    # Only scalar parameters name the build dir (string params may be paths).
    tag = "_".join(f"{k}{v}" for k, v in sorted(params.items()) if isinstance(v, int)) or "default"
    build_dir = repo / "sim_build" / f"{toplevel}_{tag}"
    test_path = Path(test_file).resolve()

    os.environ["PYTHONPATH"] = f"{test_path.parent}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    for name, value in params.items():
        os.environ[f"ZETA_{name}"] = str(value)

    sources = [source] if isinstance(source, str) else source
    runner = get_runner("verilator")
    runner.build(
        sources=[repo / s for s in sources],
        hdl_toplevel=toplevel,
        build_dir=str(build_dir),
        parameters=params,
        build_args=(["--trace-fst"] if os.getenv("WAVES") else []),
    )
    runner.test(
        hdl_toplevel=toplevel,
        test_module=test_path.stem,
        build_dir=str(build_dir),
    )
