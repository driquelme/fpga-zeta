"""Shared pytest configuration for the cocotb testbenches.

Parameterizes block tests over the standard configurations:
- `width` fixture: lzc-style raw-width blocks
- `limbs` fixture: limb-count-parametric mp/ blocks (Z64/Z128/Z256 mantissas)
"""

import pytest


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "width" in metafunc.fixturenames:
        metafunc.parametrize("width", [16, 64, 128])
    if "limbs" in metafunc.fixturenames:
        metafunc.parametrize("limbs", [1, 2, 4])
