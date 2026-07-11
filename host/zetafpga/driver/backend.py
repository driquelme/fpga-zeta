"""The Backend protocol — the single host/device contract.

Every application talks to hardware (simulated or real) exclusively through this
4-method protocol, so app code runs unmodified from cocotb simulation to PCIe.
Implementations: sim_backend (cocotb, M8), verilator_backend (M9),
pcie_backend (Phase 2).
"""

from typing import Protocol


class Backend(Protocol):
    """Minimal host<->device transport contract."""

    def write(self, addr: int, data: bytes) -> None:
        """Write raw bytes to device address space."""
        ...

    def read(self, addr: int, length: int) -> bytes:
        """Read `length` bytes from device address space."""
        ...

    def submit(self, program: bytes) -> None:
        """Stream an assembled kernel program (descriptors + tables) to the engine."""
        ...

    def wait_idle(self, timeout_s: float) -> None:
        """Block until the engine drains its descriptor queue or raise TimeoutError."""
        ...
