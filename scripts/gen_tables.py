"""Regenerate all committed coefficient/ROM tables (deterministic)."""

import subprocess
import sys


def main() -> None:
    for script in (
        "tools/coeffgen/gen_sincos.py",
        "tools/coeffgen/gen_expln.py",
        "tools/coeffgen/gen_cexp.py",
    ):
        subprocess.run([sys.executable, script], check=True)


if __name__ == "__main__":
    main()
