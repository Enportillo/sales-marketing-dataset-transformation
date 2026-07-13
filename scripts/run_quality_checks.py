"""Script utilitario para verificaciones de calidad del proyecto."""

from __future__ import annotations

import subprocess
import sys


COMMANDS = [
    [sys.executable, "-m", "py_compile", "main.py"],
    [sys.executable, "-m", "py_compile", "dashboard/app.py"],
    [sys.executable, "-m", "py_compile", "src/data_bootstrap.py"],
    [sys.executable, "-m", "py_compile", "src/etl_validation.py"],
    [sys.executable, "-m", "py_compile", "src/api/main.py"],
    [sys.executable, "tests/test_etl.py"],
    [sys.executable, "tests/test_api.py"],
]


def main() -> int:
    for cmd in COMMANDS:
        print("[RUN]", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print("[FAIL]", " ".join(cmd))
            return result.returncode
    print("[OK] Quality checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
