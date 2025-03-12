import os
import shlex
import subprocess
from pathlib import Path

from interface.interface_globals import IS_POSIX


def run_test(variant: str, num_workers: str) -> None:
    """Run a given test variant with the given number of workers"""
    original_dir = Path.cwd()
    os.chdir(Path(__file__).parent.parent)
    match variant:
        case "quick":
            subprocess.run(
                shlex.split(
                    f"pytest -m 'not "
                    f"(vivado or slow or vitis or board or notebooks or bnn_pynq)' "
                    f"--dist=loadfile -n {num_workers}",
                    posix=IS_POSIX,
                )
            )
        case "main":
            subprocess.run(
                shlex.split(
                    f"pytest -k 'not (rtlsim or end2end)' --dist=loadfile -n {num_workers}",
                    posix=IS_POSIX,
                )
            )
        case "rtlsim":
            subprocess.run(shlex.split(f"pytest -k rtlsim --workers {num_workers}", posix=IS_POSIX))
        case "end2end":
            subprocess.run("pytest -k end2end", shell=True)
        case "full":
            subprocess.run(
                shlex.split(
                    f"pytest -k 'not (rtlsim or end2end)' --dist=loadfile -n {num_workers}",
                    posix=IS_POSIX,
                )
            )
            subprocess.run(shlex.split(f"pytest -k rtlsim --workers {num_workers}", posix=IS_POSIX))
            subprocess.run("pytest -k end2end", shell=True)
    os.chdir(original_dir)
