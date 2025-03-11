"""Manage dependencies. Called by run_finn.py"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess as sp
from interface_globals import IS_POSIX
from pathlib import Path

# Tuple that defines a dep status
# Example: ("oh-my-xilinx", False, "Wrong commit")
Status = tuple[str, bool, str]

FINN_DEPS = {
    "finn-experimental": (
        "https://github.com/Xilinx/finn-experimental.git",
        "0724be21111a21f0d81a072fccc1c446e053f851",
    ),
    "brevitas": (
        "https://github.com/Xilinx/brevitas.git",
        "84f42259ec869eb151af4cb8a8b23ad925f493db",
    ),
    "cnpy": ("https://github.com/rogersce/cnpy.git", "4e8810b1a8637695171ed346ce68f6984e585ef4"),
    "oh-my-xilinx": (
        "https://github.com/maltanar/oh-my-xilinx.git",
        "0b59762f9e4c4f7e5aa535ee9bc29f292434ca7a",
    ),
    "finn-hlslib": (
        "https://github.com/Xilinx/finn-hlslib.git",
        "16e5847a5e3ef76cffe84c8fad2f010d593457d3",
    ),
    "attention-hlslib": (
        "https://github.com/iksnagreb/attention-hlslib.git",
        "afc9720f10e551e1f734e137b21bb6d0a8342177",
    ),
}

VERILATOR = ("https://github.com/verilator/verilator", "v4.224")

FINN_BOARDFILES = {
    "avnet-bdf": (
        "https://github.com/Avnet/bdf.git",
        "2d49cfc25766f07792c0b314489f21fe916b639b",
        Path(),
    ),
    "xil-bdf": (
        "https://github.com/Xilinx/XilinxBoardStore.git",
        "8cf4bb674a919ac34e3d99d8d71a9e60af93d14e",
        Path("boards/Xilinx/rfsoc2x2"),
    ),
    "rfsoc4x2-bdf": (
        "https://github.com/RealDigitalOrg/RFSoC4x2-BSP.git",
        "13fb6f6c02c7dfd7e4b336b18b959ad5115db696",
        Path("board_files/rfsoc4x2"),
    ),
    "kv260-som-bdf": (
        "https://github.com/Xilinx/XilinxBoardStore.git",
        "98e0d3efc901f0b974006bc4370c2a7ad8856c79",
        Path("boards/Xilinx/kv260_som"),
    ),
}


def check_commit(repo: Path, commit: str) -> tuple[bool, str]:
    """Return if the given repo has the correct commit and what commit it read"""
    result = sp.run("git rev-parse HEAD", text=True, capture_output=True, shell=True, cwd=str(repo))
    return result.stdout.strip() == commit, result.stdout.strip()


def run_silent(s: str, loc: str | None | Path) -> None:
    """Run a command silently directly without shell"""
    sp.run(shlex.split(s, posix=IS_POSIX), cwd=loc, stdout=sp.DEVNULL, stderr=sp.DEVNULL)


def update_dependencies(location: Path) -> list[Status]:
    """Update dependencies at the given path. Returns a list of status
    reports for the main script to display."""

    if not location.exists():
        location.mkdir(parents=True)
    status = []
    for pkg_name, (giturl, commit) in FINN_DEPS.items():
        target = (location / pkg_name).absolute()
        if target.exists():
            run_silent("git pull", target)
        else:
            run_silent(f"git clone {giturl} {target}", None)
        run_silent(f"git checkout {commit}", target)
        success, read_commit = check_commit(target, commit)
        status.append(
            (
                pkg_name,
                success,
                "Update successfull!"
                if success
                else f"Failed. Got commit {read_commit}, expected {commit}",
            )
        )
    for pkg_name, (giturl, commit, copy_from_here) in FINN_BOARDFILES.items():
        clone_location = location / pkg_name
        copy_source = clone_location / copy_from_here
        copy_target = location / "board_files" / copy_source.name
        if clone_location.exists():
            run_silent("git pull", clone_location)
        else:
            run_silent(f"git clone {giturl} {clone_location}", None)
        run_silent(f"git checkout {commit}", clone_location)
        if copy_source != clone_location:
            shutil.copytree(copy_source, copy_target, dirs_exist_ok=True)
        else:
            run_silent(f"cp -r {copy_source}/* {copy_target}", None)
        success, read_commit = check_commit(clone_location, commit)
        status.append(
            (
                pkg_name,
                success,
                "Update successfull!"
                if success
                else f"Failed. Got commit {read_commit}, expected {commit}",
            )
        )
    return status


def check_verilator_version() -> str | None:
    """Return the verilator version that is found. If no verilator version is installed or the
    version output cannot be parsed returns None"""
    if shutil.which("verilator") is None:
        return None
    result = sp.run("verilator --version", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        return result.stdout.split(" ")[1]
    except (IndexError, AttributeError):
        return None


def try_install_verilator(location: Path) -> Status:
    """Try installing verilator at the given location. If a working version is already in scope,
    return."""
    existing_verilator = check_verilator_version()
    if existing_verilator is not None and existing_verilator >= "4.224":
        return ("verilator", True, "Existing verilator version found!")
    verilator_git, verilator_checkout = VERILATOR
    target = (location / "verilator").absolute()
    configure_script = target / "configure"
    if "VERILATOR_ROOT" in os.environ.keys():
        del os.environ["VERILATOR_ROOT"]
    if not target.exists() or not configure_script.exists():
        run_silent(f"git clone {verilator_git} {target}", None)
        run_silent(f"git checkout {verilator_checkout}", target)
    res1 = sp.run(["autoconf"], cwd=target, capture_output=True, text=True)
    os.environ["VERILATOR_ROOT"] = str(target)
    res2 = sp.run(
        shlex.split("./configure", posix=IS_POSIX), cwd=target, capture_output=True, text=True
    )
    res3 = sp.run(["make"], cwd=target, capture_output=True, text=True)
    err = None
    if res1.returncode != 0 or res2.returncode != 0 or res3.returncode != 0:
        del os.environ["VERILATOR_ROOT"]
    if res1.returncode != 0:
        err = res1.stderr.split("\n")[-1]
    elif res2.returncode != 0:
        err = res2.stderr.split("\n")[-2]
    elif res3.returncode != 0:
        err = res3.stderr.split("\n")[-1]
    if err is not None:
        return ("verilator", False, f"{err}")
    os.environ["VERILATOR_ROOT"] = str(target)
    os.environ["PATH"] = f"{target}/bin:" + os.environ["PATH"]
    return ("verilator", True, f"Configured at {target}. Envvar set.")
