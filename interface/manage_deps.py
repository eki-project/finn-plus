"""Manage dependencies. Called by run_finn.py"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess as sp
from pathlib import Path

from interface.interface_globals import FINN_BOARDFILES, FINN_DEPS, IS_POSIX, VERILATOR

# Tuple that defines a dep status
# Example: ("oh-my-xilinx", False, "Wrong commit")
Status = tuple[str, bool, str]


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
        if not success:
            shutil.rmtree(target, ignore_errors=True)
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
        success, read_commit = check_commit(clone_location, commit)
        if not success:
            shutil.rmtree(clone_location, ignore_errors=True)
            run_silent(f"git clone {giturl} {clone_location}", None)
            run_silent(f"git checkout {commit}", clone_location)
            success, read_commit = check_commit(clone_location, commit)
        if copy_source != clone_location:
            shutil.copytree(copy_source, copy_target, dirs_exist_ok=True)
        else:
            run_silent(f"cp -r {copy_source}/* {copy_target}", None)
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
