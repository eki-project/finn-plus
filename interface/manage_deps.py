"""Manage dependencies. Called by run_finn.py"""
from __future__ import annotations

import concurrent
import concurrent.futures
import os
import shlex
import shutil
import subprocess as sp
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rich.console import Console
from rich.live import Live
from rich.table import Table
from threading import Lock

from interface import IS_POSIX

FINN_DEPS = {
    "finn-experimental": (
        "https://github.com/Xilinx/finn-experimental.git",
        "0724be21111a21f0d81a072fccc1c446e053f851",
    ),
    "brevitas": (
        "https://github.com/iksnagreb/brevitas.git",
        "003f9f4070c20639790c7b406a28612a089fc502",
    ),
    "cnpy": ("https://github.com/rogersce/cnpy.git", "4e8810b1a8637695171ed346ce68f6984e585ef4"),
    "oh-my-xilinx": (
        "https://github.com/maltanar/oh-my-xilinx.git",
        "0b59762f9e4c4f7e5aa535ee9bc29f292434ca7a",
    ),
    "finn-hlslib": (
        "https://github.com/Xilinx/finn-hlslib.git",
        "5c5ad631e3602a8dd5bd3399a016477a407d6ee7",
    ),
    "attention-hlslib": (
        "https://github.com/iksnagreb/attention-hlslib.git",
        "afc9720f10e551e1f734e137b21bb6d0a8342177",
    ),
    "pyxsi": (
        "https://github.com/maltanar/pyxsi.git",
        "18fa7a7e613f826a66b43298d7e8fa0ec9035a03",
    ),
}

VERILATOR = ("https://github.com/verilator/verilator", "v4.224")
REQUIRED_VERILATOR_VERSION = VERILATOR[1][1:]  # Remove the "v" from v4.224

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

# URL, do_unzip, where to download to
DIRECT_DOWNLOAD_DEPS = {
    "pynq-z1": (
        "https://github.com/cathalmccabe/pynq-z1_board_files/raw/master/pynq-z1.zip",
        True,
        Path("board_files"),
    ),
    "pynq-z2": (
        "https://dpoauwgwqsy2x.cloudfront.net/Download/pynq-z2.zip",
        True,
        Path("board_files"),
    ),
}

# TODO: Change or make it configurable
GIT_CLONE_TIMEOUT = 120


# Tuple that defines a dep status
# Example: ("oh-my-xilinx", False, "Wrong commit")
Status = tuple[str, bool, str]


def check_commit(repo: Path, commit: str) -> tuple[bool, str]:
    """Return if the given repo has the correct commit and what commit it read"""
    result = sp.run("git rev-parse HEAD", text=True, capture_output=True, shell=True, cwd=str(repo))
    return result.stdout.strip() == commit, result.stdout.strip()


def run_silent(s: str, loc: str | None | Path, timeout: int | None = None) -> None:
    """Run a command silently directly without shell"""
    sp.run(
        shlex.split(s, posix=IS_POSIX),
        cwd=loc,
        stdout=sp.DEVNULL,
        stderr=sp.DEVNULL,
        stdin=sp.DEVNULL,
        timeout=timeout,
    )


def make_status_table(data: dict[str, tuple[str, str]]) -> Table:
    """Get a dict of the form data[name] = (status, color) and convert it into a table"""
    t = Table()
    t.add_column("Name")
    t.add_column("Status")
    for name, (status, color) in data.items():
        t.add_row(name, f"[bold {color}]{status}[/bold {color}]")
    return t


def update_dependencies(location: Path) -> None:
    """Update dependencies at the given path. Display live status"""
    if not location.exists():
        location.mkdir(parents=True)
    board_file_dir = location / "board_files"
    if not board_file_dir.exists():
        board_file_dir.mkdir(parents=True)
    current_state = {}
    state_lock = Lock()
    any_failed = False
    with Live(make_status_table(current_state)) as live:

        def update_status(key: str, msg: str, color: str) -> None:
            state_lock.acquire()
            current_state[key] = (msg, color)
            live.update(make_status_table(current_state))
            state_lock.release()

        def pull_data(args: tuple) -> bool:
            name, url, do_unzip, target = args
            purl = Path(url)
            target = location.absolute() / target
            update_status(name, "Downloading data...", "orange1")
            if shutil.which("wget") is None:
                update_status(name, "wget not found - could not download data!", "red")
                return False
            if not (target / purl.name).exists():
                result = sp.run(
                    shlex.split(f"wget {url}", posix=IS_POSIX),
                    capture_output=True,
                    text=True,
                    cwd=target,
                )
                if result.returncode != 0:
                    update_status(name, f"Download failed: {result.stderr}", "red")
                    return False
            if do_unzip and not (target / purl.name.replace(purl.suffix, "")).exists():
                update_status(name, "Unzipping data...", "orange1")
                run_silent(f"unzip {purl.name}", target)
            update_status(name, "Dependency ready!", "green")
            return True

        def pull_dep(args: tuple) -> bool:
            pkg_name, giturl, commit = args
            target = (location / pkg_name).absolute()
            update_status(pkg_name, "Pulling data...", "orange1")
            if target.exists():
                run_silent("git pull", target)
            else:
                run_silent(f"git clone {giturl} {target}", None, timeout=GIT_CLONE_TIMEOUT)
            if not target.exists():
                update_status(pkg_name, "Bad Git URL or missing network connection", "red")
                return False
            update_status(pkg_name, "Checking out commit...", "orange1")
            run_silent(f"git checkout {commit}", target)
            success, read_commit = check_commit(target, commit)
            if not success:
                update_status(pkg_name, "Failed pulling. Retrying...", "orange1")
                shutil.rmtree(target, ignore_errors=True)
                run_silent(f"git clone {giturl} {target}", None)
                run_silent(f"git checkout {commit}", target)
                success, read_commit = check_commit(target, commit)
            if success:
                update_status(pkg_name, "Dependency ready!", "green")
            else:
                update_status(
                    pkg_name,
                    f"Installation failed! Expected commit {commit} but got {read_commit}",
                    "red",
                )
                return False
            return True

        def pull_board(args: tuple) -> bool:
            pkg_name, giturl, commit, copy_from_here = args
            clone_location = location / pkg_name
            copy_source = clone_location / copy_from_here
            copy_target = location / "board_files" / copy_source.name
            update_status(pkg_name, "Pulling data...", "orange1")
            if clone_location.exists():
                run_silent("git pull", clone_location)
            else:
                run_silent(f"git clone {giturl} {clone_location}", None, timeout=GIT_CLONE_TIMEOUT)
            if not clone_location.exists():
                update_status(pkg_name, "Bad Git URL or missing network connection", "red")
                return False
            update_status(pkg_name, "Checking out commit...", "orange1")
            run_silent(f"git checkout {commit}", clone_location)
            success, read_commit = check_commit(clone_location, commit)
            if not success:
                update_status(pkg_name, "Failed pulling. Retrying...", "orange1")
                shutil.rmtree(clone_location, ignore_errors=True)
                run_silent(f"git clone {giturl} {clone_location}", None)
                run_silent(f"git checkout {commit}", clone_location)
                success, read_commit = check_commit(clone_location, commit)
            if success:
                update_status(pkg_name, "Copying boardfiles over...", "orange1")
            else:
                update_status(
                    pkg_name,
                    f"Installation failed! " f"Expected commit {commit} but got {read_commit}",
                    "red",
                )
                return False
            if copy_source != clone_location:
                shutil.copytree(copy_source, copy_target, dirs_exist_ok=True)
            else:
                run_silent(f"cp -r {copy_source}/* {copy_target}", None)
            update_status(pkg_name, "Dependency ready!", "green")
            return True

        def try_install_verilator() -> bool:
            existing_verilator = check_verilator_version()
            if existing_verilator is not None and existing_verilator >= REQUIRED_VERILATOR_VERSION:
                update_status("verilator", "Found existing verilator version!", "green")
                return True
            verilator_git, verilator_checkout = VERILATOR
            target = (location / "verilator").absolute()
            configure_script = target / "configure"
            if "VERILATOR_ROOT" in os.environ.keys():
                del os.environ["VERILATOR_ROOT"]
            if not target.exists() or not configure_script.exists():
                update_status("verilator", "Pulling repository", "orange1")
                run_silent(f"git clone {verilator_git} {target}", None, timeout=GIT_CLONE_TIMEOUT)
                run_silent(f"git checkout {verilator_checkout}", target)
            if not target.exists():
                update_status("verilator", "Bad Git URL or missing network connection", "red")
                return False
            update_status("verilator", "Running autoconf...", "orange1")
            res1 = sp.run(["autoconf"], cwd=target, capture_output=True, text=True)
            os.environ["VERILATOR_ROOT"] = str(target)
            update_status("verilator", "Running configure...", "orange1")
            res2 = sp.run(
                shlex.split("./configure", posix=IS_POSIX),
                cwd=target,
                capture_output=True,
                text=True,
            )
            update_status("verilator", "Running make...", "orange1")
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
                update_status("verilator", f"Error found: {err}", "red")
                return False
            os.environ["VERILATOR_ROOT"] = str(target)
            os.environ["PATH"] = f"{target}/bin:" + os.environ["PATH"]
            update_status("verilator", "Verilator configured!", "green")
            return True

        with ThreadPoolExecutor(100) as tpe:
            futures = []
            for name, (giturl, commit) in FINN_DEPS.items():
                futures.append(tpe.submit(pull_dep, (name, giturl, commit)))
            for name, (giturl, commit, copy_from_here) in FINN_BOARDFILES.items():
                futures.append(tpe.submit(pull_board, (name, giturl, commit, copy_from_here)))
            for name, (url, do_unzip, target) in DIRECT_DOWNLOAD_DEPS.items():
                futures.append(tpe.submit(pull_data, (name, url, do_unzip, target)))
            futures.append(tpe.submit(try_install_verilator))
            for future in concurrent.futures.as_completed(futures):
                any_failed |= not future.result()

    if any_failed:
        Console().print(
            "[bold red]ERROR: [/bold red][red]"
            "Failed to retrieve all dependencies. Stopping...[/red]"
        )
        sys.exit(1)


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
