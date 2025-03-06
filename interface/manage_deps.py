"""Manage dependencies. Called by run_finn.py"""
import subprocess as sp
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
}

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


def update_dependencies(location: Path) -> list[Status]:
    """Update dependencies at the given path. Returns a list of status
    reports for the main script to display."""
    if not location.exists():
        location.mkdir(parents=True)
    status = []
    for pkg_name, (giturl, commit) in FINN_DEPS.items():
        target = (location / pkg_name).absolute()
        if target.exists():
            sp.run(
                f"git pull;git checkout {commit}",
                shell=True,
                cwd=target,
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
            )
        else:
            sp.run(
                f"git clone {giturl} {target};cd {target};git checkout {commit}",
                shell=True,
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
            )
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
            sp.run(
                f"git pull; git checkout {commit}",
                shell=True,
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
                cwd=clone_location,
            )
        else:
            sp.run(
                f"git clone {giturl} {clone_location};cd {clone_location};git checkout {commit}",
                shell=True,
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
            )
        if copy_source != clone_location:
            sp.run(
                f"cp -r {copy_source} {copy_target}",
                shell=True,
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
            )
        else:
            sp.run(
                f"cp -r {copy_source}/* {copy_target}",
                shell=True,
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
            )
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
