# Dependencies for finn
import os
from pathlib import Path
import subprocess

FINN_DEPS = {
    "finn-experimental": (
        "https://github.com/Xilinx/finn-experimental.git",
        "0724be21111a21f0d81a072fccc1c446e053f851"
    ),
    "brevitas": (
        "https://github.com/Xilinx/brevitas.git",
        "84f42259ec869eb151af4cb8a8b23ad925f493db"
    ),
    "cnpy": (
        "https://github.com/rogersce/cnpy.git",
        "4e8810b1a8637695171ed346ce68f6984e585ef4"
    ),
    "oh-my-xilinx": (
        "https://github.com/maltanar/oh-my-xilinx.git",
        "0b59762f9e4c4f7e5aa535ee9bc29f292434ca7a"
    ),
    "finn-hlslib": (
        "https://github.com/Xilinx/finn-hlslib.git",
        "16e5847a5e3ef76cffe84c8fad2f010d593457d3"
    )
}

FINN_BOARDFILES = {
    "avnet-bdf": (
        "https://github.com/Avnet/bdf.git",
        "2d49cfc25766f07792c0b314489f21fe916b639b",
        "."
    ),
    "xil-bdf": (
        "https://github.com/Xilinx/XilinxBoardStore.git",
        "8cf4bb674a919ac34e3d99d8d71a9e60af93d14e",
        os.path.join("boards", "Xilinx", "rfsoc2x2")
    ),
    "rfsoc4x2-bdf": (
        "https://github.com/RealDigitalOrg/RFSoC4x2-BSP.git",
        "13fb6f6c02c7dfd7e4b336b18b959ad5115db696",
        os.path.join("board_files", "rfsoc4x2")
    ),
    "kv260-som-bdf": (
        "https://github.com/Xilinx/XilinxBoardStore.git",
        "98e0d3efc901f0b974006bc4370c2a7ad8856c79",
        os.path.join("boards", "Xilinx", "kv260_som")
    ),
}

def pull_dep(name: str, repo: str, commit: str, deps_path: Path) -> tuple[str, bool]:
    """Pull a dependency and return on whether it was successful. If it wasnt, also return an error message"""
    target_directory = deps_path / name
    with open("install_log", 'w+') as f:
        existed = False
        if target_directory.exists():
            existed = True
            subprocess.run(f"git pull; git checkout {commit}", shell=True, stdout=f, stderr=f, cwd=target_directory)
        else:
            subprocess.run(f"git clone {repo} {target_directory};cd {target_directory};git pull;git checkout {commit}", shell=True, stdout=f, stderr=f)

    if target_directory.exists():
        # TODO: Check commit hash
        if existed:
            return "[bold green]Dependency updated[/bold green]", True
        else:
            return "[bold green]Dependency installed[/bold green]", True
    else:
        return "[bold red]Failed installing dependency![/bold red]", False
    


def pull_boardfile(name: str, repo: str, commit: str, dir_to_copy: Path, deps_path: Path) -> tuple[str, bool]:
    """Pull boardfiles, then copy the directories required into the destined board_files dir"""
    target_path = deps_path / name
    copy_from = deps_path / name / dir_to_copy
    copy_to = deps_path / "board_files" / copy_from.name
    with open("install_log", 'w+') as f:
        exists = False
        
        # Step 1: Pull
        if target_path.exists() and copy_from.exists():
            exists = True
            subprocess.run(f"git pull; git checkout {commit}", shell=True, cwd=target_path, stdout=f, stderr=f)
        else:
            subprocess.run(f"git clone {repo} {target_path};cd {target_path};git checkout {commit}", shell=True, stdout=f, stderr=f)
        
        # Step 2: Copy
        if copy_from != target_path:
            subprocess.run(f"cp -r {copy_from} {copy_to}", shell=True, stdout=f, stderr=f)
        else:
            subprocess.run(f"cp -r {copy_from}/* {copy_to}", shell=True, stdout=f, stderr=f)
    
    # TODO Checks
    if exists:
        return "[bold green]Boardfiles updated[/bold green]", True
    else:
        return "[bold green]Boardfiles installed[/bold green]", True


def deps_exist(path: Path | None = None) -> tuple[bool, list[tuple[str, Path]]]:
    """Check whether all dependencies exist. If not all exist, returns list of tuples with name and path where they were expected"""
    if path is None:
        check_path = Path.home() / ".finn" / "deps"
    else:
        check_path = path
    
    dep_not_found = []
    for name in FINN_DEPS.keys():
        current = check_path / name
        if not current.exists():
            dep_not_found.append((name, current))
    
    for name, val in FINN_BOARDFILES.items():
        if name != "avnet-bdf":
            _, _, loc = val
            current = check_path / "board_files" / Path(loc).name
            if not current.exists():
                dep_not_found.append((name, current))
        else:
            if not (check_path / name).exists():
                dep_not_found.append((name, check_path / name))
                continue

            avnet_dir = check_path / name
            for filename in avnet_dir.iterdir():
                if not (check_path / "board_files" / filename).exists():
                    dep_not_found.append((name, check_path / "board_files" / filename))
                    continue
    
    if len(dep_not_found) > 0:
        return False, dep_not_found
    else:
        return True, []
    
