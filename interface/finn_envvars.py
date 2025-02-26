from math import floor
import os
from pathlib import Path
import psutil

GLOBAL_FINN_ENVVARS = {
    "PLATFORM_REPO_PATHS": "/opt/xilinx/platforms",
    "XRT_DEB_VERSION": "xrt_202220.2.14.354_22.04-amd64-xrt",
    "XILINX_LOCAL_USER_DATA": "no",
    "VIVADO_PATH": "/tools/Xilinx/Vivado/2022.1",
    "VITIS_PATH": "/tools/Xilinx/Vitis/2022.1",
    "HLS_PATH": "/tools/Xilinx/Vitis_HLS/2022.1",
}


def generate_envvars(finnroot: Path, buildfile_path: Path, local_temps: bool, deps_path: Path, num_workers: int):
    """Generate a string to prefix the bash command with the required env vars"""
    cpucount = psutil.cpu_count(logical=False)
    if num_workers == -1:
        cpus = int(0.75 * cpucount) if cpucount is not None else 1
    else:
        cpus = num_workers
    finnbuilddir = buildfile_path.parent.absolute() / "FINN_TMP" if local_temps else Path("/tmp/FINN_TMP")
    finnhost = finnbuilddir.parent / "FINN_TMP_HOST"
    ohmyxilinx = (deps_path / "oh-my-xilinx").absolute()
    prefix = ""
    prefix += f"FINN_ROOT={finnroot} "
    prefix += f"NUM_DEFAULT_WORKERS={cpus} "
    prefix += f"FINN_BUILD_DIR={finnbuilddir} "
    prefix += f"OHMYXILINX={ohmyxilinx} "
    prefix += f"FINN_HOST_BUILD_DIR={finnhost}"
    return prefix