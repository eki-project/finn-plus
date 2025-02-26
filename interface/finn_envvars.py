from math import floor
import os
from pathlib import Path
import psutil

def required_envvars(buildfile_path: Path, local_temps: bool, depspath: Path) -> dict: 
    cpus = psutil.cpu_count(logical=False)
    if cpus is None:
        cpucount = 1
    else:
        cpucount = floor((0.75 * cpus))
    return {
        "FINN_ROOT": os.path.dirname(__file__),
        "FINN_BUILD_DIR": "/tmp/FINN_TMP" if not local_temps else buildfile_path.parent / "FINN_TMP",
        "PLATFORM_REPO_PATHS": "/opt/xilinx/platforms",
        "XRT_DEB_VERSION": "xrt_202220.2.14.354_22.04-amd64-xrt",
        "FINN_HOST_BUILD_DIR": "/tmp/FINN_TMP_HOST" if not local_temps else buildfile_path.parent /  "FINN_TMP_HOST",
        "OHMYXILINX": depspath / "oh-my-xilinx",
        "NUM_DEFAULT_WORKERS": str(cpucount),
        "XILINX_LOCAL_USER_DATA": "no",
        "VIVADO_PATH": "/tools/Xilinx/Vivado/2022.1",
        "VITIS_PATH": "/tools/Xilinx/Vitis/2022.1",
        "HLS_PATH": "/tools/Xilinx/Vitis_HLS/2022.1",
    }

def preserve_envvars(buildfile_path: Path, local_temps: bool, depspath: Path):
    # TODO
    pass


def set_missing_envvars(buildfile_path: Path, local_temps: bool, depspath: Path) -> list[tuple[str, str]]:
    """Set all missing environment variables. Does not store old ones. Return which ones were set how"""
    changed = []
    for varname, varcontent in required_envvars(buildfile_path, local_temps, depspath).items():
        if (varname not in os.environ.keys()) or (os.environ[varname] == ""):
            os.environ[varname] = varcontent
            changed.append((varname, varcontent))
    return changed