import json
import os
import psutil
import yaml
from pathlib import Path

GLOBAL_FINN_ENVVARS = {
    "PLATFORM_REPO_PATHS": "/opt/xilinx/platforms",
    "XRT_DEB_VERSION": "xrt_202220.2.14.354_22.04-amd64-xrt",
    "XILINX_LOCAL_USER_DATA": "no",
    "VIVADO_PATH": "/tools/Xilinx/Vivado/2022.1",
    "VITIS_PATH": "/tools/Xilinx/Vitis/2022.1",
    "HLS_PATH": "/tools/Xilinx/Vitis_HLS/2022.1",
}


def generate_envvars(
    finnroot: Path, buildfile_path: Path, local_temps: bool, deps_path: Path, num_workers: int
):
    """Generate a string to prefix the bash command with the required env vars"""
    cpucount = psutil.cpu_count(logical=False)
    if num_workers == -1:
        cpus = int(0.75 * cpucount) if cpucount is not None else 1
    else:
        cpus = num_workers
    finnbuilddir = (
        buildfile_path.parent.absolute() / "FINN_TMP" if local_temps else Path("/tmp/FINN_TMP")
    )
    finnhost = finnbuilddir.parent / "FINN_TMP_HOST"
    ohmyxilinx = (deps_path / "oh-my-xilinx").absolute()
    return {
        "FINN_ROOT": finnroot,
        "NUM_DEFAULT_WORKERS": cpus,
        "FINN_BUILD_DIR": finnbuilddir,
        "OHMYXILINX": ohmyxilinx,
        "FINN_HOST_BUILD_DIR": finnhost,
    }


def set_envvars(envvars: dict) -> None:
    for k, v in envvars.items():
        os.environ[k] = str(v)


def make_envvar_prefix_str(envvars: dict) -> str:
    return " ".join([f"{k}={v}" for k, v in envvars.items()])


def load_preset_envvars(location: Path) -> bool:
    """Try to load existing environment variables. Return whether the operation failed
    TODO: This is temporary and should eventually be replaced a proper configuration"""
    if not location.exists():
        return False
    data = None
    if location.name.endswith("json"):
        with location.open() as f:
            data = json.load(f)
    elif location.name.endswith(("yaml", "yml")):
        with location.open() as f:
            data = yaml.load(f, Loader=yaml.Loader)
    else:
        return False
    for k, v in data.items():
        os.environ[k] = v
    return True
