"Manage environment variables. Called by run_finn.py"
import os
import psutil
import yaml
from pathlib import Path

DEFAULT_FINN_TMP = Path("/tmp/FINN_TMP")
DEFAULT_FINN_TMP_HOST = Path("/tmp/FINN_TMP_HOST")

DEFAULT_GLOBAL_ENVVARS = {
    "PLATFORM_REPO_PATHS": "/opt/xilinx/platforms",
    "XRT_DEB_VERSION": "xrt_202220.2.14.354_22.04-amd64-xrt",
    "XILINX_LOCAL_USER_DATA": "no",
    "VIVADO_PATH": "/tools/Xilinx/Vivado/2022.1",
    "VITIS_PATH": "/tools/Xilinx/Vitis/2022.1",
    "HLS_PATH": "/tools/Xilinx/Vitis_HLS/2022.1",
}


def get_global_envvars(config_path: Path) -> dict[str, str]:
    """Get a dictionary of environment variables that are globally used.
    Precedence is: Set variables > Config > Default"""
    envvars = {}
    config_vars = {}
    if config_path.exists() and config_path.suffix in [".yml", ".yaml"]:
        with config_path.open() as f:
            config_vars = dict(yaml.load(f, yaml.Loader).items())
    for k, v in DEFAULT_GLOBAL_ENVVARS.items():
        if k in os.environ.keys():
            envvars[k] = os.environ[k]
        elif k in config_vars.keys():
            envvars[k] = config_vars[k]
        else:
            envvars[k] = v
    return envvars


def get_run_specific_envvars(
    deps: Path, config_path: Path, local_temps: bool, num_workers: int
) -> dict[str, str]:
    """Retun run specific environment variables"""
    cpucount = psutil.cpu_count(logical=False)
    if num_workers == -1:
        cpus = int(0.75 * cpucount) if cpucount is not None else 1
    else:
        cpus = num_workers
    return {
        "FINN_ROOT": Path(__file__).parent.parent,
        "NUM_DEFAULT_WORKERS": cpus,
        "OHMYXILINX": (deps / "oh-my-xilinx").absolute(),
        "FINN_BUILD_DIR": (config_path.parent / "FINN_TMP").absolute()
        if local_temps
        else DEFAULT_FINN_TMP,
        "FINN_HOST_BUILD_DIR": (config_path.parent / "FINN_TMP_HOST").absolute()
        if local_temps
        else DEFAULT_FINN_TMP_HOST,
    }
