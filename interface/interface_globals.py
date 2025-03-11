import os
from pathlib import Path

DEFAULT_DEPS = Path.home() / ".finn" / "deps"
DEFAULT_FINN_ROOT = Path(__file__).parent.parent
DEFAULT_ENVVAR_CONFIG = Path.home() / ".finn" / "envvars.yaml"
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
IS_POSIX = os.name == "posix"
