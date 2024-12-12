#!/usr/bin/python3
import argparse
import os
import subprocess
import sys
import psutil
from termcolor import colored

parser = argparse.ArgumentParser()
parser.add_argument("buildfile", action="store")
args = parser.parse_args()


if not os.path.isfile(args.buildfile):
    print(f"Could not find build file at: {args.buildfile}")
    sys.exit(1)


# Check all env variables and tool installs like verilator
required_set = {
    "FINN_BUILD_DIR": "/tmp/FINN_TMP",
    "PLATFORM_REPO_PATHS": "/opt/xilinx/platforms",
    "XRT_DEB_VERSION": "xrt_202220.2.14.354_22.04-amd64-xrt",
    "FINN_HOST_BUILD_DIR": "/tmp/FINN_TMP_HOST",
    "OHMYXILINX": os.path.join(os.getcwd(), "deps", "oh-my-xilinx"),
    "NUM_DEFAULT_WORKERS": str(int(psutil.cpu_count(logical=False) * 0.75)),
    "XILINX_LOCAL_USER_DATA": "no",
    "VIVADO_PATH": "/tools/Xilinx/Vivado/2022.1",
    "VITIS_PATH": "/tools/Xilinx/Vitis/2022.1",
    "HLS_PATH": "/tools/Xilinx/Vitis_HLS/2022.1",
}

for var, value in required_set.items():
    if (var not in os.environ.keys()) or (os.environ[var] == ""):
        print(colored(f"Environment variable {var} not set or empty. Setting to default: ", "yellow") + colored(value, "light_red", "on_dark_grey"))
        os.environ[var] = value

# Run the given dataflow
directory = os.path.dirname(os.path.abspath(args.buildfile))
buildfile = os.path.basename(args.buildfile)
subprocess.run(f"python {buildfile}", cwd=directory, shell=True)