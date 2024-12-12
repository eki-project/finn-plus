#!/usr/bin/python3
import argparse
import os
import subprocess
import sys
import psutil
from termcolor import colored

def get_repo(url, target, commit):
    targetpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deps", target)
    if os.path.isdir(targetpath):
        print(colored(f"Repository {target} already downloaded. Skipping...", "yellow"))
        return

    with open("install_log", 'w+') as f:
        subprocess.run(f"git clone {url} {targetpath};cd {targetpath};git pull;git checkout {commit}", cwd=os.path.dirname(os.path.abspath(__file__)), shell=True, stdout=f, stderr=f)

    if os.path.isdir(targetpath):
        print(colored("Successfully installed ", "green") + colored(f"{target}", "green", "on_black"))
        os.remove("install_log")
    else:
        print(colored(f"Failed installing {target}. Log can be found in \"install_log\". Stopping...", "red", "on_white"))
        sys.exit(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("buildfile", action="store")
    args = parser.parse_args()


    if not os.path.isfile(args.buildfile):
        print(colored(f"Could not find build file at: {args.buildfile}", "red", "on_white"))
        sys.exit(1)

    depsdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deps")
    if not os.path.isdir(depsdir):
        print(colored(f"Did not find deps directory. Creating...", "yellow"))
        os.mkdir(depsdir)


    # Check all env variables
    print(colored("Checking environment variables", "cyan"))
    required_set = {
        "FINN_BUILD_DIR": "/tmp/FINN_TMP",
        "PLATFORM_REPO_PATHS": "/opt/xilinx/platforms",
        "XRT_DEB_VERSION": "xrt_202220.2.14.354_22.04-amd64-xrt",
        "FINN_HOST_BUILD_DIR": "/tmp/FINN_TMP_HOST",
        "OHMYXILINX": os.path.join(os.path.abspath(os.getcwd()), "deps", "oh-my-xilinx"),
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



    # Installing dependencies if necessary
    # (Python deps are already installed by poetry)
    deps = {
        "finn-experimental": (
            "https://github.com/Xilinx/finn-experimental.git",
            "0724be21111a21f0d81a072fccc1c446e053f851"
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
    print(colored("Checking dependencies...", "cyan"))
    for target, data in deps.items():
        url, commit = data
        print(colored(f"Checking {target}...", "cyan"))
        get_repo(url, target, commit)



    # Boardfiles
    boardfiles = {
        "avnet-bdf": (
            "https://github.com/Avnet/bdf.git",
            "2d49cfc25766f07792c0b314489f21fe916b639b"
        ),
        "xil-bdf": (
            
        )
    }
    print(colored("Checking boardfiles...", "cyan"))



    # Install verilator
    print(colored("Checking verilator...", "cyan"))


    # Run the given dataflow
    poetry_python = subprocess.run("poetry env info -e", shell=True, stdout=subprocess.PIPE).stdout.decode("utf-8").replace("\n", "")
    directory = os.path.dirname(os.path.abspath(args.buildfile))
    buildfile = os.path.basename(args.buildfile)
    subprocess.run(f"{poetry_python} {buildfile}", cwd=directory, shell=True)
