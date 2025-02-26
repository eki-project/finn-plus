#!/usr/bin/python3
import argparse
import shutil
import os
import subprocess
import sys
import psutil
from termcolor import colored

def get_repo(url, target, commit, depspath):
    targetpath = os.path.join(depspath, target)
    if os.path.isdir(targetpath):
        print(colored(f"Repository {target} already downloaded. Skipping...", "yellow"))
        return

    with open("install_log", 'w+') as f:
        subprocess.run(f"git clone {url} {targetpath};cd {targetpath};git pull;git checkout {commit}", shell=True, stdout=f, stderr=f)

    if os.path.isdir(targetpath):
        print(colored("Successfully installed ", "green") + colored(f"{target}", "green", "on_black"))
        os.remove("install_log")
    else:
        print(colored(f"Failed installing {target}. Log can be found in \"install_log\". Stopping...", "red", "on_white"))
        sys.exit(1)


def get_boardfiles(url, target, commit, loc, depspath):
    # Where all boardfiles are directly stored
    boardfiles = os.path.join(depspath, "board_files")

    # Where we clone to beforehand
    clonepath = os.path.join(depspath, target)

    # Where the boardfiles lay in the cloned dir
    locpath = os.path.join(clonepath, loc)

    if not os.path.isdir(boardfiles):
        os.mkdir(boardfiles)

    exists = False
    if os.path.isdir(clonepath):
        exists = True
        for f in os.listdir(locpath): 
            if f in [".git", "README.md"]:
                continue
            if os.path.isfile(os.path.join(boardfiles, f)) or os.path.isdir(os.path.join(boardfiles, f)):
                continue
            else:
                print(colored(f"Did not find all boardfiles required in {boardfiles} from {locpath}. Reinstalling...", "yellow"))
                exists = False
                break
        if exists:
            print(colored(f"Repository {target} already downloaded. Skipping...", "yellow"))
            return

    with open("install_log", 'w+') as f_log:
        subprocess.run(f"git clone {url} {clonepath};cd {clonepath};git pull;git checkout {commit};cd {loc};cp -r * {boardfiles}", shell=True, stdout=f_log, stderr=f_log)

    if os.path.isdir(clonepath):  
        for f in os.listdir(locpath): 
            if f in [".git", "README.md"]:
                continue
            if os.path.isfile(os.path.join(boardfiles, f)) or os.path.isdir(os.path.join(boardfiles, f)):
                continue
            else:
                print(colored(f"Failed installing {target}, could not find {f} in {boardfiles}. Log can be found in \"install_log\". Stopping...", "red", "on_white"))
                sys.exit(1)
        print(colored("Successfully installed ", "green") + colored(f"{target}", "green", "on_black"))
        os.remove("install_log")
    else:
        print(colored(f"Failed properly cloning {target}. Log can be found in \"install_log\". Stopping...", "red", "on_white"))
        sys.exit(1)


def install_verilator():
    # TODO: MAKE TEMPORARY ADDITION TO PATH
    with open("install_log", 'w+') as f:
        command = ""
        if not os.path.isdir(os.path.join(os.path.dirname(__file__), "verilator")):
            command += "git clone https://github.com/verilator/verilator && "
        subprocess.run(
            command + "cd verilator && git checkout v4.224 && autoconf && ./configure && make -j4",
            shell=True,
            stdout=f,
            stderr=f
        )
        os.environ["PATH"] = os.environ["PATH"] 
        print(colored(f"Verilator was cloned and setup. Please install it by going to finn-plus/verilator and calling \"sudo make install\"! Afterwards, re-run finn.", "red"))

def preserve_env_vars(finn_specific_envvars):
    preserved_envvars = {}
    for envvar in finn_specific_envvars:
        if envvar in os.environ.keys():
            preserved_envvars[envvar] = os.environ[envvar]
        else:
            preserved_envvars[envvar] = None
    return preserved_envvars


def restore_env_vars(preserved_envvars):
    for envvar, val in preserved_envvars.items():
        if envvar is None:
            os.environ.pop(envvar)
        else:
            os.environ[envvar] = val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("buildfile", action="store")
    parser.add_argument("--local", "-l", help="Whether to store the deps locally (in $FINN_ROOT/deps). Useful for development. Otherwise everything is installed to $HOME/.finn/deps", action="store_true")
    parser.add_argument("--tempslocal", "-t", help="If passed, the FINN_BUILD_DIR and FINN_HOST_BUILD_DIR will be automatically set to the directory the buildfile resides in", action="store_true")
    parser.add_argument("--remove-temps", "-r", help="Whether to remove temporary files from FINN_BUILD_DIR and FINN_HOST_BUILD_DIR if the directories were already used before", action="store_true")
    args = parser.parse_args()

    # Create dependency location
    if args.local:
        DEPSPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "deps"))
        print(colored(f"Using local depdendencies ($FINN_ROOT/deps/): {DEPSPATH}", "magenta"))
    else:
        if "FINN_DEPS" in os.environ.keys():
            DEPSPATH = os.path.abspath(os.environ["FINN_DEPS"])
            print(colored(f"Using depdendency folder specified in FINN_DEPS env var: {DEPSPATH}", "magenta"))
        else:
            DEPSPATH = os.path.abspath(os.path.join(os.environ["HOME"], ".finn", "deps"))
            print(colored(f"Using fallback default depdendencies path ($HOME/.finn/deps/): {DEPSPATH}", "magenta"))
    if not os.path.isdir(DEPSPATH):
        print(colored(f"Dependency directory not existing, creating now..", "yellow"))
        os.system(f"mkdir -p {DEPSPATH}")


    if not os.path.isfile(args.buildfile):
        print(colored(f"Could not find build file at: {args.buildfile}", "red", "on_white"))
        sys.exit(1)

    # Saving FINN specific global env vars
    # TODO: In the future we shouldnt need this anymore
    print(colored("Checking and backing up FINN specific environment variables...", "cyan"))
    finn_specific_envvars = [
        "FINN_ROOT",
        "LIVENESS_THRESHOLD",
    ]
    preserved_envvars = preserve_env_vars(finn_specific_envvars)

    # Now that we preserved env vars we need to be able to reconstruct even if the user sends Ctrl+C
    import signal
    def sigint_handler(sig, frame):
        print(colored(f"Restoring environment variables before exiting...", "red"))
        restore_env_vars(preserved_envvars)
        sys.exit(1)
    signal.signal(signal.SIGINT, sigint_handler)

    # Check all env variables
    print(colored("Checking environment variables", "cyan"))
    required_set = {
        "FINN_ROOT": os.path.dirname(__file__),
        "FINN_BUILD_DIR": "/tmp/FINN_TMP" if not args.tempslocal else os.path.join(os.path.dirname(args.buildfile), "FINN_TMP"),
        "PLATFORM_REPO_PATHS": "/opt/xilinx/platforms",
        "XRT_DEB_VERSION": "xrt_202220.2.14.354_22.04-amd64-xrt",
        "FINN_HOST_BUILD_DIR": "/tmp/FINN_TMP_HOST" if not args.tempslocal else os.path.join(os.path.dirname(args.buildfile), "FINN_TMP_HOST"),
        "OHMYXILINX": os.path.join(DEPSPATH, "oh-my-xilinx"),
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
        else:
            print(colored(f"Using existing environment variable {var}: {os.environ[var]}", "green"))
    FINN_ROOT = os.environ["FINN_ROOT"]
    FINN_BUILD_DIR = os.environ["FINN_BUILD_DIR"]
    FINN_HOST_DIR = os.environ["FINN_HOST_BUILD_DIR"]
    print(colored(f"FINN_ROOT set to: {FINN_ROOT}", "magenta"))
    print(colored(f"Storing temporary files in:", "magenta"))
    print(colored(f"\t{FINN_BUILD_DIR}", "magenta"))
    print(colored(f"\t{FINN_HOST_DIR}", "magenta"))

    if args.remove_temps:
        if os.path.isdir(FINN_BUILD_DIR):
            print(colored(f"Clearing old files from FINN_BUILD_DIR...", "yellow"))
            for f in os.listdir(FINN_BUILD_DIR):
                os.system("rm -r " + os.path.join(FINN_BUILD_DIR, f))
        if os.path.isdir(FINN_HOST_DIR):
            print(colored(f"Clearing old files from FINN_HOST_BUILD_DIR...", "yellow"))
            for f in os.listdir(FINN_HOST_DIR):
                os.systeme("rm -r " + os.path.join(FINN_HOST_DIR, f))


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
        get_repo(url, target, commit, DEPSPATH)



    # Boardfiles
    boardfiles = {
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
    print(colored("Checking boardfiles...", "cyan"))
    for target, data in boardfiles.items():
        url, commit, loc = data
        print(colored(f"Checking {target}...", "cyan"))
        get_boardfiles(url, target, commit, loc, DEPSPATH)


    # Install verilator
    print(colored("Checking verilator...", "cyan"))
    if shutil.which("verilator") is None:
        print(colored("Verilator not found, installing...", "cyan"))
        install_verilator()
        if shutil.which("verilator") is None:
            print(colored(f"Verilator not found after attempted installation. Look into \"install_log\" for further info!", "red", "on_white"))
            sys.exit(1)
        print(colored("Successfully installed ", "green") + colored("verilator", "green", "on_black"))
    else:
        print(colored("Verilator found!", "green"))


    # Run the given dataflow
    print(colored("Running FINN", "cyan"))
    directory = os.path.dirname(os.path.abspath(args.buildfile))
    buildfile = os.path.basename(args.buildfile)
    subprocess.run(f"python {buildfile}", cwd=directory, shell=True)

    # Restore FINN specific env vars
    print(colored("Restoring FINN run specific environment variables...", "cyan"))
