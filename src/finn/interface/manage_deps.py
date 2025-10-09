"""Manage dependencies. Called by run_finn.py."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess as sp
import sys
import yaml
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from threading import RLock
from typing import Literal

from finn.interface import IS_POSIX
from finn.interface.interface_utils import _resolve_module_path, debug, error
from finn.util.exception import FINNConfigurationError, FINNUserError


class _StatusTracker:
    """Small helper class to thread-safely organize status data."""

    def __init__(self, names_types: list[tuple[str, str]], live: Live) -> None:
        # Name: (type, status, color)
        self.data = {}
        self.live = live
        self.datalock = RLock()
        for name, typ in names_types:
            self.data[name] = (typ, "Initializing...", "grey70")
        self.total = len(self.data.keys())
        self.done = 0

    def update_status(self, name: str, status: str, color: str) -> None:
        """Update the status dict. If name doesnt exist, do nothing."""
        with self.datalock:
            if name in self.data:
                self.data[name] = (self.data[name][0], status, color)

    def _generate_renderable(self) -> Table:
        """Generate a renderable for rich to display in a live context."""
        with self.datalock:
            table = Table(
                title="Dependency Updates",
                caption=(
                    f"Installed: [cyan]{self.done}[/cyan] / "
                    f"[cyan bold]{self.total}[/cyan bold]."
                ),
                box=box.SIMPLE,
                expand=True,
            )
            table.add_column("Dependency", justify="center", style="italic aquamarine3")
            table.add_column("Dependency Type", justify="center")
            table.add_column("Status", justify="center")
            for name, (typ, status, status_color) in self.data.items():
                table.add_row(name, typ, f"[{status_color}]{status}[/{status_color}]")
            return table

    def update_live(self) -> None:
        """Update the associated live rich display. Also refreshes it."""
        with self.datalock:
            self.live.update(self._generate_renderable(), refresh=True)

    def set_updating(self, name: str) -> None:
        """Set the package to updating and update the live display.
        If name doesnt exist, do nothing.
        """
        self.update_status(name, "Updating...", "yellow")
        self.update_live()

    def set_finish(self, name: str, successful: bool) -> None:
        """Set the package to finished and update the live display.
        If name doesnt exist, do nothing.
        """
        if successful:
            with self.datalock:
                self.done += 1
            self.update_status(name, "Finished updating.", "green")
        else:
            self.update_status(name, "Update failed!", "red")
        self.update_live()


class DependencyUpdater:
    """Manage non-python dependencies."""

    def __init__(
        self, dependency_location: Path, dependency_definition_file: Path, git_timeout_s: float
    ) -> None:
        """Create a new updater.

        Boardfiles will be downloaded to the specified location at
        /boardfiles_downloads/<boardfile-name>. This is used to check whether they are outdated.

        Args:
            dependency_location: Path to the directory where all files are placed / checked.
            dependency_definition_file: This points to the yaml file containing all dependencies.
            git_timeout_s: Timeout for git requests in seconds.
        """
        self.git_timeout = git_timeout_s
        self.depfile = dependency_definition_file
        if not self.depfile.exists():
            raise FINNConfigurationError(
                f"External dependency definition file not found at: {self.depfile}"
            )
        self.git_deps = {}
        self.boarfiles = {}
        self.direct_downloads = {}
        self.dep_location = dependency_location
        if not self.dep_location.exists():
            self.dep_location.mkdir(parents=True)
        self.boardfile_temporary_downloads = self.dep_location / "board_files_downloads"

        # Load the definitions
        debug("Loading dependency definitions")
        data = {}
        with self.depfile.open() as f:
            data = yaml.load(f, yaml.Loader)
        if "git_deps" in data:
            self.git_deps = data["git_deps"]
        if "boardfiles" in data:
            self.boarfiles = data["boardfiles"]
        if "direct_downloads" in data:
            self.direct_downloads = data["direct_downloads"]

        # Try to find FINN_XSI. If it cannot be found, it is ignored in the
        # list of all dependencies (since this is neither a failed nor a successful install)
        self.finn_xsi_str = _resolve_module_path("finn_xsi")

    def get_dependency_count(self) -> int:
        """Sum of all dependencies."""
        return len(self.git_deps) + len(self.boarfiles) + len(self.direct_downloads)

    def _run_silent(self, cmd: str, cwd: Path | None = None, timeout: float | None = None) -> int:
        """Run a given command silently. Return its returncode."""
        debug(f"[DependencyUpdater] Running command: {cmd}", False)
        return sp.run(
            shlex.split(cmd, posix=IS_POSIX),
            cwd=cwd,
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
            stdin=sp.DEVNULL,
            timeout=timeout,
        ).returncode

    def _git_clone(self, url: str, commit: str, target: Path) -> bool:
        """Try to clone and checkout the git url to the given target directory. If something
        went wrong return False, True otherwise."""  # noqa
        if self._run_silent(f"git clone {url} {target.absolute()}", timeout=self.git_timeout) != 0:
            return False
        return self._run_silent(f"git checkout {commit}", cwd=target.absolute()) == 0

    def _get_git_hash(self, package_name: str) -> str | None:
        """Return the hash of the given package_name dependency.
        If there is no such package return None."""  # noqa
        if package_name in self.git_deps:
            target = self.dep_location / package_name
        elif package_name in self.boarfiles:
            target = self.boardfile_temporary_downloads / package_name
        else:
            return None
        if not target.exists():
            return None
        result = sp.run(
            "git rev-parse HEAD",
            text=True,
            capture_output=True,
            shell=True,
            cwd=target,
            timeout=self.git_timeout,
        )
        return result.stdout.strip()

    def _get_fields(self, package_name: str, package_data: dict, *field_names: str) -> tuple:
        """Return a tuple with all required fields from the data. If one of the fields does not
        exist, raise an exception."""  # noqa
        for field in field_names:
            if field not in package_data:
                raise FINNUserError(
                    f"Missing field {field} in dependency definition of {package_name}!"
                )
        return tuple([package_data[field] for field in field_names])

    def _dependency_type(self, package_name: str) -> Literal["Git", "Boardfiles", "Data", "Misc"]:
        """Return a string to tell which type this dependency is."""
        if package_name in self.git_deps:
            return "Git"
        if package_name in self.boarfiles:
            return "Boardfiles"
        if package_name in self.direct_downloads:
            return "Data"
        return "Misc"

    def _install_git_dependency(self, package_name: str) -> bool:
        """Install a git dependency. Return success."""
        debug(f"Trying to install GIT dependency: {package_name}", False)
        url, commit, pip_install = self._get_fields(
            package_name, self.git_deps[package_name], "url", "commit", "pip_install"
        )
        target = self.dep_location / package_name
        if not self._git_clone(url, commit, target):
            return False
        if self.is_outdated(package_name):
            return False
        if not pip_install:
            return True
        debug(f"[{package_name}] Running pip install..")
        self._run_silent(f"{sys.executable} -m pip install -e {target}")
        return not self.is_outdated(package_name)

    def _install_boardfile_dependency(self, package_name: str) -> bool:
        """Install a board file dependency. Return success."""
        debug(f"Trying to install BOARDFILE dependency: {package_name}", False)
        url, commit, subdir = self._get_fields(
            package_name, self.boarfiles[package_name], "url", "commit", "subdirectory"
        )
        subdir = Path(subdir)
        temp_target = self.boardfile_temporary_downloads / package_name
        source = temp_target / subdir
        target = self.dep_location / "board_files" / Path(subdir).name
        if not self._git_clone(url, commit, temp_target):
            return False
        # shutil.copytree() doesnt copy the _contents_, only the whole directory
        debug(f"[{package_name}] Copying to target at {target}")
        if subdir == Path():
            self._run_silent(f"cp -r {source}/* {target}")
        else:
            shutil.copytree(source, target)
        return not self.is_outdated(package_name)

    def _install_direct_download_dependency(self, package_name: str) -> bool:
        """Install a direct download dependency. Return success."""
        debug(f"Trying to install DIRECT DOWNLOAD dependency: {package_name}", False)
        if shutil.which("wget") is None or shutil.which("unzip") is None:
            # TODO: Allow curl and gzip etc. as well
            raise FINNConfigurationError(
                'Make sure that both "wget" and "unzip" are available on your system.'
            )
        url, do_unzip, target_directory = self._get_fields(
            package_name, self.direct_downloads[package_name], "url", "do_unzip", "target_directory"
        )
        target: Path = self.dep_location / target_directory

        # Return if the download fails
        # Automatically skips if not modified
        unzipped = (target / Path(url).name).with_suffix("")
        debug(f"[{package_name}] Running: wget -N {url}", False)
        wget_download = sp.run(
            shlex.split(f"wget -N {url}"), cwd=target, capture_output=True, text=True
        )
        if wget_download.returncode != 0:
            debug(f"[{package_name}] wget failed!", False)
            return False
        if "304 Not Modified" in wget_download.stderr.strip() and unzipped.exists():
            return True

        debug(f"[{package_name}] Removing previous install if necessary.", False)
        if unzipped.exists():
            shutil.rmtree(unzipped)

        # Unzip
        debug(f"[{package_name}] Unpacking..", False)
        if do_unzip:  # noqa
            if self._run_silent(f"unzip -o {Path(url).name}", cwd=target) != 0:
                return False
        return unzipped.exists()

    def _install_finn_xsi(self) -> bool:
        """Install FINN XSI bindings and return if installation was successful."""
        debug("[finn_xsi] Preparing LD LIBRARY PATH", False)
        finn_xsi_path = Path(self.finn_xsi_str)
        vivado_path = os.environ["XILINX_VIVADO"]
        required_paths = f"/lib/x86_64-linux-gnu/:{vivado_path}/lib/lnx64.o"
        if "LD_LIBRARY_PATH" not in os.environ:
            os.environ["LD_LIBRARY_PATH"] = required_paths
        else:
            os.environ["LD_LIBRARY_PATH"] = f"{required_paths}:{os.environ['LD_LIBRARY_PATH']}"

        # Run make
        debug("[finn_xsi] Calling make..", False)
        res = sp.run(["make"], cwd=finn_xsi_path, capture_output=True, text=True)
        if res.returncode != 0:
            Console().print(res.stderr)
            return False

        # Check if .so was created
        finn_xsi_so = finn_xsi_path / "xsi.so"
        debug(f"[finn_xsi] Checking if xsi.so was generated (at {finn_xsi_so})", False)
        if not finn_xsi_so.exists():
            return False

        # Set PATH/PYTHONPATH so the .so can be imported
        debug(f"[finn_xsi] Appending {finn_xsi_path} to PYTHONPATH", False)
        os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{finn_xsi_path.absolute()}"
        sys.path.append(str(finn_xsi_path))
        return True

    def install_dependency(self, package_name: str) -> bool:
        """Install the dependency in the dependency location. If no definition for this dependency
        exists or the installation failed, return False.
        If the installation was successful return true.
        """
        if package_name == "finn_xsi":
            return self._install_finn_xsi()
        match self._dependency_type(package_name):
            case "Git":
                return self._install_git_dependency(package_name)
            case "Boardfiles":
                return self._install_boardfile_dependency(package_name)
            case "Data":
                return self._install_direct_download_dependency(package_name)
            case _:
                return False

    def is_outdated(self, package_name: str) -> bool:
        """Return whether the a package is outdated. If no such package exist return False too."""
        if package_name == "finn_xsi":
            # If finn xsi was found its outdated, if it wasnt found, its never outdated
            return self.finn_xsi_str != ""
        if package_name in self.direct_downloads:
            # TODO: Improve (e.g. by checking directly instead of by using wget).
            # Check by letting wget compare timestamps. To avoid large wait times
            # immediately delete the file again after a short timeout.
            url = self.direct_downloads[package_name]["url"]
            target = (
                self.dep_location
                / self.direct_downloads[package_name]["target_directory"]
                / Path(url).name
            )
            wget_result = sp.run(
                shlex.split(f"wget -N {url}"),
                cwd=target.parent,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "304 Not Modified" in wget_result.stderr.strip():
                return False
            debug(wget_result.stderr.strip(), False)
            if target.exists():
                target.unlink()
            return True

        # Compare hashes for git dependencies and boardfiles
        dep_data = {}
        if package_name in self.git_deps:
            dep_data = self.git_deps
        elif package_name in self.boarfiles:
            dep_data = self.boarfiles

        # Try to fetch the current hash
        has_hash = self._get_git_hash(package_name)

        # If it doesnt exist yet, its definitely outdated
        if has_hash is None:
            return True

        # Compare hashes
        if "commit" not in dep_data[package_name]:
            raise FINNConfigurationError(
                f'Git dependency {package_name} has no "commit"'
                f"field in the definition file. Please provide one."
            )
        expected_hash = dep_data[package_name]["commit"]
        return expected_hash != has_hash

    def get_all_dependencies(self) -> list[str]:
        """Return a list of all dependencies."""
        return (
            list(self.git_deps.keys())
            + list(self.boarfiles.keys())
            + list(self.direct_downloads.keys())
            + ["finn_xsi"]
        )

    def get_oudated_dependencies(self) -> list[str]:
        """Return a list of the names of all outdated packages. For Git dependencies this means
        an outdated commit hash, for the others a different URL or target directory."""  # noqa
        return list(filter(self.is_outdated, self.get_all_dependencies()))

    def update(self) -> None:
        """With a live display and multithreading update all dependencies that are outdated."""
        deps_outdated = self.get_oudated_dependencies()

        # Function passed to threadpool
        def install_wrapper(package_name: str, status: _StatusTracker) -> bool:
            try:
                status.set_updating(package_name)
                result = self.install_dependency(package_name)
                status.set_finish(package_name, result)
                return result
            except Exception:
                status.set_finish(package_name, False)
                status.update_status(
                    package_name, "Updated failed! (Internal exception!)", "purple"
                )
                status.update_live()
                return False

        # Keep track of the status of all dependencies
        status = _StatusTracker(
            [(name, self._dependency_type(name)) for name in deps_outdated], Live()
        )
        if len(deps_outdated) > 0:
            # Display live updates of the installation process
            futures: list[Future] = []
            live = Live(status._generate_renderable(), refresh_per_second=0.0001)
            with live:
                status.live = live
                with ThreadPoolExecutor(max_workers=100) as tpe:
                    for package_name in deps_outdated:
                        futures.append(tpe.submit(install_wrapper, package_name, status))
                    tpe.shutdown(wait=True)
            for future in futures:
                if not future.result():
                    error("Dependency updates failed.")
                    sys.exit(1)
            Console().print(
                Panel(
                    f"Installed [green bold]{status.total}[/green bold] dependencies. "
                    f"(Skipped [orange3 bold]{self.get_dependency_count() - status.total}"
                    f"[/orange3 bold] dependencies "
                    f"due to existing installations.)"
                ),
                justify="center",
            )
        else:
            Console().print(
                Panel("[green]All dependencies are already cached and up to date.[/green]"),
                justify="center",
            )
