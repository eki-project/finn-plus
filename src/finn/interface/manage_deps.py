"""Manage dependencies. Called by run_finn.py."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import shlex
import shutil
import subprocess as sp
import sys
import traceback
import yaml
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from threading import RLock
from typing import cast

from finn.interface import IS_POSIX
from finn.interface.interface_utils import debug, error, resolve_module_path
from finn.util.exception import (
    FINNConfigurationError,
    FINNDependencyInstallationError,
    FINNUserError,
)

from pydantic.networks import HttpUrl  # noqa


class Dependency:
    """Baseclass for all dependencies."""

    model_config = ConfigDict(strict=True)


class GitDependency(BaseModel, Dependency):
    """Data model for a Git-based dependency.

    url: Git repository URL.
    commit: Commit to checkout.
    pip_install: Whether to install the repository using pip.
    """

    url: HttpUrl
    commit: str
    pip_install: bool
    install_editable: bool

    @model_validator(mode="after")
    def editable_install_requires_pip(self) -> GitDependency:
        """Validate that editable installs can only be done on pip installed repos."""
        if self.install_editable and not self.pip_install:
            raise FINNUserError(
                f"Dependency definition file: Git URL dependency {self.url} is not "
                f"installed via Pip, but install_editable is set to True."
            )
        return self


class BoardfileDependency(BaseModel, Dependency):
    """Data model for a Boardfile dependency.

    url: Git repository URL.
    commit: Commit to checkout.
    subdirectory: Path inside the repository from which to copy to the boardfiles directoty.
    """

    url: HttpUrl
    commit: str
    subdirectory: Path = Field(strict=False)


class DirectDownloadDependency(BaseModel, Dependency):
    """Data model for a direct download dependency.

    url: URL to download from.
    do_unzip: Whether to unzip the downloaded data.
    target_directory: Where to place the downloaded (uncompressed) data.
    """

    url: HttpUrl
    do_unzip: bool
    target_directory: Path = Field(strict=False)


class CustomDependency(BaseModel, Dependency):
    """Data model for a custom dependency.

    installation_function: Name of the function that should be implemented in the DependencyUpdater
                            to install this dependency.
    outdated_function: Name of function that returns whether this dependency is outdated.
    """

    installation_function: str
    outdated_function: str


class DependencyData(BaseModel):
    """Data model that stores all dependencies."""

    git_deps: dict[str, GitDependency]
    boardfile_deps: dict[str, BoardfileDependency]
    direct_download_deps: dict[str, DirectDownloadDependency]
    custom_deps: dict[str, CustomDependency]

    def get_all_dependencies(self) -> list[str]:
        """Return a list of all packages, across dependency types."""
        return list(
            chain.from_iterable(
                [
                    map(str, (x.keys()))
                    for x in [
                        self.git_deps,
                        self.boardfile_deps,
                        self.direct_download_deps,
                        self.custom_deps,
                    ]
                ]
            )
        )

    def get_dependency_count(self) -> int:
        """Return the total number of dependencies."""
        return len(self.get_all_dependencies())

    def assert_unique_dependency_names(self) -> None:
        """Assert that all dependencies across categories have unique names.
        Raise AssertionError otherwise.
        """
        assert self.get_dependency_count() == len(set(self.get_all_dependencies()))

    def dependency_type_str(self, package_name: str) -> str:
        """Return a string to tell which type this dependency is."""
        if package_name in self.git_deps:
            if self.git_deps[package_name].pip_install:
                if self.git_deps[package_name].install_editable:
                    return "Git (pip, editable)"
                return "Git (pip)"
            return "Git"
        if package_name in self.boardfile_deps:
            return "Boardfiles"
        if package_name in self.direct_download_deps:
            return "Data"
        if package_name in self.custom_deps:
            return "Custom"
        return "Misc"

    def get_dependency_data(self, package_name: str) -> Dependency | None:
        """Return the dependency data for the given package. If no such package
        exists return None.
        """
        for depdict in [
            self.git_deps,
            self.boardfile_deps,
            self.direct_download_deps,
            self.custom_deps,
        ]:
            if package_name in depdict:
                return depdict[package_name]
        return None

    def get_fields(self, package_name: str, *field_names: str) -> tuple:
        """Return a tuple with all required fields from the data. If one of the fields does not
        exist, raise an exception."""  # noqa
        self.assert_unique_dependency_names()
        dep_data = self.get_dependency_data(package_name)
        if dep_data is None:
            raise FINNUserError(
                f"Cannot request dependency data for non-existing dependency {package_name}!"
            )

        try:
            return tuple([dep_data.__getattribute__(field) for field in field_names])
        except AttributeError as e:
            raise FINNUserError(
                f"One of the fields in {field_names} does not exist in data for {package_name}!"
            ) from e


class _StatusTracker:
    """Small helper class to thread-safely organize status data."""

    def __init__(
        self, names_types: list[tuple[str, str]], live: Live | contextlib.nullcontext
    ) -> None:
        """Create a status tracker.

        Args:
            names_types: List of tuples that associate dependency names with their type.
            live: The rich.live.Live object that is used to display the status data.
        """
        # Name: (type, status, color)
        self.non_interactive = type(live) is contextlib.nullcontext
        self.data = {}
        self.live = live
        self.datalock = RLock()
        for name, typ in names_types:
            self.data[name] = (typ, "Initializing...", "grey70")
        self.total = len(self.data.keys())
        self.done = 0

    def update_status(self, name: str, status: str, color: str) -> None:
        """Update the status dict. If name doesnt exist, do nothing."""
        if self.non_interactive:
            return
        with self.datalock:
            if name in self.data:
                self.data[name] = (self.data[name][0], status, color)

    def _generate_renderable(self) -> Table:
        """Generate a renderable for rich to display in a live context."""
        if self.non_interactive:
            return
        with self.datalock:
            table = Table(
                title="Dependency Updates",
                caption=(
                    f"Installed: [cyan]{self.done}[/cyan] / [cyan bold]{self.total}[/cyan bold]."
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
        if self.non_interactive:
            return
        with self.datalock:
            self.live.update(self._generate_renderable(), refresh=True)

    def set_updating(self, name: str) -> None:
        """Set the package to updating and update the live display.
        If name doesnt exist, do nothing.
        """
        if self.non_interactive:
            return
        self.update_status(name, "Updating...", "yellow")
        self.update_live()

    def set_finish(self, name: str, successful: bool) -> None:
        """Set the package to finished and update the live display.
        If name doesnt exist, do nothing.
        """
        if self.non_interactive:
            Console().print("✓ " + name)
            return
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
        self,
        dependency_location: Path,
        dependency_definition_file: Path,
        git_timeout_s: float,
        non_interactive: bool = False,
    ) -> None:
        """Create a new updater.

        Boardfiles will be downloaded to the specified location at
        /boardfiles_downloads/<boardfile-name>. This is used to check whether they are outdated.

        Args:
            dependency_location: Path to the directory where all files are placed / checked.
            dependency_definition_file: This points to the yaml file containing all dependencies.
            git_timeout_s: Timeout for git requests in seconds.
            non_interactive: If set, don't generate a live status report.
        """
        self.non_interactive = non_interactive
        self.git_timeout = git_timeout_s
        self.depfile = dependency_definition_file
        if not self.depfile.exists():
            raise FINNConfigurationError(
                f"External dependency definition file not found at: {self.depfile}. "
                f"(If a different path was specified in the settings.yaml file, it was "
                f"not found either!)"
            )
        self.dep_location = dependency_location
        if not self.dep_location.exists():
            self.dep_location.mkdir(parents=True)
        self.boardfile_temporary_downloads = self.dep_location / "board_files_downloads"

        # Load the definitions
        debug("Loading dependency definitions")
        data = {}
        with self.depfile.open() as f:
            data = yaml.load(f, yaml.Loader)
        try:
            self.deps = DependencyData.model_validate(data)
        except ValidationError as e:
            raise FINNUserError(f"Validation error: {e}") from e

        # Try to find FINN_XSI. If it cannot be found, it is ignored in the
        # list of all dependencies (since this is neither a failed nor a successful install)
        self.finn_xsi_str = resolve_module_path("finn_xsi")

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
        clone_result = sp.run(
            shlex.split(f"git clone {url} {target.absolute()}"),
            timeout=self.git_timeout,
            capture_output=True,
            text=True,
        )
        if clone_result.returncode != 0:
            debug(f"[{url}] Cloning failed! Output was:\n{clone_result.stderr}", False)
            return False
        checkout_result = sp.run(
            shlex.split(f"git checkout {commit}"),
            cwd=target.absolute(),
            capture_output=True,
            text=True,
        )
        if checkout_result.returncode != 0:
            debug(f"[{url}] Checkout failed! Output was:\n{checkout_result.stderr}", False)
            return False
        return True

    def _get_git_hash(self, package_name: str) -> str | None:
        """Return the hash of the given package_name dependency.
        If there is no such package return None."""  # noqa
        if package_name in self.deps.git_deps:
            target = self.dep_location / package_name
        elif package_name in self.deps.boardfile_deps:
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

    def _install_git_dependency(self, package_name: str) -> bool:
        """Install a git dependency. Return success."""
        debug(f"Trying to install GIT dependency: {package_name}", False)
        url, commit, pip_install, install_editable = self.deps.get_fields(
            package_name, "url", "commit", "pip_install", "install_editable"
        )
        target = self.dep_location / package_name
        if target.exists() and importlib.util.find_spec(package_name.replace("-", "_")) is None:
            debug(
                "Git repository seems to exist, but is not installed "
                "into this environment. Removing dependency and cloning again.",
                False,
            )
            shutil.rmtree(target, ignore_errors=True)
        if not self._git_clone(url, commit, target):
            debug(f"[{package_name}] Cloning or checkout failed.", False)
            return False
        if self.is_outdated(package_name):
            return False
        if not pip_install:
            return True
        editable_flag = "" if not install_editable else "-e"
        pip_install_command = f"{sys.executable} -m pip install {editable_flag} {target}"
        debug(f"[{package_name}] Running pip install: {pip_install_command}")
        pip_subprocess_result = sp.run(
            shlex.split(pip_install_command), capture_output=True, text=True
        )
        if pip_subprocess_result.returncode != 0:
            debug(
                f"[{package_name}] Error during pip installation: {pip_subprocess_result.stderr}",
                False,
            )
            raise FINNDependencyInstallationError(pip_subprocess_result.stderr)
            return False

        # In editable install cases we need to make the package available in this run
        # otherwise it will only be available on the next start
        if install_editable:
            sys.path.append(str(target.absolute()))
        return not self.is_outdated(package_name)

    def _install_boardfile_dependency(self, package_name: str) -> bool:
        """Install a board file dependency. Return success."""
        debug(f"Trying to install BOARDFILE dependency: {package_name}", False)
        url, commit, subdir = self.deps.get_fields(package_name, "url", "commit", "subdirectory")
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
        url, do_unzip, target_directory = self.deps.get_fields(
            package_name, "url", "do_unzip", "target_directory"
        )
        url = str(url)
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

    def _install_custom(self, package_name: str) -> bool:
        """Install the custom dependency. The function name provided by the definition file
        must exist as a method of this class. If so, it is executed and it's return value
        used to check for success.
        """
        data = self.deps.get_dependency_data(package_name)
        assert data is not None
        function_name = cast("CustomDependency", data).installation_function
        try:
            return self.__getattribute__(function_name)()
        except AttributeError as e:
            raise FINNUserError(
                f"Implementation for custom installation function for "
                f"{package_name} not found in DependencyUpdater!"
            ) from e

    def _is_outdated_finn_xsi(self) -> bool:
        """Return whether FINN XSI is outdated."""
        # If finn xsi was found its outdated, if it wasnt found, its never outdated
        return self.finn_xsi_str != ""

    def _install_finn_xsi(self) -> bool:
        """Install FINN XSI bindings and return if installation was successful."""
        debug("[finn_xsi] Preparing LD LIBRARY PATH", False)
        finn_xsi_path = Path(self.finn_xsi_str)
        if "XILINX_VIVADO" not in os.environ:
            raise FINNDependencyInstallationError(
                "Vivado is not available " "(or XILINX_VIVADO is not set in your " "environment)!"
            )
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
            debug("Make returned an error during installation of FINN XSI", False)
            debug(f"Make stderr output: {res.stderr}", False)
            if "pybind11.h: No such file or directory" in res.stderr:
                raise FINNDependencyInstallationError(
                    "pybind11 seems to be missing " "in your environment!"
                )
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
        # Match doesnt work here for technical reasons (pydantic, match semantics)
        t = type(self.deps.get_dependency_data(package_name))
        if t is GitDependency:
            return self._install_git_dependency(package_name)
        if t is BoardfileDependency:
            return self._install_boardfile_dependency(package_name)
        if t is DirectDownloadDependency:
            return self._install_direct_download_dependency(package_name)
        if t is CustomDependency:
            return self._install_custom(package_name)
        return False

    def is_outdated(self, package_name: str, installed: bool = False) -> bool:
        """Return whether the a package is outdated. If no such package exist return False too."""
        data = self.deps.get_dependency_data(package_name)
        if data is None:
            raise FINNUserError(
                f"Cannot check if non-existing dependency {package_name} is outdated."
            )
        if package_name in self.deps.custom_deps:
            function_name = cast("CustomDependency", data).outdated_function
            try:
                return self.__getattribute__(function_name)()
            except AttributeError as e:
                raise FINNUserError(
                    f"Custom package {package_name} is missing the implementation"
                    f"of the outdated check function in DependencyUpdater!"
                ) from e
        if package_name in self.deps.direct_download_deps:
            # TODO: Improve (e.g. by checking directly instead of by using wget).
            # Check by letting wget compare timestamps. To avoid large wait times
            # immediately delete the file again after a short timeout.
            data = cast("DirectDownloadDependency", data)
            target = self.dep_location / data.target_directory / Path(str(data.url)).name
            if not target.parent.exists():
                target.parent.mkdir(parents=True)
            wget_result = sp.run(
                shlex.split(f"wget -N {data.url}"),
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
        # Try to fetch the current hash
        has_hash = self._get_git_hash(package_name)

        # If it doesnt exist yet, its definitely outdated
        if has_hash is None:
            debug(f"[{package_name}] No commit hash found - outdated.", False)
            return True

        # Compare hashes
        data = cast("GitDependency", data)
        if data.commit != has_hash:
            debug(
                f"[{package_name}] No matching hash commits: expected "
                f"{data.commit}, got {has_hash}",
                False,
            )
            return True

        # In some cases we do not want to check if the dependency is installed
        if not installed:
            return False

        # For pip-installable dependencies, also check if the package is accessible
        # in the current Python process
        if package_name in self.deps.git_deps and data.pip_install:
            # Try to find the package in the current Python environment
            spec = importlib.util.find_spec(package_name.replace("-", "_"))
            if spec is None:
                debug(f"[{package_name}] is not importable!", False)
                # Package is not importable, mark as outdated
                return True

        return False

    def get_outdated_dependencies(self) -> list[str]:
        """Return a list of the names of all outdated packages. For Git dependencies this means
        an outdated commit hash, for the others a different URL or target directory."""  # noqa
        return list(
            map(
                str,
                filter(
                    lambda pkg: self.is_outdated(pkg, installed=True),
                    self.deps.get_all_dependencies(),
                ),
            )
        )

    def update(self) -> None:
        """With a live display and multithreading update all dependencies that are outdated."""
        deps_outdated = self.get_outdated_dependencies()

        # Function passed to threadpool
        def install_wrapper(package_name: str, status: _StatusTracker) -> bool:
            """Wrap the installation function. Can be passed to a thread.

            Installs the given dependency and updates the status tracker along the way.
            """
            try:
                status.set_updating(package_name)
                result = self.install_dependency(package_name)
                status.set_finish(package_name, result)
                return result
            except FINNDependencyInstallationError as e:
                status.set_finish(package_name, False)
                status.update_status(package_name, f"Error: {e}", "purple")
                status.update_live()
                return False
            except Exception as e:
                status.set_finish(package_name, False)
                debug(f"[{package_name}] Exception: {e}", False)
                debug(f"[{package_name}] {traceback.format_exc()}", False)
                status.update_status(
                    package_name, "Updated failed! (Internal exception!)", "purple"
                )
                status.update_live()
                return False

        # Keep track of the status of all dependencies
        if self.non_interactive:
            live = contextlib.nullcontext()
            Console().print("Updating dependencies...")
        else:
            live = Live(Panel(""), refresh_per_second=0.0001)
        status = _StatusTracker(
            [(name, self.deps.dependency_type_str(name)) for name in deps_outdated], live
        )
        if len(deps_outdated) > 0:
            # Display live updates of the installation process
            futures: list[Future] = []
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
                    f"(Skipped [orange3 bold]{self.deps.get_dependency_count() - status.total}"
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
