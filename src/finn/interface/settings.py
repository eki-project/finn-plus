"""Manage settings for FINN+."""
from __future__ import annotations

import os
import psutil
import yaml
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from finn.interface.interface_utils import error, warning

if TYPE_CHECKING:
    from collections.abc import Generator


class FINNSettings:
    """Keeps track of FINN settings."""

    KNOWN_KEYS: Final[dict[str, str]] = {
        "AUTOMATIC_DEPENDENCY_UPDATES": "Execute dependency updates at the start of a FINN+ build.",
        "DEPS_GIT_TIMEOUT": "Timeout in seconds for pulling a Git dependency.",
        "FINN_BUILD_DIR": (
            "Directory that contains all temporary build related files."
            "Absolute or relative to the build config."
        ),
        "FINN_DEPS": (
            "Directory that contains all non-Python dependencies."
            "Absolute or relative to the FINN+ root directory."
        ),
        "FINN_DEPS_DEFINITIONS": (
            "Path to the non-Python dependency definition file. Absolute or"
            "relative to the FINN+ root directory"
        ),
        "IGNORE_UNKNOWN_SETTINGS": (
            "Whether to emit warnings in case " "the settings contain unknown field names."
        ),
    }

    # Keys that need to be converted to str/Path when saving/loading
    PATH_KEYS: Final[list[str]] = ["FINN_DEPS", "FINN_BULD_DIR", "FINN_DEPS_DEFINITIONS"]

    def __init__(
        self,
        sync: bool = False,
        fallback_settings_path: Path | None = None,
        override_path: Path | None = None,
    ) -> None:
        """Create a new settings instance. Automatically resolves settings location.
        **NOTE**: This class can be constructed even if the resolved settings location does not
        exist. Note that in this case it has to be written either automatically via sync=True or
        manually by calling .save().

        Args:
            sync: If True, changes to settings are automatically synced back to the settings file.
                    If no such file exists, one is created.
            fallback_settings_path: If no settings file is found, a new one may be created there.
                                    If None, the fallback-path ~/.finn/settings.yaml is used.
            override_path: If given, use this path instead of trying to resolve the default
                            location. Can for example be used to generate a new config from scratch.
        """
        self.sync = sync
        self._settings_path: Final[Path]
        if override_path is not None:
            self._settings_path = override_path
        else:
            settings_path = FINNSettings._resolve_settings_path()
            if settings_path is None:
                if fallback_settings_path is None:
                    if "FINN_SETTINGS" in os.environ:
                        error("FINN_SETTINGS environment variable points to an invalid path.")
                        error("Defaulting to fallback settings path.")
                    self._settings_path = Path.home() / ".finn" / "settings.yaml"
                else:
                    self._settings_path = fallback_settings_path
            else:
                self._settings_path = settings_path
        self._settings: dict[str, Any] = {}
        # Set to true only for initial loading to avoid double printed warnings
        self.ignore_unknown_settings = True
        self.load()
        self.ignore_unknown_settings = self._settings.get("IGNORE_UNKNOWN_SETTINGS", False)
        if self.ignore_unknown_settings:
            for key in self._settings.keys():
                if key not in FINNSettings.KNOWN_KEYS:
                    warning(f"Found unknown key {key} in settings.")

    @staticmethod
    def _resolve_settings_path() -> Path | None:
        """Best effort to find the settings file. If it is found nowhere and isnt provided
        via an environment variable (FINN_SETTINGS), return None"""  # noqa
        if "FINN_SETTINGS" in os.environ.keys():
            p = Path(os.environ["FINN_SETTINGS"])
            if p.exists():
                return p
            return None
        paths = [
            Path(__file__).parent.parent.parent.parent / "settings.yaml",
            Path.home() / ".finn" / "settings.yaml",
            Path.home() / ".config" / "settings.yaml",
        ]
        for path in paths:
            if path.exists():
                return path
        return None

    def load_defaults(self) -> bool:
        """Load the default values into the settings by updating them.If sync=True was given,
        the settings are immediately written.

        Returns:
            bool: If sync=True the return value indicates if writing the file was successful.
                    If sync=False, True is always returned.
        """
        defaults = {
            "DEPS_GIT_TIMEOUT": 120,
            "AUTOMATIC_DEPENDENCY_UPDATES": True,
            "FINN_DEPS": Path("deps"),
            "FINN_BUILD_DIR": Path("FINN_TMP"),
            "IGNORE_UNKNOWN_SETTINGS": False,
            "FINN_DEPS_DEFINITIONS": Path("external_dependencies.yaml"),
        }
        for key in FINNSettings.KNOWN_KEYS.keys():
            assert key in defaults, f"Key {key} is missing in settings defaults."
        self._settings.update(defaults)
        if self.sync:
            return self.save()
        return True

    def get_path(self) -> Path:
        """Get the path to the settings file. If not existent, it will be created when needed."""
        return self._settings_path

    def load(self) -> bool:
        """Update the settings.

        Returns:
            bool: True if the data was updated. False if no settings file exists.
        """
        if not self._settings_path.exists():
            return False
        temp = {}
        with self._settings_path.open() as f:
            temp = yaml.load(f, yaml.Loader)
        for key, value in temp.items():
            if key in FINNSettings.PATH_KEYS:
                temp[key] = Path(value)
        if not self.ignore_unknown_settings:
            for key in temp.keys():
                if key not in FINNSettings.KNOWN_KEYS:
                    warning(f"Loaded unknown key {key} in settings.")
        self._settings.update(temp)
        return True

    def save(self) -> bool:
        """Save settings to the settings path."""
        # Convert Path objects to strings for readability
        data = {k: (v if not isinstance(v, Path) else str(v)) for k, v in self._settings.items()}
        with self._settings_path.open("w+") as f:
            yaml.dump(
                data,
                f,
                yaml.Dumper,
            )
        return True

    def resolve_build_dir(
        self, build_dir: Path | None, flow_config: Path, is_test_run: bool
    ) -> Path:
        """Resolve the path of the build directory.

        **NOTE**: This does *not* modify the settings.
        """
        if build_dir is not None:
            return build_dir
        if "FINN_BUILD_DIR" in os.environ:
            p = Path(os.environ["FINN_BUILD_DIR"])
            if not p.is_absolute():
                return flow_config.parent / p
            return p
        if "FINN_BUILD_DIR" in self:
            p = Path(self["FINN_BUILD_DIR"])
            if not p.is_absolute():
                return flow_config.parent / p
            return p
        if is_test_run:
            # Need a different fallback because tests have no build config
            return Path("/tmp/FINN_TEST_BUILD_DIR")
        return flow_config.parent / "FINN_TMP"

    def resolve_deps_path(self, deps: Path | None) -> Path:
        """Resolve the path of the deps directory.

        **NOTE**: This does *not* modify the settings.
        """
        if deps is not None:
            return deps
        if "FINN_DEPS" in os.environ.keys():
            p = Path(os.environ["FINN_DEPS"])
            if not p.is_absolute():
                return Path(__file__).parent.parent.parent.parent / p
            return p
        if "FINN_DEPS" in self:
            p = Path(self["FINN_DEPS"])
            if not p.is_absolute():
                return Path(__file__).parent.parent.parent.parent / p
            return p
        return Path.home() / ".finn" / "deps"

    def resolve_deps_definitions_path(self, dep_def: Path | None) -> Path:
        """Resolve the path of the deps definition file.
        **NOTE**: This does *not* modify the settings.
        """
        if dep_def is not None:
            return dep_def
        if "FINN_DEPS_DEFINITIONS" in os.environ.keys():
            p = Path(os.environ["FINN_DEPS_DEFINITIONS"])
            if not p.is_absolute():
                return Path(__file__).parent.parent.parent.parent / p
            return p
        if "FINN_DEPS_DEFINITIONS" in self:
            p = Path(self["FINN_DEPS_DEFINITIONS"])
            if not p.is_absolute():
                return Path(__file__).parent.parent.parent.parent / p
            return p
        return Path(__file__).parent.parent.parent.parent / "external_dependencies.yaml"

    def resolve_num_workers(self, num: int) -> int:
        """Resolve the number of workers to use. Uses 75% of cores available as default fallback.

        **NOTE**: This does *not* modify the settings.
        """
        if num > -1:
            return num
        if "NUM_DEFAULT_WORKERS" in os.environ.keys() and os.environ["NUM_DEFAULT_WORKERS"] != "":
            return int(os.environ["NUM_DEFAULT_WORKERS"])
        if "NUM_DEFAULT_WORKERS" in self:
            return int(self["NUM_DEFAULT_WORKERS"])
        cpus = psutil.cpu_count()
        if cpus is None or cpus == 1:
            return 1
        return int(cpus * 0.75)

    def resolve_skip_update(self, do_skip_cmdline_arg: bool) -> bool:
        """Resolve whether the dependency update should be skipped.

        **NOTE**: This does *not* modify the settings.
        """
        if "AUTOMATIC_DEPENDENCY_UPDATES" in self:
            default = not self["AUTOMATIC_DEPENDENCY_UPDATES"]
            return do_skip_cmdline_arg or default
        return do_skip_cmdline_arg

    def __getitem__(self, key: Any) -> Any:
        """Get a settings value from a given key."""
        if self.sync:
            self.load()
        return self._settings.__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set a settings value."""
        self._settings.__setitem__(key, value)
        if self.sync:
            self.save()

    def update(self, other: dict) -> None:
        """Update the settings with the given dict."""
        self._settings.update(other)
        if self.sync:
            self.save()

    def __contains__(self, key: Any) -> bool:
        """Return whether the settings contain the given key."""
        return key in self._settings

    def keys(self) -> Generator:
        """Return a generator to loop all settings keys."""
        yield from self._settings.keys()

    def items(self) -> Generator:
        """Return a generator to loop all settings keys and values."""
        yield from self._settings.items()

    def values(self) -> Generator:
        """Return a generator to loop all settings values."""
        yield from self._settings.values()

    def delete(self, key: Any) -> None:
        """Delete the given key/value in the settings."""
        del self._settings[key]
        if self.sync:
            self.save()
