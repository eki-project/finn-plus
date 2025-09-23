"""Manage settings for FINN+."""
from __future__ import annotations

import os
import psutil
import yaml
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from finn.interface.interface_utils import error

if TYPE_CHECKING:
    from collections.abc import Generator


class FINNSettings:
    """Keeps track of FINN settings."""

    def __init__(
        self,
        sync: bool = False,
        fallback_settings_path: Path | None = None,
        override_path: Path | None = None,
    ) -> None:
        """Create a new settings instance. Automatically resolves settings location.

        Args:
            sync: If True, changes to settings are automatically synced back to the settings file.
                    If no such file exists, one is created.
            fallback_settings_path: If no settings file is found, a new one may be created there.
                                    If None, the fallback-path ~/.finn/settings.yaml is used.
            override_path: If given, use this path instead of trying to resolve the default
                            location. Can for example be used to generate a new config from scratch.
        """
        self.sync = sync
        self._settigs_path_keys: list[str] = ["FINN_DEPS", "FINN_BULD_DIR"]
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
        self._settings: dict[str, Any] = {}
        self.load()

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
            if key in self._settigs_path_keys:
                temp[key] = Path(value)
        self._settings.update(temp)
        return True

    def save(self) -> bool:
        """Save settings to the settings path."""
        with self._settings_path.open("w+") as f:
            yaml.dump(
                {k: (v if type(k) is not Path else str(v)) for k, v in self._settings.items()},
                f,
                yaml.Dumper,
            )

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

    def resolve_deps_path(self, deps: Path | None) -> Path | None:
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
        p = Path.home() / ".finn" / deps
        if p.exists():
            return p
        return None

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

    def __getitem__(self, key: Any) -> Any:  # noqa
        if self.sync:
            self.load()
        return self._settings.__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:  # noqa
        self._settings.__setitem__(key, value)
        if self.sync:
            self.save()

    def update(self, other: dict) -> None:
        """Update the settings with the given dict."""
        self._settings.update(other)
        if self.sync:
            self.save()

    def __contains__(self, key: Any) -> bool:  # noqa
        return key in self._settings

    def items(self) -> Generator:  # noqa
        yield from self._settings

    def values(self) -> Generator:  # noqa
        yield from self._settings.values()

    def delete(self, key: Any) -> None:  # noqa
        del self._settings[key]
        if self.sync:
            self.save()
