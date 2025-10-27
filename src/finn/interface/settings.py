"""Manage settings for FINN+."""
# ruff: noqa: SLF001
from __future__ import annotations

import os
import psutil
import yaml
from pathlib import Path
from pydantic import BaseModel, PrivateAttr, ValidationError, computed_field
from typing import Any

from finn.util.exception import FINNUserError, FINNValidationError


# Modify yaml globally to convert paths to strings
# (Path() can be created from strings, and strings are more easily human-readable and changeable)
def path_repr(dumper, data):  # noqa
    """Represent a given object as a string."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))


yaml.add_multi_representer(Path, path_repr)

# Assumes we are in finn-plus/src/finn/interface/settings.py
FINN_ROOT = Path(__file__).parent.parent.parent.parent


def resolve_relative(path: Path | str, to: Path | str | None) -> Path:
    """If `path` is absolute, return it. If relative, return the combined absolute path.
    If `to` is None, just return path (absolute).
    """
    p = Path(path)
    if to is None:
        return p.absolute()
    if p.is_absolute():
        return p
    return (Path(to) / p).absolute()


class FINNSettings(BaseModel):
    """Keeps track of FINN+ settings. Instantiate with `FINNSettings.init()`."""

    _settings_path: Path = PrivateAttr()
    _auto_set_envvars: bool = PrivateAttr()
    _flow_config: Path | None = PrivateAttr()
    _finn_build_dir: str = PrivateAttr("FINN_TMP")
    _finn_deps: str = PrivateAttr("deps")
    _finn_deps_definitions: str = PrivateAttr("external_dependencies.yaml")
    _num_default_workers: int = PrivateAttr(-1)
    automatic_dependency_updates: bool = True
    deps_git_timeout: int = 100

    @computed_field
    @property
    def num_default_workers(self) -> int:
        """Number of default parallel workers."""
        if self._num_default_workers > -1:
            return self._num_default_workers
        cpus = psutil.cpu_count()
        if cpus is None or cpus == 1:
            return 1
        return int(cpus * 0.75)

    @num_default_workers.setter
    def num_default_workers(self, new_value: int | str) -> None:
        """Set number of default workers."""
        self._num_default_workers = int(new_value)
        if self._auto_set_envvars:
            os.environ["NUM_DEFAULT_WORKERS"] = str(self.num_default_workers)

    @computed_field
    @property
    def finn_build_dir(self) -> Path:
        """Absolute path to the FINN_BUILD_DIR."""
        if self._flow_config is None and not Path(self._finn_build_dir).is_absolute():
            raise FINNUserError(
                f"Settings can't resolve relative FINN BUILD DIR at "
                f"{self._finn_build_dir} without a flow config path."
            )
        return resolve_relative(
            self._finn_build_dir, Path(self._flow_config).parent  # type:ignore
        )

    @finn_build_dir.setter
    def finn_build_dir(self, new_path: str | Path | None) -> None:
        """Set the FINN_BUILD_DIR."""
        if new_path is None:
            return
        self._finn_build_dir = str(new_path)
        if self._auto_set_envvars:
            os.environ["FINN_BUILD_DIR"] = str(self.finn_build_dir)

    @computed_field
    @property
    def finn_deps(self) -> Path:
        """Absolute path to the FINN_DEPS dir."""
        return resolve_relative(self._finn_deps, FINN_ROOT)

    @finn_deps.setter
    def finn_deps(self, new_path: str | Path) -> None:
        """Set the FINN_DEPS dir."""
        if new_path is None:
            return
        self._finn_deps = str(new_path)
        if self._auto_set_envvars:
            os.environ["FINN_DEPS"] = str(self.finn_deps)

    @computed_field
    @property
    def finn_deps_definitions(self) -> Path:
        """Absolute path to the FINN_DEPS_DEFINITIONS."""
        return resolve_relative(self._finn_deps_definitions, FINN_ROOT)

    @finn_deps_definitions.setter
    def finn_deps_definitions(self, new_path: str | Path) -> None:
        """Set the FINN_DEPS_DEFINITIONS."""
        if new_path is None:
            return
        self._finn_deps_definitions = str(new_path)
        if self._auto_set_envvars:
            os.environ["FINN_DEPS_DEFINITIONS"] = str(self.finn_deps_definitions)

    def update_environment(self) -> None:
        """Update the environment variables according to field values."""
        os.environ["FINN_BUILD_DIR"] = str(self.finn_build_dir)
        os.environ["FINN_DEPS"] = str(self.finn_deps)
        os.environ["FINN_DEPS_DEFINITIONS"] = str(self.finn_deps_definitions)
        os.environ["NUM_DEFAULT_WORKERS"] = str(self.num_default_workers)

    @staticmethod
    def resolve_settings_file(override_settings_path: Path | None) -> Path:
        """Resolve the location of the settings file. Checks the following locations in the
        following order.

        1. Override (method argument)
        2. FINN_SETTINGS environment variable
        3. In the FINN+ repository root
        4. In ~/.finn/
        If the file does not exist, the function will still return the path for 1,2 and 4.
        It must then be created by another function.
        """
        if override_settings_path is not None:
            return override_settings_path
        if "FINN_SETTINGS" in os.environ:
            return Path(os.environ["FINN_SETTINGS"])
        repo_settings = FINN_ROOT / "settings.yaml"
        if repo_settings.exists():
            return repo_settings
        finn_home = Path("~/.finn").expanduser()
        if not finn_home.exists():
            finn_home.mkdir()
        return finn_home / "settings.yaml"

    @staticmethod
    def init(
        override_settings_path: Path | None = None,
        must_exist: bool = False,
        flow_config: Path | None = None,
        auto_set_environment_vars: bool = True,
        **kwargs: Any,
    ) -> FINNSettings:
        """Create a settings object. Tries to resolve the settings path.

        The settings file directory is resolved in this order:
        FINN_SETTINGS environment variable -> Repository Root -> `~/.finn/`.
        If override_settings_path exists, it is used instead.

        The settings are updated with the lowest priority first.
        (Defaults -> Settings file -> Environment -> Command line param)
        Thus, if a command line param is given, it is used over the others.

        Validation on whether the paths exist should be done by components using the settings,
        not the settings themselves.

        If `auto_set_environment_vars` is True, the envvars get set automatically when setting
        a value.

        Args:
            **kwargs: A dictionary with initial values for the settings.
            override_settings_path: If set, overrides the resolution of the settings file.
            must_exist: If True, object creation fails if the settings file does not exist.
            flow_config: If given, this can be used to resolve finn_build_dir. If not given,
                            relative FINN_BUILD_DIR settings cannot be resolved.
            auto_set_environment_vars: If True, set the environment variables too when assigning
                                        a setting.

        Returns:
            FINNSettings
        """
        # Sanity check
        assert list(FINNSettings.model_fields.keys()) == [
            k.lower() for k in FINNSettings.model_fields.keys()
        ], "All FINNSettings fields must be lowercase due to implementation details."

        # Resolve settings path
        settings_path = FINNSettings.resolve_settings_file(override_settings_path)

        # Verify file exists
        if must_exist and not settings_path.exists():
            raise FINNUserError(
                f"FINN Settings were resolved to {settings_path}. "
                f"The file does not yet exist, but must_exist was passed to "
                f"the settings constructor."
            )
        settings_data = {}
        if settings_path.exists():
            with settings_path.open() as f:
                settings_data = yaml.load(f, yaml.Loader)

        # Keep track of which fields have been set.
        set_fields = []

        # Sources for the settings (in order of priority)
        data = {
            "CLI Arguments": kwargs,
            "Environment Variables": os.environ.copy(),
            "Settings File": settings_data,
        }

        # Init settings with defaults
        settings = FINNSettings()
        settings._auto_set_envvars = False
        settings._flow_config = flow_config
        settings._auto_set_envvars = False
        settings._settings_path = settings_path

        # Update settings values from various sources
        for name, data_source in data.items():
            settings = settings.update_from(data_source, name, ignore=set_fields)
            settings._flow_config = flow_config
            settings._auto_set_envvars = False
            settings._settings_path = settings_path
            set_fields += [k.lower() for k in data_source.keys()]
        settings._auto_set_envvars = auto_set_environment_vars
        if auto_set_environment_vars:
            settings.update_environment()
        return settings

    def update_from(
        self, data: dict[str, Any], update_type: str | None = None, ignore: list[str] | None = None
    ) -> FINNSettings:
        """Update the model from the given data (Environment, custom dict, ...) and return it.

        Args:
            data: The data to update from.
            update_type: If set, this is used in the error raised in case validation fails.
            ignore: List of dict keys to NOT update from.

        Returns:
            FINNSettings: Newly validated settings.
        """
        # Make everything lower case
        modified_data = self.model_dump()
        for key in data.keys():
            lkey = key.lower()
            if lkey in modified_data:
                if ignore is not None and lkey in ignore:
                    continue
                modified_data[lkey] = data[key]
        try:
            new_model = FINNSettings()
            new_model._settings_path = self._settings_path
            new_model._auto_set_envvars = False
            new_model._flow_config = self._flow_config
            new_model = FINNSettings.model_validate(modified_data)
            new_model._settings_path = self._settings_path
            new_model._auto_set_envvars = False
            new_model._flow_config = self._flow_config
            # We need to manually update computed fields
            for key in modified_data.keys():
                if key in FINNSettings.model_computed_fields.keys():
                    new_model.__setattr__(key, modified_data[key])
        except ValidationError as e:
            err = str(e)
            if update_type is not None:
                err = f"Erro during settings validation in {update_type}: {e}"
            raise FINNValidationError(err) from e
        return new_model

    def settingsfile_exists(self) -> bool:
        """Return whether the settings file exists."""
        return self._settings_path.exists()

    def save(self, path: Path | None = None) -> None:
        """Save settings. If None is given saves to resolved path."""
        if path is None:
            path = self._settings_path
        with path.open("w+") as f:
            yaml.dump(self.model_dump(), f, yaml.Dumper)

    @staticmethod
    def get_settings_keys() -> list[str]:
        """Return all settings keys there are."""
        computed_fields = list(FINNSettings.model_computed_fields.keys())
        normal_fields = list(FINNSettings.model_fields.keys())
        return computed_fields + normal_fields
