############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################
"""FINN XSI (Xilinx Simulation Interface) support module.

This module provides utilities for RTL simulation support via finn_xsi.
The finn_xsi extension must be built separately using the setup command.

Importing this module is always safe: it never touches FINN settings or the
filesystem, so it can be imported from any process (including
multiprocessing.Pool workers spawned with the "forkserver"/"spawn" start
methods) without FINN having been booted there.

The actual availability check (and Vivado-version-triggered rebuild) only
happens when ensure_available() is called explicitly. FINN's boot sequence
does this once, early: run_finn.py's prepare_finn() for the CLI, and
tests/conftest.py's pytest_configure() for pytest runs. Both happen before
any parallel work (multiprocessing.Pool / pytest-xdist workers) is
dispatched, so the build is never raced.

Usage:
    from finn import xsi
    xsi.ensure_available()
    if xsi.is_available():
        ...
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any

from finn.util.exception import FINNUserError
from finn.util.logging import log
from finn.util.settings import get_settings

if TYPE_CHECKING:
    # Real imports for type checkers/IDEs only - never executed at runtime.
    # The actual runtime bindings are created lazily by _load_modules().
    from finn_xsi.adapter import (  # noqa
        close_rtlsim,
        compile_sim_obj,
        get_simkernel_so,
        load_sim_obj,
        locate_glbl,
        reset_rtlsim,
        rtlsim_multi_io,
    )
    from finn_xsi.sim_engine import SimEngine  # noqa

# Track if auto-install has been attempted
_auto_install_attempted = False

# Cache for loaded modules
_adapter_module: Any | None = None
_sim_engine_module: Any | None = None

_LAZY_NAMES = frozenset(
    {
        "SimEngine",
        "locate_glbl",
        "compile_sim_obj",
        "get_simkernel_so",
        "load_sim_obj",
        "reset_rtlsim",
        "close_rtlsim",
        "rtlsim_multi_io",
    }
)


def _xsi_path() -> Path:
    """Return the current finn_xsi installation directory from FINN settings."""
    return get_settings().finn_xsi


def _xsi_so_path() -> Path:
    """Return the assumed xsi.so path. Does not necessarily point to an existing file."""
    return _xsi_path() / "xsi.so"


def is_available() -> bool:
    """Check if XSI (RTL simulation) support is available.

    Returns:
        bool: True if finn_xsi can be imported, False otherwise
    """
    xsi_path = _xsi_path()

    # Check if xsi.so exists
    xsi_so = _xsi_so_path()
    vivado_path = os.environ.get("XILINX_VIVADO")
    if vivado_path is None:
        raise OSError("XILINX_VIVADO environment variable not set. Please source Vivado settings.")
    match = re.search(r"\b(20\d{2})\.(1|2)\b", vivado_path)
    if not match:
        raise ValueError(f"Could not parse Vivado version from XILINX_VIVADO path: {vivado_path}")
    year, minor = int(match.group(1)), int(match.group(2))

    version_file = xsi_path / "VERSION"

    if not xsi_so.exists() or not version_file.exists():
        # Attempt auto-install if not yet tried
        _attempt_auto_install()
        # Check again after auto-install attempt
        if not xsi_so.exists():
            print("XSI INSTALL: xsi.so does not exist")
            return False
    with version_file.open() as f:
        version_info = f.read().strip()
    if version_info != f"Vivado {year}.{minor}":
        # Attempt auto-install if not yet tried
        _attempt_auto_install()
        # Check again after auto-install attempt
        if not xsi_so.exists():
            print("XSI INSTALL: xsi.so does not exist")
            return False

    # Try loading the modules (this will cache them if successful)
    return _load_modules()


def ensure_available() -> bool:
    """Verify (and if needed, build/rebuild) XSI for the currently loaded Vivado version.

    This is the intended entry point for FINN's boot sequence: call this once,
    early, before any parallel work (multiprocessing.Pool / pytest-xdist workers)
    is dispatched. Merely importing this module never triggers this implicitly -
    see the module docstring.

    Returns:
        bool: True if XSI is available.

    Raises:
        FINNUserError: if XSI could not be made available.
    """
    if not is_available():
        raise FINNUserError("XSI not available. Please run 'finn deps update' to install XSI.")
    return True


def _attempt_auto_install() -> bool:
    """Attempt to automatically install XSI if not available.

    Returns:
        bool: True if installation succeeded, False otherwise
    """
    global _auto_install_attempted

    # Only try once
    if _auto_install_attempted:
        return False

    _auto_install_attempted = True

    print("finn_xsi not found. Attempting automatic installation...")

    try:
        # Import and run the setup main function
        from finn.xsi import setup

        # Suppress output by temporarily redirecting stdout/stderr
        original_argv = sys.argv
        try:
            # Run setup with --quiet flag
            sys.argv = ["setup", "--quiet"]
            result = setup.main()

            if result == 0:
                print("✓ XSI installation completed successfully!")
                return True
            print("✗ XSI installation failed. Run 'python -m finn.xsi.setup' for details.")
            return False
        finally:
            sys.argv = original_argv

    except Exception as e:
        log.error(f"✗ XSI auto-installation failed: {e}.")
        return False


def _load_modules() -> bool:
    """Load finn_xsi modules if available and bind the public names on this module."""
    global _adapter_module, _sim_engine_module

    if _adapter_module is not None:
        return True

    xsi_path = _xsi_path()
    xsi_so = xsi_path / "xsi.so"

    if not xsi_so.exists():
        print("XSI INSTALL: xsi.so does not exist (load modules)")
        return False

    # Temporarily add to path for import
    path_added = str(xsi_path) not in sys.path
    if path_added:
        sys.path.insert(0, str(xsi_path))

    try:
        import finn_xsi.adapter
        import finn_xsi.sim_engine

        _adapter_module = finn_xsi.adapter
        _sim_engine_module = finn_xsi.sim_engine

        # Bind the public names as real attributes on this module, so that
        # subsequent access (including via __getattr__ below) is a plain,
        # fast attribute lookup - exactly like a normal eager import.
        module = sys.modules[__name__]
        module.SimEngine = finn_xsi.sim_engine.SimEngine  # type: ignore
        module.locate_glbl = finn_xsi.adapter.locate_glbl  # type: ignore
        module.compile_sim_obj = finn_xsi.adapter.compile_sim_obj  # type: ignore
        module.get_simkernel_so = finn_xsi.adapter.get_simkernel_so  # type: ignore
        module.load_sim_obj = finn_xsi.adapter.load_sim_obj  # type: ignore
        module.reset_rtlsim = finn_xsi.adapter.reset_rtlsim  # type: ignore
        module.close_rtlsim = finn_xsi.adapter.close_rtlsim  # type: ignore
        module.rtlsim_multi_io = finn_xsi.adapter.rtlsim_multi_io  # type: ignore

        return True
    except ImportError as e:
        # Log the specific import error for debugging
        log.debug(f"Failed to import finn_xsi modules: {e}")
        print("XSI INSTALL: import error: " + str(e))
        return False
    except Exception as e:
        # Catch any unexpected errors during module loading
        log.warning(f"Unexpected error loading finn_xsi: {type(e).__name__}: {e}")
        print(f"Unexpected error loading finn_xsi: {type(e).__name__}: {e}")
        return False
    finally:
        # Remove from path if we added it
        if path_added and str(xsi_path) in sys.path:
            with contextlib.suppress(ValueError):
                sys.path.remove(str(xsi_path))


def __getattr__(name: str) -> Any:
    """Lazy safety net for the names normally bound by ensure_available()/_load_modules().

    Normal FINN flows call ensure_available() once at boot (see run_finn.py's
    prepare_finn() and tests/conftest.py's pytest_configure()), at which point
    these names become plain module attributes and this hook is never consulted
    again. It exists for ad hoc scripts/REPL usage that import finn.xsi without
    going through FINN's usual startup path.
    """
    if name not in _LAZY_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    ensure_available()
    return getattr(sys.modules[__name__], name)
