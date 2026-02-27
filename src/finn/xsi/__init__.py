############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################
"""FINN XSI (Xilinx Simulation Interface) support module

This module provides utilities for RTL simulation support via finn_xsi.
The finn_xsi extension must be built separately using the setup command.

Usage:
    # Check if XSI support is available
    from finn import xsi
    if xsi.is_available():
        import finn_xsi.adapter
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional
from finn.util.logging import log


# Track if auto-install has been attempted
_auto_install_attempted = False

# Cache for loaded modules
_adapter_module: Optional[Any] = None
_sim_engine_module: Optional[Any] = None
_xsi_module: Optional[Any] = None


def is_available() -> bool:
    """Check if XSI (RTL simulation) support is available.

    Returns:
        bool: True if finn_xsi can be imported, False otherwise
    """
    # Check if xsi.so exists
    xsi_path = Path(os.environ["FINN_XSI"])
    xsi_so = xsi_path / "xsi.so"
    if not xsi_so.exists():
        # Attempt auto-install if not yet tried
        _attempt_auto_install()
        # Check again after auto-install attempt
        if not xsi_so.exists():
            print("XSI INSTALL: xsi.so does not exist")
            return False

    # Try loading the modules (this will cache them if successful)
    return _load_modules()


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
            else:
                print("✗ XSI installation failed. Run 'python -m finn.xsi.setup' for details.")
                return False
        finally:
            sys.argv = original_argv

    except Exception as e:
        log.error(f"✗ XSI auto-installation failed: {e}.")
        return False


def _load_modules() -> bool:
    """Load finn_xsi modules if available."""
    global _adapter_module, _sim_engine_module, _xsi_module

    if _adapter_module is not None:
        return True

    xsi_path = Path(os.environ["FINN_XSI"])
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
        import xsi

        _xsi_module = xsi
        _adapter_module = finn_xsi.adapter
        _sim_engine_module = finn_xsi.sim_engine

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
            try:
                sys.path.remove(str(xsi_path))
            except ValueError:
                pass  # Path was already removed somehow


# List of functions to wrap from finn_xsi.adapter
_ADAPTER_FUNCTIONS = [
    "locate_glbl",
    "compile_sim_obj",
    "get_simkernel_so",
    "load_sim_obj",
    "reset_rtlsim",
    "close_rtlsim",
    "rtlsim_multi_io",
]


def __getattr__(name: str) -> Any:
    """Dynamically wrap finn_xsi.adapter functions."""
    if name in _ADAPTER_FUNCTIONS:

        def wrapper(*args, **kwargs):
            if not _load_modules():
                raise ImportError("finn_xsi not available. Run: python -m finn.xsi.setup")
            return getattr(_adapter_module, name)(*args, **kwargs)

        wrapper.__name__ = name
        wrapper.__doc__ = f"Wrapper for finn_xsi.adapter.{name}"
        return wrapper
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# SimEngine class wrapper
class SimEngine:
    """Wrapper for finn_xsi.sim_engine.SimEngine."""

    def __init__(self, *args, **kwargs):
        if not _load_modules():
            raise ImportError("finn_xsi not available. Run: python -m finn.xsi.setup")
        self._engine = _sim_engine_module.SimEngine(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._engine, name)


# Trigger auto-install at import time
is_available()
