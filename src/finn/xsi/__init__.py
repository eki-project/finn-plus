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

Usage:
    # Check if XSI support is available
    from finn import xsi
    if xsi.is_available():
        import finn_xsi.adapter
"""

import contextlib
import os
import re
import sys
from typing import Any

from finn.util.exception import FINNUserError
from finn.util.logging import log
from finn.util.settings import get_settings

# Track if auto-install has been attempted
_auto_install_attempted = False

# Cache for loaded modules
_adapter_module: Any | None = None
_sim_engine_module: Any | None = None

xsi_path = get_settings().finn_xsi


def is_available() -> bool:
    """Check if XSI (RTL simulation) support is available.

    Returns:
        bool: True if finn_xsi can be imported, False otherwise
    """
    # Check if xsi.so exists
    xsi_so = xsi_path / "xsi.so"
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
    """Load finn_xsi modules if available."""
    global _adapter_module, _sim_engine_module

    if _adapter_module is not None:
        return True

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


# Trigger auto-install at import time
xsi_avail = is_available()

if xsi_avail is False:
    raise FINNUserError("XSI not available. Please run 'finn deps update' to install XSI.")

from finn_xsi.sim_engine import SimEngine  # noqa
from finn_xsi.adapter import (  # noqa
    locate_glbl,
    compile_sim_obj,
    get_simkernel_so,
    load_sim_obj,
    reset_rtlsim,
    close_rtlsim,
    rtlsim_multi_io,
)
