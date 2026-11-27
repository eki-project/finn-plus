"""Shared test setup for the driver tests.

The overlay package imports PYNQ at module level, which only exists on the board image. The
search strategies (``overlays/fifo_search.py``) are pure Python, so a minimal stand-in for
``pynq`` is installed here when the real package is missing, which lets the search tests run
on a development machine as well.
"""

import importlib.util
import sys
import types


if importlib.util.find_spec("pynq") is None:  # pragma: no cover - only without PYNQ

    class _StubModule(types.ModuleType):
        """Module whose every attribute is a placeholder class."""

        def __getattr__(self, name):
            """Placeholder class for any public attribute."""
            if name.startswith("__"):
                raise AttributeError(name)
            return type(name, (), {"fclk0_mhz": 100.0})

    for name in ("pynq", "pynq.ps", "pynq.lib", "pynq.lib.dma"):
        sys.modules[name] = _StubModule(name)
    sys.modules["pynq"].ps = sys.modules["pynq.ps"]
    sys.modules["pynq"].lib = sys.modules["pynq.lib"]
