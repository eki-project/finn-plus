"""Zero-cost (training-free) proxies for ranking NAS candidates without training them.

The measures are vendored from NASLib (NAS-Bench-Suite-Zero) and adapted to Brevitas
networks; see the README of this package for origin, license and local modifications.
Unlike the rest of :mod:`finn.nas`, this subpackage imports torch and brevitas at module
level.
"""

from finn.nas.zc.proxies import DEFAULT_MEASURES, ProxyResult, available_measures, compute_proxies

__all__ = ["DEFAULT_MEASURES", "ProxyResult", "available_measures", "compute_proxies"]
