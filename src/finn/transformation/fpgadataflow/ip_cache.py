"""Manage IP caching for FINN."""

from __future__ import annotations

import hashlib
import json
import numpy as np
import shlex
import shutil
import subprocess
from pathlib import Path
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from typing import TYPE_CHECKING, Callable, Final

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.util.basic import make_build_dir
from finn.util.deps import get_cache_path, get_deps_path
from finn.util.exception import FINNConfigurationError, FINNInternalError
from finn.util.logging import log

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


CACHE_IP_DEFINITIONS: dict[type, dict[str, list[str]]] = {}
"""Contains all node attributes that a custom operator needs to be characterized.
Filled by the cache_ip decorator. If the field "use" is defined, these attributes are
used to hash the op.
>>> CACHE_IP_DEFINITIONS[my_operator]["use"] = [...]

However if "ignore" is used, every attribute _except_ those listed are used.
>>> CACHE_IP_DEFINITIONS[my_operator]["ignore"] = [...]
"""


def cache_ip(attributes: list[str] | None = None) -> HWCustomOp:
    """Decorate the given custom operator to be cacheable.

    Args:
        attributes: List of the key names of all node attributes needed to
                    identify IP cores.
    """
    global CACHE_IP_DEFINITIONS

    def wrapper(op_cls: type) -> type:
        assert issubclass(op_cls, HWCustomOp), (
            f"Can only cache HWCustomOp instances, " f"but {op_cls.__name__} is not a HWCustomOP!"
        )
        if op_cls not in CACHE_IP_DEFINITIONS.keys():
            CACHE_IP_DEFINITIONS[op_cls] = {}
        if attributes is not None:
            CACHE_IP_DEFINITIONS[op_cls]["use"] = attributes
        else:
            # List of fields that don't define the IP core itself,
            # and can thus be ignored when hashing
            ignore_fields = [
                "code_gen_dir_ipgen",
                "ipgen_path",
                "ip_path",
                "cycles_rtlsim",
                "cycles_estimate",
                "res_estimate",
                "res_synth",
                "rtlsim_so",
                "executable_path",
                "res_hls",
                "code_gen_dir_cppsim",
            ]
            CACHE_IP_DEFINITIONS[op_cls]["ignore"] = ignore_fields
        print(f"Added custom op {op_cls.__name__} to the cacheable IP registry!")
        return op_cls

    return wrapper


class IPCache:
    """Manage IP caching."""

    # TODO: Update hash functions
    allowed_hashfuncs: Final[list[str]] = ["sha256"]

    def __init__(self, cache_dir: Path, hashfunc: str) -> None:
        """Construct a new IPCache object.

        Args:
            cache_dir: The path of the cache directory.
            hashfunc: The name of the hash function to be used.
        """
        self.cache_dir = cache_dir
        if not self.cache_dir.exists():
            self.cache_dir.mkdir()
        log.info(f"[IPCache] Cache directory: {self.cache_dir}")
        if hashfunc not in dir(hashlib):
            raise FINNConfigurationError(f"There is no hash function with the name {hashfunc}!")
        if hashfunc not in self.allowed_hashfuncs:
            raise FINNConfigurationError(
                f"Hash function {hashfunc} not available for caching. "
                f"Choose one of: {self.allowed_hashfuncs}"
            )

        self.hashfunc_name = hashfunc
        self.hasher: Callable = getattr(hashlib, hashfunc)

        # Prepare some always needed values
        # FINN Commit
        self.finn_commit = subprocess.run(
            shlex.split("git rev-parse HEAD"),
            text=True,
            capture_output=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
        log.info(f"[IPCache] FINN Commit reads: {self.finn_commit}")

        # FINN HLSLIB Commit
        self.hlslib_commit = subprocess.run(
            shlex.split("git rev-parse HEAD"),
            text=True,
            capture_output=True,
            cwd=get_deps_path() / "finn-hlslib",
        ).stdout.strip()
        log.info(f"[IPCache] HLSLIB Commit reads: {self.hlslib_commit}")

    def _get_key(self, op: HWCustomOp, model: ModelWrapper) -> str:
        """Return the key that can be hashed, for the given custom op.

        Returns:
            str: The human-readable key. Can be used to generate the caching
                    hash and the metadata file packed with the cached data.
        """
        global CACHE_IP_DEFINITIONS

        # TODO: Maybe exchange simple string concat for something more elegant at some point.
        # TODO: Practical, because we can include the unhashed key in the directory for debugging
        # Always use the current FINN and HLSLIB commits so that the correct versions are used
        key = f"FINN: {self.finn_commit}\nHLSLIB: {self.hlslib_commit}\n"

        # Two custom ops might need the same attributes, so add the type
        key += "type:" + str(type(op)) + "\n"

        # Add all node attributes required
        typ = type(op)
        if typ not in CACHE_IP_DEFINITIONS.keys():
            raise FINNInternalError(
                "Tried getting the hash for a non-cacheable custom operator. "
                "Did you perhaps forget to register the op for caching via "
                "@cache_ip(...)?"
            )

        # If both "use" and "ignore" are given, only use "use"
        if "use" in CACHE_IP_DEFINITIONS[typ].keys():
            keys = CACHE_IP_DEFINITIONS[typ]["use"]
            for attr in keys:
                data = None
                try:
                    data = op.get_nodeattr(attr)
                except Exception:
                    continue
                try:
                    data = str(data)
                except Exception as e:
                    raise FINNInternalError(
                        f"Unable to create string-representation for node "
                        f"attribute {attr} of custom op {op.onnx_node.name} of "
                        f"type {type(op)}."
                    ) from e
                # Finally add to key
                key += f"{attr}:{data}\n"

        elif "ignore" in CACHE_IP_DEFINITIONS[typ].keys():
            for name in op.get_nodeattr_types().keys():
                if name not in CACHE_IP_DEFINITIONS[typ]["ignore"]:
                    data = None
                    try:
                        data = op.get_nodeattr(name)
                    except Exception:
                        continue
                    try:
                        data = str(data)
                    except Exception as e:
                        raise FINNInternalError(
                            f"Unable to create string-representation for node "
                            f"attribute {name} of custom "
                            f"op {op.onnx_node.name} of "
                            f"type {type(op)}."
                        ) from e
                key += f"{name}:{data}\n"

        # Add parameters if existing
        # TODO: Extend to all custom ops that require this
        if isinstance(op, MVAU):
            mem_mode = None
            try:
                mem_mode = op.get_nodeattr("mem_mode")
            except Exception as e:
                raise FINNInternalError(
                    f"Cannot cache {op.onnx_node.name} because op is of "
                    f"type MVAU but has no mem_mode set!"
                ) from e
            if mem_mode in ["internal_embedded", "internal_decoupled"]:
                # TODO: Add shape!
                weight = np.ascontiguousarray(model.get_initializer(op.onnx_node.input[1]))
                array_hash = self.hasher(weight.tobytes()).hexdigest()
                key += f"weights_hash:{array_hash}\n"

        # TODO: Other ops that require parameters
        return key

    def _get_hash_hex(self, key: str) -> str:
        """Return the hex repr of the hash of the given key.

        The key can be created using _get_key(...)
        """
        return self.hasher(key.encode("UTF-8")).hexdigest()

    def _create_key_file(self, key: str, path: Path) -> None:
        """Write the given key data into a file at the given path."""
        with path.open("w+") as f:
            f.write(f"Hashed using {self.hashfunc_name}. Key:\n------------------------\n")
            f.write(key)

    def _cache_dir_path(self, hashed_key: str) -> Path:
        """Return the path to the directory matching the hashed key."""
        return self.cache_dir / hashed_key

    def _dump_nodeattrs(self, op: HWCustomOp, path: Path) -> None:
        """Dump the custom ops node attributes at the given path as a JSON."""
        required = ["ip_vlnv"]
        d = {}
        for name in op.get_nodeattr_types().keys():
            if name in required:
                try:
                    d[name] = op.get_nodeattr(name)
                except Exception:
                    continue
        with path.open("w+") as f:
            json.dump(d, f)

    def _prepare_from_cached_ip(self, op: HWCustomOp, hashed_key: str, make_copy: bool) -> None:
        """Prepare the given custom op for usage of the given cached IP.

        We have to set some node attributes normally set by HLSSynth and PrepareIP. This needs to
        be done to use the cached IP.

        Args:
            op: The operator of which the node attributes we have to modify.
            hashed_key: The hash hex repr of the key for this op. Used to find the cached IP.
            make_copy: If True, first makes a copy of the cached IP in the current FINN_BUILD_DIR
                        and sets the path towards this copy instead of the cached original.
        """
        log.info(f"Preparing {op.onnx_node.name} from cached IP (hashed key: {hashed_key[:10]}...)")
        ip_dir = self._cache_dir_path(hashed_key)
        saved_nodeattrs = {}

        # Check if the cached IP really exists
        if not ip_dir.exists():
            raise FINNInternalError(
                f"Cannot use hashed key {hashed_key}: Cache dir {ip_dir} does not exist!"
            )

        # Read node attributes from saved directory
        with (ip_dir / "nodeattrs.json").open("r") as f:
            saved_nodeattrs = json.load(f)

        # If needed make copy of the cached dir
        if make_copy:
            copied_dir = Path(make_build_dir(prefix=f"cached_code_gen_ipgen_{op.onnx_node.name}"))
            shutil.copytree(ip_dir, copied_dir, dirs_exist_ok=True)
            ip_dir = copied_dir

        # Set node attributes correctly to point to cached directory
        op.set_nodeattr("code_gen_dir_ipgen", str(ip_dir))
        op.set_nodeattr("ip_vlnv", saved_nodeattrs["ip_vlnv"])
        op.set_nodeattr(
            "ip_path", str(ip_dir / f"project_{op.onnx_node.name}" / "sol1" / "impl" / "ip")
        )
        op.set_nodeattr("ipgen_path", str(ip_dir / f"project_{op.onnx_node.name}"))

    def apply_cache(self, model: ModelWrapper) -> ModelWrapper:
        """First apply all cached IPs, then run synthesis and cache the ones not yet cached."""

        # TODO: Include PrepareIP for RTL nodes (not only synthesis)!

        # First Pass: Apply all cached IPs
        log.info("[IPCache] First pass: Applying cached IPs...")
        for node in model.graph.node:
            op = getCustomOp(node)
            key = self._get_key(op, model)
            hashed_key = self._get_hash_hex(key)
            cache_dir = self._cache_dir_path(hashed_key)
            if cache_dir.exists():
                self._prepare_from_cached_ip(op, hashed_key, make_copy=True)

        # Second Pass: Run synthesis and cache not yet cached nodes
        log.info("[IPCache] Second pass: Synthesizing and caching new IPs...")
        model = model.transform(HLSSynthIP())
        for node in model.graph.node:
            op = getCustomOp(node)
            key = self._get_key(op, model)
            hashed_key = self._get_hash_hex(key)
            target_dir = self._cache_dir_path(hashed_key)
            if not target_dir.exists():
                code_gen_dir = Path(op.get_nodeattr("code_gen_dir_ipgen"))
                if not code_gen_dir.exists():
                    raise FINNInternalError(
                        f"PrepareIP and/or HLSSynth for {node.name} "
                        f"were not successful: code_gen_dir_ipgen not found!"
                    )
                shutil.copytree(code_gen_dir, target_dir, dirs_exist_ok=True)
                self._create_key_file(key, target_dir / "key.txt")
                self._dump_nodeattrs(op, target_dir / "nodeattrs.json")
                log.info(f"Cached node {node.name}. Cached at: {target_dir} from {code_gen_dir}!")
        return model


class CachedHLSSynthIP(Transformation):
    """HLSSynth but cached."""

    # TODO: Remove / reorder steps hw_ipgen and hw_codegen
    def __init__(self, hash_function: str) -> None:
        """HLSSynth but cached."""
        super().__init__()
        self.hashfunc = hash_function

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply cached HLS Synthesis."""
        cache = IPCache(cache_dir=get_cache_path(), hashfunc=self.hashfunc)
        model = cache.apply_cache(model)
        return model, False
