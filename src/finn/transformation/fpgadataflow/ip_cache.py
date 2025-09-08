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
from typing import TYPE_CHECKING, Any, Callable, Final, cast

from finn.custom_op.fpgadataflow.attention import ScaledDotProductAttention
from finn.custom_op.fpgadataflow.channelwise_op import ChannelwiseOp
from finn.custom_op.fpgadataflow.elementwise_binary import ElementwiseBinaryOperation
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.custom_op.fpgadataflow.lookup import Lookup
from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
from finn.custom_op.fpgadataflow.thresholding import Thresholding
from finn.custom_op.fpgadataflow.vectorvectoractivation import VVAU
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.util.basic import make_build_dir
from finn.util.deps import get_cache_path, get_deps_path
from finn.util.exception import FINNConfigurationError, FINNInternalError
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.logging import log

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper


CACHE_IP_DEFINITIONS: dict[type, dict[str, list[str]]] = {}
"""Contains all node attributes that a custom operator needs to be characterized.
Filled by the cache_ip decorator. If the field "use" is defined, these attributes are
used to hash the op.
>>> CACHE_IP_DEFINITIONS[my_operator]["use"] = [...]

However if "ignore" is used, every attribute _except_ those listed are used.
>>> CACHE_IP_DEFINITIONS[my_operator]["ignore"] = [...]
"""


def cache_ip(attributes: list[str] | None = None) -> Callable[[type], type]:
    """Decorate the given custom operator to be cacheable.

    Args:
        attributes: List of the key names of all node attributes needed to
                    identify IP cores.
    """
    global CACHE_IP_DEFINITIONS

    def wrapper(op_cls: type) -> type:
        assert issubclass(
            op_cls, HWCustomOp
        ), f"Can only cache HWCustomOp instances, but {op_cls.__name__} is not a HWCustomOP!"
        if op_cls not in CACHE_IP_DEFINITIONS.keys():
            CACHE_IP_DEFINITIONS[op_cls] = {}
        else:
            # Already marked
            return op_cls
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
        return op_cls

    return wrapper


class IPCache:
    """Manage IP caching.

    Application: To apply this in a normal flow, execute somewhat like this:
    ```
    cache = IPCache(...)
    model = cache.apply(model)              # Apply already cached IPs
    model = model.transform(HLSSynthIP())   # Generate IPs that weren't available
    cache.update(model)                     # Cache the newly generated IPs too
    ```
    """

    # TODO: Update hash functions
    allowed_hashfuncs: Final[list[str]] = ["sha256"]

    def __init__(
        self,
        cache_dir: Path,
        hashfunc: str,
        hls_clk_period: float,
        cache_hls_clk: bool,
        fpgapart: str,
        cache_fpgapart: bool,
    ) -> None:
        """Construct a new IPCache object.

        Args:
            cache_dir: The path of the cache directory.
            hashfunc: The name of the hash function to be used.
        """
        self.cache_dir = cache_dir
        self.cache_hls_clk = cache_hls_clk
        self.cache_fpgapart = cache_fpgapart
        if not self.cache_dir.exists():
            self.cache_dir.mkdir()
        log.info(f"Opened cache handler. Cache directory: {self.cache_dir}")
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
        log.info(f"FINN Commit reads: {self.finn_commit}")

        # FINN HLSLIB Commit
        self.hlslib_commit = subprocess.run(
            shlex.split("git rev-parse HEAD"),
            text=True,
            capture_output=True,
            cwd=get_deps_path() / "finn-hlslib",
        ).stdout.strip()
        log.info(f"HLSLIB Commit reads: {self.hlslib_commit}")

        # HLS Clk and device
        self.clk = hls_clk_period
        self.fpgapart = fpgapart

    def _get_key_part_attributes(self, op: HWCustomOp) -> str:
        """Return the part of the key that contains attributes and their values."""
        key_part = ""
        typ = type(op)
        attrs: list[str] = []
        if "use" in CACHE_IP_DEFINITIONS[typ].keys():
            attrs = CACHE_IP_DEFINITIONS[typ]["use"]
        elif "ignore" in CACHE_IP_DEFINITIONS[typ].keys():
            attrs = [
                k
                for k in op.get_nodeattr_types().keys()
                if k not in CACHE_IP_DEFINITIONS[typ]["ignore"]
            ]
        else:
            raise FINNInternalError("This codepath should not be reachable!")
        for attr in attrs:
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
            key_part += f"{attr}:{data}\n"
        return key_part

    def _get_key_part_parameter(self, op: HWCustomOp, model: ModelWrapper) -> str:
        """Get the key part defined by the op parameters.

        If, for example, weights, are embedded into the operators, they need to
        be part of the hashed key as well.
        """

        def ndarray_to_bytes(tensor: Any) -> bytes:
            cont = np.ascontiguousarray(tensor)
            assert type(tensor) is np.ndarray
            return cont.tobytes() + str(tensor.shape).encode("UTF-8")

        if isinstance(op, (MVAU, VVAU)):
            mem_mode = None
            try:
                mem_mode = op.get_nodeattr("mem_mode")
            except Exception as e:
                raise FINNInternalError(
                    f"Cannot cache {op.onnx_node.name} because op is of "
                    f"type MVAU but has no mem_mode set!"
                ) from e
            if mem_mode in ["internal_embedded", "internal_decoupled"]:
                weightbytes = ndarray_to_bytes(model.get_initializer(op.onnx_node.input[1]))
                threshbytes = ndarray_to_bytes(model.get_initializer(op.onnx_node.input[2]))
                array_hash = self.hasher(weightbytes + threshbytes).hexdigest()
                return f"param_hash:{array_hash}\n"
        elif isinstance(op, (Thresholding, ChannelwiseOp, Lookup)):
            parambytes = ndarray_to_bytes(model.get_initializer(op.onnx_node.input[1]))
            array_hash = self.hasher(parambytes).hexdigest()
            return f"param_hash:{array_hash}\n"
        elif isinstance(op, (ElementwiseBinaryOperation,)):
            parambytes0 = ndarray_to_bytes(model.get_initializer(op.onnx_node.input[0]))
            parambytes1 = ndarray_to_bytes(model.get_initializer(op.onnx_node.input[1]))
            array_hash = self.hasher(parambytes0 + parambytes1).hexdigest()
            return f"param_hash:{array_hash}\n"
        elif isinstance(op, ScaledDotProductAttention):
            key_part = ""
            if op.get_nodeattr("ActQKMatMul") == "thresholds":
                thresholds = model.get_initializer(
                    op.get_input_name_by_name("thresholds_qk_matmul")
                )
                hashed = self.hasher(ndarray_to_bytes(thresholds)).hexdigest()
                key_part += f"thresholds_qk_matmul:{hashed}\n"
            if op.get_nodeattr("ActASoftmax") == "thresholds":
                thresholds = model.get_initializer(
                    op.get_input_name_by_name("thresholds_a_softmax")
                )
                hashed = self.hasher(ndarray_to_bytes(thresholds)).hexdigest()
                key_part += f"thresholds_a_softmax:{hashed}\n"
            if op.get_nodeattr("ActAVMatMul") == "thresholds":
                thresholds = model.get_initializer(
                    op.get_input_name_by_name("thresholds_av_matmul")
                )
                hashed = self.hasher(ndarray_to_bytes(thresholds)).hexdigest()
                key_part += f"thresholds_av_matmul:{hashed}\n"
            if op.get_nodeattr("mask_mode") == "const":
                mask = model.get_initializer(op.get_input_name_by_name("M"))
                hashed = self.hasher(ndarray_to_bytes(mask)).hexdigest()
                key_part += f"M:{hashed}\n"
            return key_part
        return ""

    def get_key(self, op: HWCustomOp, model: ModelWrapper) -> str:
        """Return the key that can be hashed, for the given custom op.

        Returns:
            str: The human-readable key. Can be used to generate the caching
                    hash and the metadata file packed with the cached data.
        """
        # TODO: Maybe exchange simple string concat for something more elegant at some point.
        # TODO: Practical, because we can include the unhashed key in the directory for debugging
        global CACHE_IP_DEFINITIONS
        if type(op) not in CACHE_IP_DEFINITIONS.keys():
            log.error(
                f"Tried getting the key for a non-cacheable custom operator ({type(op).__name__}). "
                "Did you perhaps forget to register the op for caching via "
                "@cache_ip(...)?"
            )

        # Always use the current FINN and HLSLIB commits so that the correct versions are used
        key = f"FINN: {self.finn_commit}\nHLSLIB: {self.hlslib_commit}\n"

        # Two custom ops might need the same attributes, so add the type
        key += "type:" + type(op).__name__ + "\n"

        # Hash synth clk period and part
        if self.cache_hls_clk:
            key += f"hls_clk_period_ns:{self.clk}\n"
        if self.cache_fpgapart:
            key += f"fpgapart:{self.fpgapart}\n"

        # Add all node attributes required
        key += self._get_key_part_attributes(op) + "\n"

        # Add parameters if existing
        key += self._get_key_part_parameter(op, model)

        return key

    def get_hash_hex(self, key: str) -> str:
        """Return the hex repr of the hash of the given key.

        The key can be created using get_key(...)
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

    def _get_node_data(
        self, node: NodeProto, model: ModelWrapper
    ) -> tuple[HWCustomOp, str, str, Path]:
        """Return the op, key, hashed key, cache dir path for a given node."""
        op = getCustomOp(node)
        key = self.get_key(op, model)
        hashed_key = self.get_hash_hex(key)
        return op, key, hashed_key, self._cache_dir_path(hashed_key)

    def get_num_cached_ips(self, model: ModelWrapper) -> int:
        """Return the number of cached IPs in the model."""
        count = 0
        for node in model.graph.node:
            _, _, _, cache_dir = self._get_node_data(node, model)
            if cache_dir.exists():
                count += 1
        return count

    def apply(self, model: ModelWrapper) -> ModelWrapper:
        """Apply all IPs that were cached to the model and return it."""
        for node in model.graph.node:
            op, key, hashed_key, cache_dir = self._get_node_data(node, model)
            if cache_dir.exists():
                self._prepare_from_cached_ip(op, hashed_key, make_copy=True)
        return model

    def update(self, model: ModelWrapper) -> None:
        """Check a model for generated IPs that were not yet cached, and cache them.

        Requires HLSSynthIP() to be run before.
        """

        def attribute_path_exists(name: str, op: HWCustomOp) -> bool:
            try:
                data = op.get_nodeattr(name)
                if data is None or data == "":
                    return False
                return Path(cast(str, data)).exists()
            except Exception:
                return False

        for node in model.graph.node:
            op, key, hashed_key, target_dir = self._get_node_data(node, model)
            if not target_dir.exists():
                # Check to make sure we only cache synthesized IPs
                if is_hls_node(node) or is_rtl_node(node):
                    is_done = (
                        attribute_path_exists("code_gen_dir_ipgen", op)
                        and attribute_path_exists("ip_path", op)
                        and attribute_path_exists("ipgen_path", op)
                    )
                    if not is_done:
                        log.warning(
                            f"Node {node.name} is hasn't been synthesized yet. "
                            f"Cannot cache. Skipping."
                        )
                        continue
                else:
                    log.warning(f"Cannot cache node {node.name}. Node is not a HW node!")
                code_gen_dir = Path(cast(str, op.get_nodeattr("code_gen_dir_ipgen")))
                if not code_gen_dir.exists():
                    log.warning(
                        f"Could not cache {node.name}: code_gen_dir_ipgen not set. "
                        f"Did HLSSynthIP() fail/was not run?"
                    )
                shutil.copytree(code_gen_dir, target_dir, dirs_exist_ok=True)
                self._create_key_file(key, target_dir / "key.txt")
                self._dump_nodeattrs(op, target_dir / "nodeattrs.json")
                typ = ""
                if is_hls_node(node):
                    typ = "HLS"
                elif is_rtl_node(node):
                    typ = "RTL"
                else:
                    typ = "unknown type"
                log.info(
                    f"Cached {typ} node {node.name}. "
                    f"Cached at: {target_dir} from {code_gen_dir}!"
                )


class CachedIPGen(Transformation):
    """(PrepareIP and) HLSSynth but cached."""

    def __init__(
        self,
        hash_function: str,
        include_prepare_ip: bool,
        cache_clock: bool,
        clk: float,
        cache_fpgapart: bool,
        fpgapart: str,
    ) -> None:
        """(PrepareIP and) HLSSynth but cached.

        Args:
            hash_function: Hashfunction to use.
            include_prepare_ip: If True, also run PrepareIP before synthesis.
            fpgapart: Required if PrepareIP is being run.
            clk: Required if PrepareIP is being run.
        """
        super().__init__()
        self.hashfunc = hash_function
        self.prepareip = include_prepare_ip
        self.part = fpgapart
        self.cache_part = cache_fpgapart
        self.clk = clk
        self.cache_clock = cache_clock

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply cached HLS Synthesis (and PrepareIP)."""
        cache = IPCache(
            cache_dir=get_cache_path(),
            hashfunc=self.hashfunc,
            hls_clk_period=self.clk,
            cache_hls_clk=self.cache_clock,
            fpgapart=self.part,
            cache_fpgapart=self.cache_part,
        )
        log.info(
            f"Applying cache to {cache.get_num_cached_ips(model)} "
            f"/ {len(model.graph.node)} nodes!"
        )
        model = cache.apply(model)
        if self.prepareip:
            if self.part is None or self.clk is None:
                raise FINNInternalError(
                    "Cannot run PrepareIP in CachedIPGen without " "fpgapart and clk being passed!"
                )
            log.info("Running PrepareIP for uncached IPs...")
            model = model.transform(PrepareIP(self.part, self.clk))
            cache.update(model)
        log.info("Running synthesis for uncached IPs...")
        model = model.transform(HLSSynthIP())
        log.info("Updating cache with newly generated IPs...")
        cache.update(model)
        return model, False
