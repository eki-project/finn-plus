"""Manage IP caching for FINN."""

from __future__ import annotations

import contextlib
import hashlib
import json
import numpy as np
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_num_default_workers
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


# UTILITY FUNCTIONS
def _ndarray_to_bytes(tensor: Any) -> bytes:
    cont = np.ascontiguousarray(tensor)
    assert type(tensor) is np.ndarray
    return cont.tobytes() + str(tensor.shape).encode("UTF-8")


def _attribute_path_exists(name: str, op: HWCustomOp) -> bool:
    """Check that the node attribute path exists.
    If the node attribute cannot be loaded, return False."""  # noqa
    try:
        data = op.get_nodeattr(name)
        if data is None or data == "":
            return False
        return Path(cast(str, data)).exists()
    except Exception:
        return False


def _check_path_lengths_okay(
    pc_name_max: int, pc_path_max: int, hashed_key: str, target_dir: Path
) -> bool:
    """Check if we follow the path length limits. If not return False, otherwise True."""
    if len(hashed_key) > pc_name_max:
        log.error(
            f"Cannot cache an IP: The hash hex representation "
            f"is too long to be allowed as a filename on your "
            f"system (best effort detected limit: "
            f"{pc_name_max}). Skipping caching."
        )
        return False
    path_bytes = len(str(target_dir.absolute()).encode("UTF-8"))
    if path_bytes > pc_path_max:
        log.error(
            f"Cannot cache an IP: the generated path length of "
            f"the cache location is not allowed on your system! "
            f"The best effort detected limit is: "
            f"{pc_path_max} bytes, the path length is "
            f"{path_bytes} bytes. Skipping caching."
        )
        return False
    return True


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
                "gen_top_module",
                "ip_vlnv",
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

    Public methods that are relevant for the caches usage:
    - `model = cache.apply(model)`: Fetch cached IPs and apply them to the model,
        returning the new model
    - `cache.update(model)`: Update the cache by adding synthesized IPs that are not
        yet cached into the cache.
    - `cache.get_key(op, model)`: Get the key (string) of the given custom op
    - `cache.get_hash_hex(key)`: Get the hex representation of the hash of the given key.
    - `cache.get_num_cached_ips(model)`: Get the number of cached IPs in the given model.
    """

    allowed_hashfuncs: Final[list[str]] = ["sha256", "sha512", "blake2s", "blake2b"]

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
            hls_clk_period: HLS clock period in ns.
            cache_hls_clk: Use the HLS clock as part of the key.
            fpgapart: FPGA-part used for HLSSynth and PrepareIP.
            cache_fpgapart: Use the fpgapart as part of the key.
        """
        self.cache_dir = cache_dir
        self.cache_hls_clk = cache_hls_clk
        self.cache_fpgapart = cache_fpgapart

        # Used to check validity of cache directory names
        if sys.platform != "win32":
            self.max_hash_len = os.pathconf("/", "PC_NAME_MAX")
            self.max_path_len = os.pathconf("/", "PC_PATH_MAX")
        else:
            # TODO: Implement filesystem checks
            # 256 seems to be the default max path length under windows
            self.max_hash_len = 256
            self.max_path_len = 256

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
                weightbytes = _ndarray_to_bytes(model.get_initializer(op.onnx_node.input[1]))
                try:
                    threshbytes = _ndarray_to_bytes(model.get_initializer(op.onnx_node.input[2]))
                except IndexError:
                    # No thresholds
                    threshbytes = b""
                array_hash = self.hasher(weightbytes + threshbytes).hexdigest()
                return f"param_hash:{array_hash}\n"
        elif isinstance(op, (Thresholding, ChannelwiseOp, Lookup)):
            parambytes = _ndarray_to_bytes(model.get_initializer(op.onnx_node.input[1]))
            array_hash = self.hasher(parambytes).hexdigest()
            return f"param_hash:{array_hash}\n"
        elif isinstance(op, (ElementwiseBinaryOperation,)):
            parambytes0 = _ndarray_to_bytes(model.get_initializer(op.onnx_node.input[0]))
            parambytes1 = _ndarray_to_bytes(model.get_initializer(op.onnx_node.input[1]))
            array_hash = self.hasher(parambytes0 + parambytes1).hexdigest()
            return f"param_hash:{array_hash}\n"
        elif isinstance(op, ScaledDotProductAttention):
            key_part = ""
            if op.get_nodeattr("ActQKMatMul") == "thresholds":
                thresholds = model.get_initializer(
                    op.get_input_name_by_name("thresholds_qk_matmul")
                )
                hashed = self.hasher(_ndarray_to_bytes(thresholds)).hexdigest()
                key_part += f"thresholds_qk_matmul:{hashed}\n"
            if op.get_nodeattr("ActASoftmax") == "thresholds":
                thresholds = model.get_initializer(
                    op.get_input_name_by_name("thresholds_a_softmax")
                )
                hashed = self.hasher(_ndarray_to_bytes(thresholds)).hexdigest()
                key_part += f"thresholds_a_softmax:{hashed}\n"
            if op.get_nodeattr("ActAVMatMul") == "thresholds":
                thresholds = model.get_initializer(
                    op.get_input_name_by_name("thresholds_av_matmul")
                )
                hashed = self.hasher(_ndarray_to_bytes(thresholds)).hexdigest()
                key_part += f"thresholds_av_matmul:{hashed}\n"
            if op.get_nodeattr("mask_mode") == "const":
                mask = model.get_initializer(op.get_input_name_by_name("M"))
                hashed = self.hasher(_ndarray_to_bytes(mask)).hexdigest()
                key_part += f"M:{hashed}\n"
            return key_part
        return ""

    def get_key(self, op: HWCustomOp, model: ModelWrapper) -> str:
        """Return the key that can be hashed, for the given custom op.

        These parts are used to build the key which is then hashed for the cache:
        - FINN commit
        - FINN-HLSLIB commit
        - Custom Op type
        - (Optional) HLS clock
        - (Optional) HLS Synthesis FPGA-part
        - All node attributes that define a unique instance of the operator (set by @cache_ip(...))
        - All external parameters for ops that have these (for example MVAU)
            - These are hashed themselves for brevity, otherwise the key might be megabytes of data

        **IMPORTANT**: Keep in mind that changes in this function will require caching everything
        again.

        Returns:
            str: The human-readable key. Can be used to generate the caching
                    hash and the metadata file packed with the cached data.
        """
        global CACHE_IP_DEFINITIONS
        if type(op) not in CACHE_IP_DEFINITIONS.keys():
            log.error(
                f"Tried getting the key for a non-cacheable custom operator ({type(op).__name__}). "
                "Did you perhaps forget to register the op for caching via "
                "@cache_ip(...)?"
            )
        key = f"FINN: {self.finn_commit}\nHLSLIB: {self.hlslib_commit}\n"
        key += "type:" + type(op).__name__ + "\n"
        if self.cache_hls_clk:
            key += f"hls_clk_period_ns:{self.clk}\n"
        if self.cache_fpgapart:
            key += f"fpgapart:{self.fpgapart}\n"
        key += self._get_key_part_attributes(op) + "\n"
        key += self._get_key_part_parameter(op, model)
        return key

    def get_hash_hex(self, key: str) -> str:
        """Return the hex repr of the hash of the given key."""
        return self.hasher(key.encode("UTF-8")).hexdigest()

    def _create_key_file(self, key: str, path: Path) -> None:
        """Write the given key data into a file at the given path."""
        with path.open("w+") as f:
            f.write(f"Hashed using {self.hashfunc_name}. Key:\n------------------------\n")
            f.write(key)

    def _dump_nodeattrs(
        self, op: HWCustomOp, path: Path, additional_attributes: list[str] | None = None
    ) -> None:
        """Dump the custom ops node attributes at the given path as a JSON.

        If a node attribute cannot be accessed, it is silently ignored.

        Args:
            op: The HWCustom op of which the node attributes are the target
            path: Where to dump the node attributes
            additional_attributes: A list of additional attribute keys that
                should be included in the dump.
        """
        if additional_attributes is None:
            additional_attributes = []
        required = {"ip_vlnv", "gen_top_module", *additional_attributes}
        d = {}
        for name in op.get_nodeattr_types().keys():
            if name in required:
                try:
                    d[name] = op.get_nodeattr(name)
                except Exception:
                    continue
        with path.open("w+") as f:
            json.dump(d, f)

    @staticmethod
    def _prepare_from_cached_ip(
        op: HWCustomOp, hashed_key: str, make_copy: bool, cache_dir: Path
    ) -> None:
        """Prepare the given custom op for usage of the given cached IP.

        We have to set some node attributes normally set by HLSSynth and PrepareIP. This needs to
        be done to use the cached IP.

        Args:
            op: The operator of which the node attributes we have to modify.
            hashed_key: The hash hex repr of the key for this op. Used to find the cached IP.
            make_copy: If True, first makes a copy of the cached IP in the current FINN_BUILD_DIR
                        and sets the path towards this copy instead of the cached original.
            cache_dir: FINN_IP_CACHE directory, as passed from the calling IPCache instance.
        """
        log.info(f"Preparing {op.onnx_node.name} from cached IP (hashed key: {hashed_key[:10]}...)")
        ip_dir = cache_dir / hashed_key
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

        # If needed insert gen_top_module. If not saved or the attr doesnt exist ignore
        with contextlib.suppress(Exception):
            op.set_nodeattr("gen_top_module", saved_nodeattrs["gen_top_module"])

    def _get_node_data(
        self, node: NodeProto, model: ModelWrapper
    ) -> tuple[HWCustomOp, str, str, Path]:
        """Return the op, key, hashed key, cache dir path for a given node."""
        op = getCustomOp(node)
        key = self.get_key(op, model)
        hashed_key = self.get_hash_hex(key)
        return op, key, hashed_key, self.cache_dir / hashed_key

    def _is_op_synthesized(self, op: HWCustomOp) -> bool:
        """Return whether the given op is synthesized. This is derived from the existence and
        validity of the paths in code_gen_dir_ipgen, ipgen_path and ip_path."""  # noqa
        return (
            _attribute_path_exists("code_gen_dir_ipgen", op)
            and _attribute_path_exists("ip_path", op)
            and _attribute_path_exists("ipgen_path", op)
        )

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
        with ThreadPoolExecutor(max_workers=get_num_default_workers()) as pool:
            for node in model.graph.node:
                op, key, hashed_key, op_cache_dir = self._get_node_data(node, model)
                if op_cache_dir.exists():
                    pool.submit(
                        IPCache._prepare_from_cached_ip,
                        op=op,
                        hashed_key=hashed_key,
                        make_copy=True,
                        cache_dir=self.cache_dir,
                    )
            pool.shutdown(wait=True)
        return model

    def update(self, model: ModelWrapper) -> None:
        """Check a model for generated IPs that were not yet cached, and cache them.

        Requires HLSSynthIP() to be run before.
        """
        total_cached = 0
        for node in model.graph.node:
            op, key, hashed_key, target_dir = self._get_node_data(node, model)
            if not _check_path_lengths_okay(
                self.max_hash_len, self.max_path_len, hashed_key, target_dir
            ):
                return
            if not (is_hls_node(node) or is_rtl_node(node)):
                log.warning(f"Cannot cache node {node.name}. Node is not a HW node!")
                continue
            if not target_dir.exists():
                if not self._is_op_synthesized(op):
                    log.warning(
                        f"{node.name} hasn't been synthesized yet and can't be cached "
                        f"(one of code_gen_dir_ipgen, ip_path, ipgen_path is missing or "
                        f"invalid!). Hash after synthesis will be: {hashed_key}"
                    )
                    continue
                code_gen_dir = Path(cast(str, op.get_nodeattr("code_gen_dir_ipgen")))
                if not code_gen_dir.exists():
                    log.warning(
                        f"Could not cache {node.name}: code_gen_dir_ipgen not set. "
                        f"Did HLSSynthIP() fail/was not run?"
                    )
                shutil.copytree(code_gen_dir, target_dir, dirs_exist_ok=True)
                self._create_key_file(key, target_dir / "key.txt")
                self._dump_nodeattrs(op, target_dir / "nodeattrs.json")
                log.info(f"Cached node {node.name}. Cached at: {target_dir} from {code_gen_dir}!")
                total_cached += 1
        log.info(f"Cached a total of {total_cached} new ops.")


class CachedIPGen(Transformation):
    """(PrepareIP and) HLSSynth but cached."""

    def __init__(
        self,
        hash_function: str,
        include_prepare_ip: bool,
        clk: float,
        cache_clock: bool,
        fpgapart: str,
        cache_fpgapart: bool,
    ) -> None:
        """(PrepareIP and) HLSSynth but cached.

        Args:
            hash_function: Hashfunction to use.
            include_prepare_ip: If True, also run PrepareIP before synthesis.
            fpgapart: Required if PrepareIP is being run.
            cache_fpgapart: Whether or not to use the fpgapart for the cache ky
            clk: Required if PrepareIP is being run.
            cache_clock: Whether or not to use the clock for the cache key
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
                    "Cannot run PrepareIP in CachedIPGen without fpgapart and clk being passed!"
                )
            log.info("Running PrepareIP for uncached IPs...")
            model = model.transform(PrepareIP(self.part, self.clk))
            cache.update(model)
        log.info("Running synthesis for uncached IPs...")
        model = model.transform(HLSSynthIP())
        log.info("Updating cache with newly generated IPs...")
        cache.update(model)
        return model, False
