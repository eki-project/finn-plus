"""IP cache: re-use code generation and HLS synthesis results across FINN+ builds.

Generating the IP of a dataflow layer (``PrepareIP`` followed by ``HLSSynthIP`` for HLS layers)
is, after the final synthesis itself, the most time-consuming part of a build. Its result only
depends on a well-defined set of inputs, so it can be stored on disk and restored in later builds:

* the FINN+ code (package version, git commit and local modifications, if available) and the
  HLS library dependencies (``finn-hlslib``, ``attention-hlslib``),
* the Xilinx tool version, the FPGA part and the HLS clock period,
* the node type, the node name (which is baked into module and IP names), all node attributes
  that influence code generation, the shapes and datatypes of all input and output tensors and
  the contents of all initializers (weights, thresholds, ...) feeding the node.

All of these are combined into a human-readable key which is hashed (SHA-256) to obtain the name
of the cache entry directory. An entry is a pruned copy of the node's ``code_gen_dir_ipgen``
(large HLS project internals such as ``.autopilot`` are dropped) plus ``ip_cache_key.txt`` and
``ip_cache_meta.json``. Entries are written atomically (copy to a staging directory, then rename),
so several builds or tests may share one cache directory concurrently.

Integration into the build flow:

* ``RestoreCachedIPs`` runs before ``PrepareIP`` (``step_hw_codegen``). Nodes with a cache hit get
  a private copy of the entry in ``FINN_BUILD_DIR`` and are subsequently skipped by ``PrepareIP``
  and ``HLSSynthIP``. The keys of all cache misses are remembered in a model metadata property.
* ``StoreGeneratedIPs`` runs after ``HLSSynthIP`` (``step_hw_ipgen``) and stores the newly
  generated IPs under the remembered keys. Keys are computed *before* code generation on purpose,
  since some operators rewrite their own attributes while generating code.

The cache location is configured via the ``FINN_IP_CACHE`` setting (``settings.yaml``, environment
variable or ``finn build --ip-cache-path``); ``DataflowBuildConfig.use_ip_cache`` switches the
cache on or off per build.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import os
import re
import shutil
import socket
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from onnx import AttributeProto
from pathlib import Path
from qonnx.transformation.base import Transformation
from typing import TYPE_CHECKING, Any, Literal

import finn
from finn.interface.settings import IP_CACHE_DISABLED_VALUES
from finn.util.basic import getHWCustomOp, make_build_dir
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.logging import log
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

#: Name of the human-readable key file inside a cache entry.
CACHE_KEY_FILE = "ip_cache_key.txt"
#: Name of the metadata file inside a cache entry. Its presence marks a complete entry.
CACHE_META_FILE = "ip_cache_meta.json"
#: Model metadata property carrying the keys of cache misses from restoring to storing.
PENDING_KEYS_METADATA_PROP = "finn_ip_cache_pending_keys"
#: Version of the on-disk entry format. Bump when the layout of an entry changes.
CACHE_FORMAT_VERSION = 1

#: Node attributes that do not influence the generated code/IP and are thus excluded from the key.
#: Everything else is included by default, so new attributes are covered automatically.
IGNORED_NODEATTRS: frozenset[str] = frozenset(
    {
        # Paths and results filled in by code generation, synthesis and simulation
        "code_gen_dir_ipgen",
        "code_gen_dir_cppsim",
        "ipgen_path",
        "ip_path",
        "ip_vlnv",
        "gen_top_module",
        "executable_path",
        "rtlsim_so",
        "rtlsim_trace",
        "cycles_rtlsim",
        "cycles_estimate",
        "res_estimate",
        "res_synth",
        "res_hls",
        "exec_mode",
        "output_hook",
        # Only relevant for specialization (already happened) and floorplanning/stitching
        "backend",
        "preferred_impl_style",
        "slr",
        "mem_port",
        "partition_id",
        "device_id",
        # FIFO depths only affect the separately inserted FIFO nodes
        "inFIFODepths",
        "outFIFODepths",
        # Characterization results
        "io_chrc_in",
        "io_chrc_out",
        "io_chrc_period",
        "io_chrc_pads_in",
        "io_chrc_pads_out",
    }
)

#: Operators wrapping whole subgraphs are never cached, their IP depends on the subgraph.
UNCACHEABLE_OP_TYPES: frozenset[str] = frozenset(
    {"FINNLoop", "NodeContainer", "DNNContainer", "StreamingDataflowPartition"}
)

#: Directory names that are never copied into the cache (HLS project databases etc.).
COPY_IGNORE_DIRS: frozenset[str] = frozenset({".autopilot", ".debug", "xsim.dir", "__pycache__"})
#: File suffixes that are never copied into the cache (compiled simulation objects and traces).
#: Note that .npy files must be kept, some operators generate them during code generation
#: (e.g. the weight stream of MVAUs with decoupled weights) and read them during rtlsim.
COPY_IGNORE_SUFFIXES: frozenset[str] = frozenset({".so", ".o", ".wdb", ".vcd"})

#: Attributes restored from the metadata of a cache entry (if the operator defines them).
RESTORED_NODEATTRS: tuple[str, ...] = ("ipgen_path", "ip_path", "ip_vlnv", "gen_top_module")

_GIT_TIMEOUT_S = 30
_REWRITE_MAX_FILE_SIZE = 256 * 1024 * 1024


def get_ip_cache_dir() -> Path | None:
    """Return the configured IP cache directory or None if IP caching is disabled.

    The directory comes from the global FINN+ settings (``FINN_IP_CACHE``). If no settings are
    available (e.g. transformations used outside of the FINN+ CLI), the ``FINN_IP_CACHE``
    environment variable is consulted directly.
    """
    try:
        return get_settings().finn_ip_cache
    except FINNUserError:
        value = os.environ.get("FINN_IP_CACHE", "")
        if value.strip().lower() in IP_CACHE_DISABLED_VALUES:
            return None
        return Path(value).expanduser().absolute()


def _run_git(args: list[str], cwd: Path) -> str | None:
    """Run a git command and return its stripped stdout, or None if it fails."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _find_source_repo() -> Path | None:
    """Return the root of the git repository the running FINN+ package is loaded from.

    Only accepts a repository whose ``src/finn`` directory is the package that is actually
    imported, so that a virtual environment living inside some unrelated repository is not
    mistaken for the FINN+ sources (as would happen for a non-editable installation).
    """
    package_dir = Path(finn.__file__).resolve().parent
    for candidate in package_dir.parents:
        if (candidate / ".git").exists():
            if (candidate / "src" / "finn").resolve() == package_dir:
                return candidate
            return None
    return None


def _source_identity() -> tuple[str, str]:
    """Return (commit, local_changes) describing the FINN+ sources being executed.

    ``local_changes`` is a digest of uncommitted modifications to the code-generating parts of the
    repository, so that a developer editing an operator does not get IPs of the unmodified version.
    Both values are "unknown" if the package does not come from a git checkout.
    """
    repo = _find_source_repo()
    if repo is None:
        return "unknown", "unknown"
    commit = _run_git(["rev-parse", "HEAD"], repo)
    if commit is None:
        return "unknown", "unknown"
    paths = ["--", "src", "custom_hls", "finn-rtllib", "finn_xsi"]
    diff = _run_git(["diff", "HEAD", *paths], repo)
    untracked = _run_git(["ls-files", "--others", "--exclude-standard", *paths], repo)
    if diff is None or untracked is None:
        return commit, "unknown"
    if diff == "" and untracked == "":
        return commit, "none"
    digest = hashlib.sha256((diff + "\n" + untracked).encode("utf-8")).hexdigest()
    return commit, digest


def _package_version() -> str:
    """Return the installed finn-plus package version (contains the commit for dev versions)."""
    try:
        from importlib.metadata import version

        return version("finn-plus")
    except Exception:
        return "unknown"


def _dependency_commit(name: str) -> str:
    """Return the checked out commit of a git dependency in FINN_DEPS, or "unknown"."""
    try:
        dep_dir = get_settings().finn_deps / name
    except FINNUserError:
        return "unknown"
    if not (dep_dir / ".git").exists():
        return "unknown"
    return _run_git(["rev-parse", "HEAD"], dep_dir) or "unknown"


def _tool_version(envvar: str) -> str:
    """Return the Xilinx tool version derived from the given environment variable."""
    value = os.environ.get(envvar)
    if value is None:
        return "unknown"
    match = re.search(r"(\d{4}\.\d(?:\.\d+)?)", value)
    return match.group(1) if match is not None else value


@lru_cache(maxsize=1)
def get_environment_identity() -> dict[str, str]:
    """Return everything about the environment that influences generated IPs.

    Computed once per process (the result is cached), since it involves several git calls.
    Use ``get_environment_identity.cache_clear()`` to force a re-evaluation.
    """
    commit, local_changes = _source_identity()
    identity = {
        "finn_plus_version": _package_version(),
        "finn_plus_commit": commit,
        "finn_plus_local_changes": local_changes,
    }
    for dep in ("finn-hlslib", "attention-hlslib"):
        identity[f"dependency {dep}"] = _dependency_commit(dep)
    identity["xilinx_vivado"] = _tool_version("XILINX_VIVADO")
    identity["xilinx_hls"] = _tool_version("XILINX_HLS")
    unknown = [k for k, v in identity.items() if v == "unknown"]
    if unknown:
        log.warning(
            f"IP cache: could not determine {', '.join(unknown)}. Cached IPs will be re-used "
            "regardless of changes to these, make sure they are consistent between builds."
        )
    return identity


def is_cacheable_node(node: NodeProto) -> bool:
    """Return whether the IP of the given node can be cached.

    Only HLS and RTL backend nodes that do not wrap subgraphs are cacheable.
    """
    if not (is_hls_node(node) or is_rtl_node(node)):
        return False
    if node.op_type in UNCACHEABLE_OP_TYPES:
        return False
    return not any(
        attr.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS) for attr in node.attribute
    )


def _array_digest(array: Any) -> str:
    """Return a short, deterministic description of a numpy array including a content digest."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256(contiguous.tobytes()).hexdigest()
    return f"sha256={digest} dtype={contiguous.dtype} shape={contiguous.shape}"


def _value_repr(value: Any) -> str:
    """Return a deterministic string representation of a node attribute value."""
    if isinstance(value, np.ndarray):
        return _array_digest(value)
    return repr(value)


def _has_valid_code_gen_dir(op: HWCustomOp) -> bool:
    """Return whether the operator already points to an existing code generation directory."""
    code_gen_dir = str(op.get_nodeattr("code_gen_dir_ipgen"))
    return code_gen_dir != "" and Path(code_gen_dir).is_dir()


def _is_ip_generated(node: NodeProto, op: HWCustomOp) -> bool:
    """Return whether code generation (and for HLS nodes: IP generation) has completed."""
    if not _has_valid_code_gen_dir(op):
        return False
    code_gen_dir = str(op.get_nodeattr("code_gen_dir_ipgen"))
    if not any(Path(code_gen_dir).iterdir()):
        return False
    if is_hls_node(node):
        ipgen_path = str(op.get_nodeattr("ipgen_path"))
        ip_path = str(op.get_nodeattr("ip_path"))
        return (
            ipgen_path.startswith(code_gen_dir)
            and Path(ipgen_path).is_dir()
            and ip_path.startswith(code_gen_dir)
            and Path(ip_path).is_dir()
        )
    return True


def _copy_ignore(_directory: str, names: list[str]) -> list[str]:
    """Ignore function for ``shutil.copytree`` dropping large or build-specific artifacts."""
    return [
        name
        for name in names
        if name in COPY_IGNORE_DIRS or Path(name).suffix in COPY_IGNORE_SUFFIXES
    ]


def _rewrite_paths(root: Path, old: str, new: str) -> int:
    """Replace all occurrences of the path ``old`` with ``new`` in all files below ``root``.

    Works on raw bytes, so that binary files are handled gracefully. Returns the number of
    modified files.
    """
    modified = 0
    old_bytes = old.encode("utf-8")
    new_bytes = new.encode("utf-8")
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.stat().st_size > _REWRITE_MAX_FILE_SIZE:
            continue
        content = path.read_bytes()
        if old_bytes not in content:
            continue
        path.write_bytes(content.replace(old_bytes, new_bytes))
        modified += 1
    return modified


def _num_workers(num_jobs: int) -> int:
    """Return the number of threads to use for copying ``num_jobs`` entries."""
    try:
        workers = get_settings().num_default_workers
    except FINNUserError:
        workers = int(os.environ.get("NUM_DEFAULT_WORKERS", "4"))
    return max(1, min(workers, num_jobs))


class IPCache:
    """Access to one IP cache directory: key computation, restoring and storing of entries."""

    def __init__(self, cache_dir: Path, fpgapart: str, clk_ns: float) -> None:
        """Open (and create if needed) the cache at ``cache_dir`` for the given target.

        Args:
            cache_dir: Root directory of the cache.
            fpgapart: FPGA part the IPs are generated for.
            clk_ns: HLS clock period in ns the IPs are generated for.
        """
        self.cache_dir = Path(cache_dir)
        self.fpgapart = fpgapart
        self.clk_ns = float(clk_ns)
        self.environment = get_environment_identity()
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise FINNUserError(
                f"IP cache directory {self.cache_dir} cannot be created: {e}. "
                "Change the FINN_IP_CACHE setting or disable IP caching."
            ) from e

    # --- Keys -----------------------------------------------------------------------------------

    def build_key(self, node: NodeProto, model: ModelWrapper) -> str:
        """Return the human-readable key uniquely describing the IP generated for ``node``.

        **Important**: Changing what is included here changes the keys of all entries, i.e.
        existing caches are effectively invalidated.
        """
        op = getHWCustomOp(node)
        lines = [f"{name}: {value}" for name, value in self.environment.items()]
        lines += [
            f"fpgapart: {self.fpgapart}",
            f"hls_clk_period_ns: {self.clk_ns!r}",
            f"op_type: {node.op_type}",
            f"domain: {node.domain}",
            f"node_name: {node.name}",
        ]
        for name in sorted(op.get_nodeattr_types()):
            if name in IGNORED_NODEATTRS:
                continue
            try:
                value = _value_repr(op.get_nodeattr(name))
            except Exception:
                value = "<unset>"
            lines.append(f"attr {name}: {value}")
        for index, tensor in enumerate(node.input):
            lines.append(f"input[{index}]: {self._tensor_description(tensor, model)}")
            initializer = model.get_initializer(tensor)
            if initializer is not None:
                lines.append(f"input[{index}] initializer: {_array_digest(initializer)}")
        for index, tensor in enumerate(node.output):
            lines.append(f"output[{index}]: {self._tensor_description(tensor, model)}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _tensor_description(tensor: str, model: ModelWrapper) -> str:
        """Return shape and datatype of a tensor as part of the key."""
        shape = model.get_tensor_shape(tensor)
        datatype = model.get_tensor_datatype(tensor)
        return f"shape={shape} datatype={datatype.name}"

    @staticmethod
    def key_hash(key: str) -> str:
        """Return the hex digest identifying the cache entry for the given key."""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    # --- Entries --------------------------------------------------------------------------------

    def entry_dir(self, key_hash: str) -> Path:
        """Return the directory of the entry with the given hash (may not exist)."""
        return self.cache_dir / key_hash

    def has_entry(self, key_hash: str) -> bool:
        """Return whether a complete entry with the given hash exists."""
        return (self.entry_dir(key_hash) / CACHE_META_FILE).is_file()

    def num_entries(self) -> int:
        """Return the number of complete entries in the cache."""
        return sum(
            1
            for path in self.cache_dir.iterdir()
            if not path.name.startswith(".") and (path / CACHE_META_FILE).is_file()
        )

    def restore(self, node: NodeProto, key_hash: str) -> dict[str, str]:
        """Copy the entry into a fresh code generation directory in FINN_BUILD_DIR.

        Only performs file operations, so it can run in a worker thread. Returns the node
        attributes that have to be set to use the restored IP.
        """
        entry = self.entry_dir(key_hash)
        meta = json.loads((entry / CACHE_META_FILE).read_text())
        if meta.get("format_version") != CACHE_FORMAT_VERSION:
            raise FINNInternalError(
                f"IP cache entry {entry} has format version {meta.get('format_version')}, "
                f"expected {CACHE_FORMAT_VERSION}. Delete the cache directory to recreate it."
            )
        new_dir = str(make_build_dir(prefix=f"code_gen_ipgen_{node.name}_"))
        shutil.copytree(
            entry,
            new_dir,
            dirs_exist_ok=True,
            ignore=lambda directory, names: (
                [n for n in names if n in (CACHE_KEY_FILE, CACHE_META_FILE)]
                if Path(directory) == entry
                else []
            ),
        )
        old_dir = meta["code_gen_dir_ipgen"]
        rewritten = _rewrite_paths(Path(new_dir), old_dir, new_dir)
        if rewritten > 0:
            log.debug(
                f"IP cache: rewrote the original build directory path in {rewritten} "
                f"file(s) of restored entry for {node.name}"
            )
        attributes = {"code_gen_dir_ipgen": new_dir}
        for name, value in meta["nodeattrs"].items():
            attributes[name] = value.replace(old_dir, new_dir)
        return attributes

    def store(
        self, node: NodeProto, key: str, key_hash: str, code_gen_dir: str, nodeattrs: dict[str, str]
    ) -> bool:
        """Add the generated IP of ``node`` to the cache. Returns False if it was already cached.

        The entry is assembled in a staging directory and atomically renamed into place, so
        concurrent builds never observe partial entries; whoever renames first wins.
        """
        if self.has_entry(key_hash):
            return False
        staging_root = self.cache_dir / ".staging"
        staging_root.mkdir(exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f"{key_hash[:16]}_", dir=staging_root))
        try:
            shutil.copytree(code_gen_dir, staging, dirs_exist_ok=True, ignore=_copy_ignore)
            (staging / CACHE_KEY_FILE).write_text(key)
            meta = {
                "format_version": CACHE_FORMAT_VERSION,
                "key_hash": key_hash,
                "op_type": node.op_type,
                "domain": node.domain,
                "node_name": node.name,
                "code_gen_dir_ipgen": code_gen_dir,
                "nodeattrs": nodeattrs,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "created_on": socket.gethostname(),
                "finn_plus_version": self.environment["finn_plus_version"],
            }
            (staging / CACHE_META_FILE).write_text(json.dumps(meta, indent=2))
            try:
                staging.rename(self.entry_dir(key_hash))
            except OSError:
                if self.has_entry(key_hash):
                    return False
                raise
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        return True


def _load_pending_keys(model: ModelWrapper) -> dict[str, dict[str, str]]:
    """Return the keys of cache misses remembered by ``RestoreCachedIPs``."""
    value = model.get_metadata_prop(PENDING_KEYS_METADATA_PROP)
    return json.loads(value) if value else {}


def _set_pending_keys(model: ModelWrapper, pending: dict[str, dict[str, str]]) -> None:
    """Remember (or forget, if empty) the keys of cache misses in the model metadata."""
    props = model.graph.metadata_props
    for index, prop in enumerate(props):
        if prop.key == PENDING_KEYS_METADATA_PROP:
            del props[index]
            break
    if pending:
        model.set_metadata_prop(PENDING_KEYS_METADATA_PROP, json.dumps(pending))


class RestoreCachedIPs(Transformation):
    """Restore previously generated IPs from the IP cache. Run before ``PrepareIP``.

    Nodes with a cache hit get their ``code_gen_dir_ipgen``, ``ipgen_path``, ``ip_path`` (and
    ``ip_vlnv``/``gen_top_module``) attributes set to a private copy of the cached entry, so that
    ``PrepareIP`` and ``HLSSynthIP`` skip them. The keys of all other nodes are remembered in the
    model metadata for ``StoreGeneratedIPs``. Does nothing if no cache directory is configured.
    """

    def __init__(self, fpgapart: str, clk_ns: float, cache_dir: Path | None = None) -> None:
        """Initialize with the target FPGA part, HLS clock period and an optional cache directory.

        If ``cache_dir`` is None, the directory from the FINN+ settings is used.
        """
        super().__init__()
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.cache_dir = cache_dir

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, Literal[False]]:
        """Restore all cache hits and remember the keys of all misses."""
        cache_dir = self.cache_dir if self.cache_dir is not None else get_ip_cache_dir()
        if cache_dir is None:
            log.debug("IP cache: no cache directory configured, nothing to restore.")
            return model, False
        cache = IPCache(cache_dir, self.fpgapart, self.clk_ns)

        hits: list[tuple[NodeProto, str]] = []
        pending: dict[str, dict[str, str]] = {}
        for node in model.graph.node:
            if not is_cacheable_node(node) or _has_valid_code_gen_dir(getHWCustomOp(node)):
                continue
            key = cache.build_key(node, model)
            key_hash = cache.key_hash(key)
            if cache.has_entry(key_hash):
                hits.append((node, key_hash))
            else:
                pending[node.name] = {"hash": key_hash, "key": key}

        if hits:
            with ThreadPoolExecutor(max_workers=_num_workers(len(hits))) as pool:
                futures = [(node, pool.submit(cache.restore, node, h)) for node, h in hits]
            for node, future in futures:
                op = getHWCustomOp(node)
                attributes = future.result()
                for name, value in attributes.items():
                    if name in op.get_nodeattr_types():
                        op.set_nodeattr(name, value)
                log.debug(f"IP cache: restored {node.name} into {attributes['code_gen_dir_ipgen']}")
        _set_pending_keys(model, pending)

        log.info(
            f"IP cache ({cache.cache_dir}, {cache.num_entries()} entries): restored "
            f"{len(hits)} of {len(hits) + len(pending)} IPs to be generated, "
            f"{len(pending)} will be generated and added to the cache."
        )
        return model, False


class StoreGeneratedIPs(Transformation):
    """Add newly generated IPs to the IP cache. Run after ``HLSSynthIP``.

    Uses the keys remembered by ``RestoreCachedIPs``; nodes without a remembered key (e.g. their
    IP was generated in a previous, uncached build) are keyed by their current state. Does nothing
    if no cache directory is configured.
    """

    def __init__(self, fpgapart: str, clk_ns: float, cache_dir: Path | None = None) -> None:
        """Initialize with the target FPGA part, HLS clock period and an optional cache directory.

        If ``cache_dir`` is None, the directory from the FINN+ settings is used.
        """
        super().__init__()
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.cache_dir = cache_dir

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, Literal[False]]:
        """Store the IPs of all generated nodes that are not in the cache yet."""
        cache_dir = self.cache_dir if self.cache_dir is not None else get_ip_cache_dir()
        if cache_dir is None:
            log.debug("IP cache: no cache directory configured, nothing to store.")
            return model, False
        cache = IPCache(cache_dir, self.fpgapart, self.clk_ns)
        pending = _load_pending_keys(model)

        jobs: list[tuple[NodeProto, str, str, str, dict[str, str]]] = []
        already_cached = 0
        for node in model.graph.node:
            if not is_cacheable_node(node):
                continue
            op = getHWCustomOp(node)
            if not _is_ip_generated(node, op):
                continue
            if node.name in pending:
                key, key_hash = pending[node.name]["key"], pending[node.name]["hash"]
            else:
                key = cache.build_key(node, model)
                key_hash = cache.key_hash(key)
            if cache.has_entry(key_hash):
                already_cached += 1
                continue
            nodeattrs = {
                name: str(op.get_nodeattr(name))
                for name in RESTORED_NODEATTRS
                if name in op.get_nodeattr_types()
            }
            jobs.append(
                (node, key, key_hash, str(op.get_nodeattr("code_gen_dir_ipgen")), nodeattrs)
            )

        stored = 0
        if jobs:
            with ThreadPoolExecutor(max_workers=_num_workers(len(jobs))) as pool:
                futures = [
                    (node, pool.submit(cache.store, node, key, key_hash, code_gen_dir, nodeattrs))
                    for node, key, key_hash, code_gen_dir, nodeattrs in jobs
                ]
            for node, future in futures:
                if future.result():
                    stored += 1
                    log.debug(f"IP cache: stored IP of {node.name}")
                else:
                    already_cached += 1
        _set_pending_keys(model, {})

        log.info(
            f"IP cache ({cache.cache_dir}): stored {stored} newly generated IPs, "
            f"{already_cached} were already cached."
        )
        return model, False
