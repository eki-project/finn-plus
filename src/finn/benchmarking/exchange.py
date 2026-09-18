"""Exchange of benchmark artifacts between CI runners via a shared filesystem.

The build job (cluster), the measurement job (FPGA board) and the result collection job run
on different GitLab runners. Instead of GitLab artifacts (which hit size limits with
hundreds of bitstreams), the large per-run artifacts are exchanged through a directory on
the cluster fileshare that is mounted on all runners, possibly under different paths::

    <FINN_BENCH_EXCHANGE_DIR>/CI_<pipeline id>/
        CREATED                                    # marker, mtime used for retention
        build_artifacts[_followup]/
            runs_output/run_<id>/{reports/, deploy.zip, DONE}
            TASK_<k>_DONE
        measurement_artifacts[_followup]/
            runs_output/run_<id>/{reports/, DONE}

Every job exports ``FINN_BENCH_EXCHANGE_DIR`` from its runner-specific project variable
(``OTUS_EXCHANGE_DIR``, ``BOARD_EXCHANGE_DIR``, ``LOCAL_EXCHANGE_DIR``). If the variable is
unset (local runs, old pipelines), all paths fall back to the previous layout relative to
the working directory (``build_artifacts/...``), so nothing changes for local use.

Completion is signalled with ``DONE`` markers written atomically; consumers never trust a
``deploy.zip`` without one (a SLURM time-out may kill a task mid-archive).

Stdlib only: this module is imported by the PYNQ python on the board and by the bare
python of the collection runner via ``sys.path.insert(0, "<repo>/src")``.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Union

EXCHANGE_ENV_VAR = "FINN_BENCH_EXCHANGE_DIR"
PIPELINE_ENV_VAR = "FINN_BENCH_EXCHANGE_PIPELINE"
DONE_MARKER = "DONE"
CREATED_MARKER = "CREATED"
KINDS = ("build", "measurement")

PathLike = Union[str, os.PathLike]


def exchange_root() -> Optional[Path]:
    """Root of the exchange directory from the environment, None if unset or empty."""
    value = os.environ.get(EXCHANGE_ENV_VAR, "").strip()
    return Path(value) if value else None


def pipeline_dir_name(pipeline_id: Optional[str] = None) -> str:
    """Name of the per-pipeline subdirectory: ``$FINN_BENCH_EXCHANGE_PIPELINE`` if set,
    else ``CI_<pipeline_id>`` (default ``$CI_PIPELINE_ID``), else ``local``."""
    override = os.environ.get(PIPELINE_ENV_VAR, "").strip()
    if override:
        return override
    pipeline_id = pipeline_id or os.environ.get("CI_PIPELINE_ID", "").strip()
    return f"CI_{pipeline_id}" if pipeline_id else "local"


def pipeline_exchange_dir(create: bool = False) -> Optional[Path]:
    """``<root>/<pipeline dir>``, None if no exchange root is configured. With ``create``
    the directory (and its CREATED marker) is created."""
    root = exchange_root()
    if root is None:
        return None
    path = root / pipeline_dir_name()
    if create:
        ensure_dir(root)
        ensure_dir(path)
        marker = path / CREATED_MARKER
        if not marker.exists():
            marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    return path


def _kind_dirname(kind: str, followup: bool) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown artifact kind {kind!r}, expected one of {KINDS}")
    return f"{kind}_artifacts" + ("_followup" if followup else "")


def artifacts_dir(kind: str, followup: bool = False, base: Optional[PathLike] = None) -> Path:
    """Directory holding the ``runs_output`` tree of one kind (``build``/``measurement``):
    inside the pipeline exchange dir if configured, else ``<base>/<kind>_artifacts`` as
    before (``base`` defaults to the working directory)."""
    exchange = pipeline_exchange_dir()
    if exchange is not None:
        return exchange / _kind_dirname(kind, followup)
    return Path(base or ".") / _kind_dirname(kind, followup)


def run_dir(
    kind: str, run_id: int, followup: bool = False, base: Optional[PathLike] = None
) -> Path:
    """``<artifacts dir>/runs_output/run_<id>``."""
    return artifacts_dir(kind, followup, base) / "runs_output" / f"run_{int(run_id)}"


def list_run_ids(
    kind: str,
    followup: bool = False,
    base: Optional[PathLike] = None,
    require_done: Optional[bool] = None,
) -> list[int]:
    """Sorted run ids found under ``<artifacts dir>/runs_output``; [] if it does not exist.

    ``require_done=True`` only returns runs with a DONE marker, ``False`` all runs, ``None``
    (default) requires the marker only if at least one run has one (old trees without
    markers are consumed as before).
    """
    runs_root = artifacts_dir(kind, followup, base) / "runs_output"
    if not runs_root.is_dir():
        return []
    found = {}
    for entry in runs_root.iterdir():
        if entry.is_dir() and entry.name.startswith("run_"):
            try:
                found[int(entry.name[4:])] = entry
            except ValueError:
                continue
    if require_done is None:
        require_done = any(is_done(path) for path in found.values())
    if require_done:
        found = {run_id: path for run_id, path in found.items() if is_done(path)}
    return sorted(found)


def mark_done(run_path: PathLike, status: str, **info) -> Path:
    """Atomically write the DONE marker (JSON with status, time and extra info)."""
    run_path = Path(run_path)
    ensure_dir(run_path)
    marker = run_path / DONE_MARKER
    tmp = run_path / (DONE_MARKER + ".tmp")
    payload = {"status": status, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, marker)
    return marker


def is_done(run_path: PathLike) -> bool:
    """Whether a run directory carries a DONE marker."""
    return (Path(run_path) / DONE_MARKER).is_file()


def read_done(run_path: PathLike) -> Optional[dict]:
    """Contents of the DONE marker, None if absent or unreadable."""
    marker = Path(run_path) / DONE_MARKER
    if not marker.is_file():
        return None
    try:
        with open(marker) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def ensure_dir(path: PathLike, mode: int = 0o2775) -> Path:
    """Create ``path`` (parents included) and make it group-writable with the setgid bit so
    that the different runner users can share it (best effort: chmod errors are ignored)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, mode)
    except OSError:
        pass
    return path


def check_writable(path: PathLike) -> None:
    """Create and delete a probe file; raise RuntimeError with a hint if that fails."""
    path = Path(path)
    probe = path / f".probe_{os.getpid()}_{int(time.time())}"
    try:
        probe.write_text("probe\n")
        probe.unlink()
    except OSError as e:
        raise RuntimeError(
            f"Exchange directory {path} is not writable ({e}). Check the mount, the group "
            "permissions/setgid bit of the exchange root and (on the board) NFS root_squash."
        ) from e


def set_shared_umask() -> None:
    """Make files created by this process group-writable when an exchange dir is used."""
    if exchange_root() is not None:
        os.umask(0o002)


def chown_to_sudo_user(path: PathLike) -> None:
    """When running under sudo, hand the files created under ``path`` back to the invoking
    user (``SUDO_UID``/``SUDO_GID``) so the runner user can delete them later."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if not uid or not gid:
        return
    uid, gid = int(uid), int(gid)
    for root, dirs, files in os.walk(path):
        for name in [*dirs, *files]:
            try:
                os.chown(os.path.join(root, name), uid, gid)
            except OSError:
                pass
    try:
        os.chown(path, uid, gid)
    except OSError:
        pass


def cleanup_pipeline(
    pipeline_dir: PathLike, patterns: tuple[str, ...] = ("deploy.zip",), dry_run: bool = False
) -> dict:
    """Delete the large per-run files (default: ``deploy.zip``) of one pipeline directory,
    keeping the reports. Returns ``{"files": n, "bytes": total}``."""
    pipeline_dir = Path(pipeline_dir)
    removed, total = 0, 0
    for pattern in patterns:
        for path in pipeline_dir.glob(f"*_artifacts*/runs_output/*/{pattern}"):
            if path.is_file():
                total += path.stat().st_size
                removed += 1
                if not dry_run:
                    try:
                        path.unlink()
                    except OSError:
                        removed -= 1
    return {"files": removed, "bytes": total}


def cleanup_stale(
    root: PathLike, max_age_days: float, dry_run: bool = False, keep: tuple[str, ...] = ()
) -> list[Path]:
    """Delete pipeline directories under ``root`` whose CREATED marker is older than
    ``max_age_days``. Only ``CI_*`` directories with a marker are considered; the current
    pipeline's directory and names in ``keep`` are never touched."""
    root = Path(root)
    if not root.is_dir():
        return []
    protected = set(keep) | {pipeline_dir_name()}
    cutoff = time.time() - max_age_days * 86400
    removed = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("CI_") or entry.name in protected:
            continue
        marker = entry / CREATED_MARKER
        if not marker.is_file() or marker.stat().st_mtime > cutoff:
            continue
        removed.append(entry)
        if not dry_run:
            shutil.rmtree(entry, ignore_errors=True)
    return removed


def describe() -> str:
    """One-line description of the resolved exchange configuration for job logs."""
    root = exchange_root()
    if root is None:
        return f"{EXCHANGE_ENV_VAR} not set: artifacts are exchanged via the working directory"
    return f"{EXCHANGE_ENV_VAR}={root}, pipeline directory {pipeline_exchange_dir()}"
