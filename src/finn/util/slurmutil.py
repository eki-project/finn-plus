"""Utility functions for SLURM/MPI host discovery and allocation-aware worker
sizing used by distributed simulation.
"""

import os
import re
import shutil
import subprocess

from finn.util.basic import launch_process_helper
from finn.util.logging import log


def detect_slurm_hosts() -> str | None:
    """Return a comma-separated hostname list for the current SLURM allocation, or None
    if not running under SLURM (or the node list could not be expanded).

    Uses the standard SLURM_JOB_NODELIST/SLURM_NODELIST environment variables (always
    set by SLURM for a job, no FINN-specific configuration needed) and expands SLURM's
    hostlist range syntax (e.g. "cn[0101,0103,0106]") via `scontrol show hostnames`.
    """
    nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not nodelist:
        return None

    scontrol_path = shutil.which("scontrol")
    if scontrol_path is None:
        log.warning(
            "SLURM allocation detected (SLURM_JOB_NODELIST/SLURM_NODELIST is set) but "
            "'scontrol' was not found in PATH, so the node list could not be expanded. "
            "Falling back to localhost for the distributed simulation. Set "
            "fifosim_mpi_hosts explicitly to override."
        )
        return None

    try:
        cmd_out, _cmd_err = launch_process_helper(
            [scontrol_path, "show", "hostnames", nodelist],
            print_stdout=False,
            print_stderr=False,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        log.warning(
            f"Failed to expand SLURM node list {nodelist!r} via scontrol: {e}. "
            "Falling back to localhost for the distributed simulation."
        )
        return None

    hostnames = [h.strip() for h in cmd_out.splitlines() if h.strip()]
    if not hostnames:
        return None
    return ",".join(hostnames)


def get_local_cores() -> int:
    """Return the number of CPU cores available to this process."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    cpu_count = os.cpu_count()
    return max(1, int(cpu_count) if cpu_count is not None else 1)


def parse_hosts(raw: str | None) -> list[str]:
    """Parse a "host[:slots],host[:slots],..." string into a plain host list."""
    if raw is None or raw.strip() == "":
        return ["localhost"]
    hosts: list[str] = []
    for entry in raw.split(","):
        host = entry.strip()
        if host == "":
            continue
        # Accept both "host" and "host:slots" input forms.
        if ":" in host:
            host = host.split(":", 1)[0]
        hosts.append(host)
    return hosts if hosts else ["localhost"]


def parse_hosts_with_slots(raw: str | None) -> list[tuple[str, int]]:
    """Parse a "host[:slots],host[:slots],..." string into (host, slots) pairs.

    Hosts without an explicit slot count default to the local core count.
    """
    if raw is None or raw.strip() == "":
        return [("localhost", get_local_cores())]

    hosts_with_slots: list[tuple[str, int]] = []
    default_slots = get_local_cores()
    for entry in raw.split(","):
        token = entry.strip()
        if token == "":
            continue
        host = token
        slots = default_slots
        if ":" in token:
            host_part, slot_part = token.split(":", 1)
            host = host_part.strip()
            try:
                parsed_slots = int(slot_part.strip())
                if parsed_slots > 0:
                    slots = parsed_slots
            except ValueError:
                pass
        if host != "":
            hosts_with_slots.append((host, slots))

    if len(hosts_with_slots) == 0:
        return [("localhost", default_slots)]
    return hosts_with_slots


def try_int(value: str | None) -> int | None:
    """Cast value to int, return None if it fails or if value is None."""
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def parse_slurm_job_cpus_per_node(value: str | None) -> int | None:
    """Return parse slurm job cpus per node."""
    if value is None:
        return None
    # Example values: "16", "16(x2)", "16(x2),8"
    first_chunk = value.split(",")[0].strip()
    match = re.match(r"^(\d+)", first_chunk)
    if match is None:
        return None
    parsed = int(match.group(1))
    return parsed if parsed > 0 else None


def get_slurm_cpus() -> int | None:
    """Return slurm workers while considering cpu allocation."""
    cpus_per_task = try_int(os.environ.get("SLURM_CPUS_PER_TASK"))
    if cpus_per_task is not None:
        return cpus_per_task

    cpus_on_node = try_int(os.environ.get("SLURM_CPUS_ON_NODE"))
    if cpus_on_node is not None:
        return cpus_on_node

    return parse_slurm_job_cpus_per_node(os.environ.get("SLURM_JOB_CPUS_PER_NODE"))


def get_slurm_mem_workers(cpus_alloc: int | None) -> int | None:
    """Return number of slurm workers while considering memory allocation."""
    # SLURM memory env vars are in MB.
    mem_per_node_mb = try_int(os.environ.get("SLURM_MEM_PER_NODE"))
    if mem_per_node_mb is not None:
        return max(1, mem_per_node_mb // (10 * 1024))  # 10GB per synthesis

    mem_per_cpu_mb = try_int(os.environ.get("SLURM_MEM_PER_CPU"))
    if mem_per_cpu_mb is not None and cpus_alloc is not None:
        return max(1, (mem_per_cpu_mb * cpus_alloc) // (10 * 1024))

    return None
