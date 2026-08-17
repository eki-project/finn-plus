#!/usr/bin/env python3
"""Launch a per-rank binary/port assignment based on hostname and local MPI rank.

Expected mapping JSON format:

{
  "nodeA": [[5000, "/path/bin0"], [5001, "/path/bin1"]],
  "nodeB": [[6000, "/path/bin0"], [6001, "/path/bin1"]]
}

Selection rules:
1. Determine local host name.
2. Resolve it to a mapping key (exact match, then short-name match).
3. Determine local MPI rank on node.
4. Select mapping[host_key][local_rank] -> [port, binary].
5. execv(binary, [binary, "--socket-host", "0.0.0.0", "--socket-port", "<port>"]).
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Any


def get_local_rank() -> int:
    """Return the local rank (rank index within the current node).

    Reads one of:
    - OMPI_COMM_WORLD_LOCAL_RANK (Open MPI)
    - MPI_LOCALRANKID (MPICH/Intel MPI Hydra)
    - SLURM_LOCALID (Slurm)

    Raises:
        RuntimeError: If no supported local-rank environment variable is set.
    """
    for key in ("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID"):
        value = os.environ.get(key)
        if value is not None:
            return int(value)

    raise RuntimeError(
        "Local rank env var not found "
        "(OMPI_COMM_WORLD_LOCAL_RANK / MPI_LOCALRANKID / SLURM_LOCALID)."
    )


def short_hostname(name: str) -> str:
    """Return hostname without domain suffix."""
    return name.split(".", 1)[0]


def resolve_host_key(mapping: dict[str, Any]) -> str:
    """Resolve current machine hostname to a key in the mapping.

    Matching strategy:
    1) Exact match against candidates:
       socket.gethostname(), socket.getfqdn(), HOSTNAME env,
       and their short-host variants.
    2) Short-name match against mapping keys.

    socket.gethostname()/getfqdn() are direct syscalls and are tried first, since under
    mpirun, environment variables from the launching process are forwarded to remote
    ranks by default: HOSTNAME env can be stale on a rank running on a different host
    than the launcher, matching the wrong key in the mapping. It is kept as a last-resort
    fallback for environments where the syscalls don't return a usable name.

    Args:
        mapping: Loaded mapping JSON object.

    Returns:
        Mapping key corresponding to the current host.

    Raises:
        RuntimeError: If no host key can be resolved.
    """
    candidates: list[str] = []
    for host in (socket.gethostname(), socket.getfqdn(), os.environ.get("HOSTNAME")):
        if host and host not in candidates:
            candidates.append(host)
        if host:
            short = short_hostname(host)
            if short not in candidates:
                candidates.append(short)

    for candidate in candidates:
        if candidate in mapping:
            return candidate

    short_to_full: dict[str, str] = {short_hostname(key): key for key in mapping}
    for candidate in candidates:
        short = short_hostname(candidate)
        if short in short_to_full:
            return short_to_full[short]

    raise RuntimeError(
        "Could not match hostname to mapping keys. "
        f"Candidates={candidates}, keys={list(mapping.keys())}"
    )


def parse_entry(entry: Any, host_key: str, local_rank: int) -> tuple[int, Path]:
    """Parse and validate a single mapping entry of form [port, binary].

    Args:
        entry: The raw mapping entry.
        host_key: Resolved hostname key (for error messages).
        local_rank: Local rank index (for error messages).

    Returns:
        Tuple of (port, binary_path).

    Raises:
        RuntimeError: If entry is malformed.
    """
    if not (isinstance(entry, list) and len(entry) == 2):
        raise RuntimeError(
            f"Invalid entry at host {host_key!r} index {local_rank}: {entry!r}. "
            "Expected [port, bin]"
        )

    raw_port, raw_binary = entry
    port = int(raw_port)
    binary = Path(str(raw_binary)).expanduser().resolve()

    return port, binary


def load_mapping(mapping_path: Path) -> dict[str, Any]:
    """Load hostname -> [[port, bin], ...] mapping from JSON file."""
    with mapping_path.open("r", encoding="utf-8") as fp:
        mapping: dict[str, Any] = json.load(fp)

    if not isinstance(mapping, dict):
        raise RuntimeError("Mapping must be an object: hostname -> [[port, bin], ...]")

    return mapping


def main(argv: list[str]) -> int:
    """Program entrypoint.

    Args:
        argv: Command-line arguments (typically sys.argv).

    Returns:
        Process exit code (only returned on argument/validation failure paths).
        On success, this function does not return because os.execv replaces process.
    """
    if len(argv) != 2:
        print(f"Usage: {Path(argv[0]).name} <mapping.json>", file=sys.stderr)
        return 2

    mapping_path = Path(argv[1]).expanduser().resolve()
    mapping = load_mapping(mapping_path)

    local_rank = get_local_rank()
    host_key = resolve_host_key(mapping)

    entries = mapping[host_key]
    if not isinstance(entries, list):
        raise RuntimeError(f"Mapping for host {host_key!r} must be a list")

    if local_rank < 0 or local_rank >= len(entries):
        raise RuntimeError(
            f"Local rank {local_rank} out of range for host {host_key!r}; "
            f"have {len(entries)} entries"
        )

    port, binary = parse_entry(entries[local_rank], host_key, local_rank)

    if not binary.exists():
        raise RuntimeError(f"Binary does not exist: {binary}")
    if not os.access(binary, os.X_OK):
        raise RuntimeError(f"Binary is not executable: {binary}")

    os.chdir(binary.parent)
    os.execv(
        str(binary),
        [str(binary), "--socket-host", "0.0.0.0", "--socket-port", str(port)],
    )
    return 0  # Unreachable after successful execv.


if __name__ == "__main__":
    main(sys.argv)
