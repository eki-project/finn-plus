#!/usr/bin/env python3
"""Find a configurable number of unused TCP ports within a given range.
Always prints JSON.

Example:
  python find_unused_ports.py --start 8000 --end 9000 --count 5
"""

import argparse
import json
import socket
import sys


def is_port_free(port: int, host: str = "0.0.0.0") -> bool:
    """Return True if TCP port can be bound on host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_unused_ports(start: int, end: int, count: int, host: str = "0.0.0.0") -> list[int]:
    """Return a list of unused TCP ports within the given range."""
    if start < 1 or end > 65535 or start > end:
        raise ValueError("Invalid port range. Must satisfy: 1 <= start <= end <= 65535")
    if count < 1:
        raise ValueError("count must be at least 1")

    ports = []
    for port in range(start, end + 1):
        if is_port_free(port, host):
            ports.append(port)
            if len(ports) == count:
                break
    return ports


def emit(payload: dict, exit_code: int = 0) -> None:
    """Emit JSON payload to stdout and exit with the given code."""
    print(json.dumps(payload, separators=(",", ":")))
    sys.exit(exit_code)


def main() -> None:
    """Main entry point for the script."""  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Return unused ports in a range (JSON output only)."
    )
    parser.add_argument("--start", type=int, required=True, help="Start of port range (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End of port range (inclusive)")
    parser.add_argument("--count", type=int, required=True, help="Number of unused ports requested")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host/interface to test bind (default matches LayerSimulationBackend's "
        "--socket-host 0.0.0.0, so port freeness here reflects what will actually be bound)",
    )
    args = parser.parse_args()

    try:
        ports = find_unused_ports(args.start, args.end, args.count, args.host)
    except Exception as e:
        emit(
            {
                "ok": False,
                "error": str(e),
                "ports": [],
                "requested_count": args.count,
            },
            exit_code=2,
        )
        return

    emit(
        {
            "ok": True,
            "hostname": socket.gethostname(),
            "requested_count": args.count,
            "ports": ports,
        },
        exit_code=0 if len(ports) == args.count else 1,
    )


if __name__ == "__main__":
    main()
