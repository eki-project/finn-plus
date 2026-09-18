#!/usr/bin/env python3
"""Retention for the benchmark exchange directory (see finn.benchmarking.exchange).

    python ci/exchange_cleanup.py --pipeline        # delete this pipeline's deploy.zip files
    python ci/exchange_cleanup.py --stale-days 14   # delete other pipelines' dirs > 14 days old
    python ci/exchange_cleanup.py --pipeline --stale-days 14 --dry-run

Reports are kept; only the large bitstream packages of the current pipeline are removed
(unless KEEP_EXCHANGE_DEPLOY is set), and whole pipeline directories only after they aged
past the retention period (judged by their CREATED marker).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from finn.benchmarking import exchange  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pipeline", action="store_true", help="remove this pipeline's deploy.zip")
    parser.add_argument("--stale-days", type=float, default=None, help="remove older pipeline dirs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = exchange.exchange_root()
    print(exchange.describe())
    if root is None:
        print("Nothing to clean up.")
        return 0

    if args.pipeline:
        if os.environ.get("KEEP_EXCHANGE_DEPLOY", "").strip():
            print("KEEP_EXCHANGE_DEPLOY set, keeping deployment packages of this pipeline")
        else:
            pipeline_dir = exchange.pipeline_exchange_dir()
            if pipeline_dir is not None and pipeline_dir.is_dir():
                result = exchange.cleanup_pipeline(pipeline_dir, dry_run=args.dry_run)
                print(
                    "%s %d deployment packages (%.1f MB) in %s"
                    % (
                        "Would remove" if args.dry_run else "Removed",
                        result["files"],
                        result["bytes"] / 1e6,
                        pipeline_dir,
                    )
                )
    if args.stale_days is not None:
        removed = exchange.cleanup_stale(root, args.stale_days, dry_run=args.dry_run)
        print(
            "%s %d pipeline directories older than %g days: %s"
            % (
                "Would remove" if args.dry_run else "Removed",
                len(removed),
                args.stale_days,
                ", ".join(p.name for p in removed) or "-",
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
