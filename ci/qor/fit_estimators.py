#!/usr/bin/env python3
"""Fit, evaluate and store empirical QoR estimators on the CI microbenchmark database.

Run by the GitLab CI after microbenchmark results were added to the database (see
ci/.gitlab-bench.yml), but also usable locally::

    FINN_MICROBENCHMARK_DATABASE=/path/to/db FINN_QOR_MODEL_DIR=/path/to/models \\
        python ci/qor/fit_estimators.py --output-dir qor_fit_artifacts

For every operator and target, all regressors of the grid are tuned by cross-validation, the
best one is compared with the currently stored model (if any) and stored if it is at least as
good. Selection results, plots and a summary are written to the output directory.

Only needs pandas, numpy, scikit-learn and matplotlib (no FINN installation), the repository's
src/ directory is added to the path automatically.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from finn.qor.database import (
    DATABASE_ENV_VAR,
    DEFAULT_POWER_COL,
    POWER_TARGET_COL,
    load_microbenchmark_database,
)
from finn.qor.estimator import (
    MODEL_DIR_ENV_VAR,
    REGRESSOR_GRID,
    REGRESSOR_GRID_QUICK,
    QoREstimator,
    learning_curve,
    model_basename,
    select_regressor,
)
from finn.qor.evaluation import (
    dataframe_to_markdown,
    estimation_error_table,
    plot_learning_curve,
    plot_regressor_comparison,
)

logger = logging.getLogger("fit_estimators")

DEFAULT_TARGETS = ["metrics.synth.resources.LUT", POWER_TARGET_COL]

# Columns of the analytical estimates in the database, compared against the new model
REFERENCE_ESTIMATES = {
    "metrics.synth.resources.LUT": {
        "FINN": "metrics.estimate.resources.LUT",
        "HLS": "metrics.hls_estimate.resources.LUT",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--database", default=os.environ.get(DATABASE_ENV_VAR), help=f"default: ${DATABASE_ENV_VAR}"
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get(MODEL_DIR_ENV_VAR),
        help=f"default: ${MODEL_DIR_ENV_VAR}",
    )
    parser.add_argument("--output-dir", default="qor_fit_artifacts")
    parser.add_argument("--operators", default="mvau", help="comma-separated operator names")
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help="comma-separated target columns (default: %(default)s)",
    )
    parser.add_argument(
        "--power-col",
        default=DEFAULT_POWER_COL,
        help="measured power column the 'power' target is derived from (default: %(default)s)",
    )
    parser.add_argument("--exclude-commit", nargs="*", default=None, help="commit hash prefixes")
    parser.add_argument("--exclude-pipeline-id", nargs="*", type=int, default=None)
    parser.add_argument("--cv", type=int, default=5, help="number of CV folds")
    parser.add_argument("--n-repeats", type=int, default=1, help="CV repetitions")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--learning-curve",
        action="store_true",
        help="also evaluate performance vs. training data size (slow)",
    )
    parser.add_argument("--quick", action="store_true", help="use a tiny regressor grid")
    parser.add_argument(
        "--force",
        action="store_true",
        help="store the new model even if the stored one scores better",
    )
    parser.add_argument(
        "--no-store", action="store_true", help="evaluate only, do not store models"
    )
    return parser.parse_args()


def stored_score(model_dir: str, operator: str, target: str, scoring: str) -> float | None:
    """CV score of the currently stored model, if it exists and used the same scoring."""
    sidecar = Path(model_dir) / (model_basename(operator, target) + ".json")
    if not sidecar.is_file():
        return None
    meta = json.loads(sidecar.read_text())
    if meta.get("cv_scoring") != scoring:
        return None
    return meta.get("cv_score")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.database:
        logger.error("No database given (--database or $%s)", DATABASE_ENV_VAR)
        return 2
    if not args.model_dir and not args.no_store:
        logger.error("No model directory given (--model-dir or $%s)", MODEL_DIR_ENV_VAR)
        return 2
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = REGRESSOR_GRID_QUICK if args.quick else REGRESSOR_GRID
    cv_kwargs = dict(cv=args.cv, n_repeats=args.n_repeats, n_jobs=args.n_jobs, grid=grid)
    summary: dict[str, dict] = {}
    failures = 0

    for operator in args.operators.split(","):
        df, stats = load_microbenchmark_database(
            operator,
            args.database,
            exclude_commit=args.exclude_commit,
            exclude_pipeline_id=args.exclude_pipeline_id,
            power_col=args.power_col,
        )
        for target in args.targets.split(","):
            name = f"{operator}/{target}"
            if target not in df.columns:
                logger.warning("%s: target column not in database, skipping", name)
                summary[name] = {"status": "skipped", "reason": "target column missing"}
                continue
            try:
                estimator = QoREstimator(operator, target)
                result = select_regressor(estimator, df, **cv_kwargs)
            except Exception as e:  # noqa: BLE001 - report and continue with the next target
                logger.exception("%s: model selection failed", name)
                summary[name] = {"status": "failed", "reason": str(e)}
                failures += 1
                continue

            base = out_dir / model_basename(operator, target)
            result.scores.to_csv(f"{base}_selection.csv")
            plot_regressor_comparison(result, f"{base}_regressors.png")
            entry = {
                "status": "ok",
                "database": str(stats),
                "n_samples": int(len(df)),
                "scoring": result.scoring,
                "best_regressor": result.best_name,
                "cv_score": result.best_score,
                "fold_mape": float(result.scores.loc[result.best_name, "fold_mape"]),
            }

            # Compare out-of-fold predictions of the new model with the analytical estimates
            if target in REFERENCE_ESTIMATES:
                oof = df.assign(_oof=result.sample_errors[result.best_name]["y_pred"].values)
                table = estimation_error_table(
                    oof, target, {**REFERENCE_ESTIMATES[target], "Empirical (out-of-fold)": "_oof"}
                )
                table.to_csv(f"{base}_comparison.csv")
                (out_dir / f"{model_basename(operator, target)}_comparison.md").write_text(
                    f"# {name}: estimator comparison\n\n{dataframe_to_markdown(table)}"
                )
                entry["comparison_mape"] = {k: float(v) for k, v in table[("All", "MAPE")].items()}

            if args.learning_curve:
                curve = learning_curve(estimator, df, [0.1, 0.2, 0.4, 0.6, 0.8, 1.0], **cv_kwargs)
                curve.to_csv(f"{base}_learning_curve.csv")
                plot_learning_curve(curve, f"{base}_learning_curve.png")

            if not args.no_store:
                previous = stored_score(args.model_dir, operator, target, result.scoring)
                if previous is not None and result.best_score < previous and not args.force:
                    logger.warning(
                        "%s: new model (%s=%.4f) is worse than stored model (%.4f), keeping stored "
                        "model (use --force to overwrite)",
                        name,
                        result.scoring,
                        result.best_score,
                        previous,
                    )
                    entry["stored"] = False
                    entry["previous_cv_score"] = previous
                else:
                    estimator.save(args.model_dir)
                    entry["stored"] = True
                    entry["previous_cv_score"] = previous
            summary[name] = entry

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Summary:\n%s", json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
