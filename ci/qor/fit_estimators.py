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
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from finn.qor.database import DATABASE_ENV_VAR, load_microbenchmark_database
from finn.qor.estimator import (
    DEFAULT_TARGETS,
    MODEL_DIR_ENV_VAR,
    RESOURCE_TARGET_PREFIX,
    RESOURCE_TARGETS,
    QoREstimator,
    available_regressor_grid,
    has_enough_signal,
    learning_curve,
    model_basename,
    regressor_name,
    select_regressor,
    target_signal,
)
from finn.qor.evaluation import (
    dataframe_to_markdown,
    estimation_error_table,
    plot_learning_curve,
    plot_pareto_front,
    plot_regressor_comparison,
    symbolic_equation_markdown,
    write_symbolic_equation_tex,
)
from finn.qor.features import SPECS

logger = logging.getLogger("fit_estimators")

# Columns of the analytical estimates in the database, compared against the new model
# (the HLS DSP column is coalesced from DSP48E/DSP58E by the database loader)
REFERENCE_ESTIMATES = {
    RESOURCE_TARGETS["LUT"]: {
        "FINN": "metrics.estimate.resources.LUT",
        "HLS": "metrics.hls_estimate.resources.LUT",
    },
    RESOURCE_TARGETS["DSP"]: {
        "FINN": "metrics.estimate.resources.DSP",
        "HLS": "metrics.hls_estimate.resources.DSP",
    },
    RESOURCE_TARGETS["BRAM_18K"]: {
        "FINN": "metrics.estimate.resources.BRAM_18K",
        "HLS": "metrics.hls_estimate.resources.BRAM_18K",
    },
    RESOURCE_TARGETS["URAM"]: {
        "FINN": "metrics.estimate.resources.URAM",
        "HLS": "metrics.hls_estimate.resources.URAM",
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
    parser.add_argument(
        "--operators",
        default="all",
        help="comma-separated operator names (default: all with a feature spec)",
    )
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help="comma-separated target columns (default: %(default)s)",
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
        "--regressors",
        default=None,
        help="comma-separated regressor names to evaluate (default: all available)",
    )
    parser.add_argument(
        "--skip-regressors", default=None, help="comma-separated regressor names to leave out"
    )
    parser.add_argument(
        "--list-regressors",
        action="store_true",
        help="print the available regressors and their optional dependencies, then exit",
    )
    parser.add_argument(
        "--subset",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="only use database rows with COLUMN == VALUE (repeatable, e.g. "
        "params.backend=hls for a per-backend symbolic formula); implies --no-store",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="REGRESSOR.param=JSON",
        help="override one grid list, e.g. SymbolicRegressor.regressor__niterations=[100]",
    )
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


def _parse_value(text: str):
    """Parse a CLI value as JSON, falling back to the raw string."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def apply_grid_overrides(grid, overrides: list[str]):
    """Apply ``REGRESSOR.param=JSON`` overrides to the grid entries (in place, returned)."""
    for item in overrides:
        key, _, value = item.partition("=")
        name, _, param = key.partition(".")
        if not param or not value:
            raise ValueError(f"--set expects REGRESSOR.param=JSON, got {item!r}")
        parsed = _parse_value(value)
        if not isinstance(parsed, list):
            parsed = [parsed]
        matched = False
        for reg_cls, param_grid in grid:
            if regressor_name(reg_cls) == name:
                param_grid[param] = parsed
                matched = True
        if not matched:
            raise ValueError(f"--set: no regressor named {name!r} in the grid")
    return grid


def _float_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if value != value else value  # NaN -> None (JSON)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.list_regressors:
        from finn.qor.models import dependency_status

        for reg_cls, param_grid in available_regressor_grid(quick=False):
            n_combinations = 1
            for values in param_grid.values():
                n_combinations *= len(values)
            print(f"{regressor_name(reg_cls)}: {n_combinations} parameter combinations")
        for name, reason in dependency_status().items():
            if reason:
                print(f"{name}: not available ({reason})")
        return 0
    if not args.database:
        logger.error("No database given (--database or $%s)", DATABASE_ENV_VAR)
        return 2
    if not args.model_dir and not args.no_store:
        logger.error("No model directory given (--model-dir or $%s)", MODEL_DIR_ENV_VAR)
        return 2
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    include = args.regressors.split(",") if args.regressors else None
    exclude = args.skip_regressors.split(",") if args.skip_regressors else None
    grid = apply_grid_overrides(available_regressor_grid(args.quick, include, exclude), args.set)
    logger.info("Regressors: %s", [regressor_name(cls) for cls, _ in grid])
    column_filters = {}
    for item in args.subset:
        key, _, value = item.partition("=")
        column_filters[key] = _parse_value(value)
    if column_filters:
        logger.info("Fitting on the subset %s (models are not stored)", column_filters)
        args.no_store = True
    subset_suffix = "".join(
        "__" + "".join(c if c.isalnum() else "_" for c in f"{k}_{v}")
        for k, v in column_filters.items()
    )
    cv_kwargs = dict(cv=args.cv, n_repeats=args.n_repeats, n_jobs=args.n_jobs, grid=grid)
    summary: dict[str, dict] = {}
    failures = 0

    operators = list(SPECS) if args.operators == "all" else args.operators.split(",")
    for operator in operators:
        try:
            df, stats = load_microbenchmark_database(
                operator,
                args.database,
                exclude_commit=args.exclude_commit,
                exclude_pipeline_id=args.exclude_pipeline_id,
                **column_filters,
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning("%s: no database entries (%s), skipping", operator, e)
            summary[operator] = {"status": "skipped", "reason": str(e)}
            continue
        for target in args.targets.split(","):
            name = f"{operator}/{target}"
            if target not in df.columns:
                logger.warning("%s: target column not in database, skipping", name)
                summary[name] = {"status": "skipped", "reason": "target column missing"}
                continue
            signal = target_signal(df[target].values)
            if target.startswith(RESOURCE_TARGET_PREFIX) and not has_enough_signal(
                df[target].values
            ):
                # e.g. URAM is zero for almost every configuration: keep the analytical
                # estimate (and a previously stored model) instead of fitting on noise
                logger.warning("%s: not enough signal to fit a model (%s), skipping", name, signal)
                summary[name] = {"status": "skipped", "reason": "not enough signal", **signal}
                continue
            try:
                estimator = QoREstimator(operator, target)
                result = select_regressor(estimator, df, **cv_kwargs)
            except Exception as e:  # noqa: BLE001 - report and continue with the next target
                logger.exception("%s: model selection failed", name)
                summary[name] = {"status": "failed", "reason": str(e)}
                failures += 1
                continue

            base = out_dir / (model_basename(operator, target) + subset_suffix)
            result.scores.to_csv(f"{base}_selection.csv")
            plot_regressor_comparison(result, f"{base}_regressors.png")
            if result.cv_results:
                pd.concat(result.cv_results.values(), ignore_index=True).to_csv(
                    f"{base}_cv_results.csv", index=False
                )
            oof = pd.DataFrame({"y_true": estimator.rows_with_target(df)[target].values})
            for reg, errors in result.sample_errors.items():
                oof[reg] = errors["y_pred"].values
            oof.to_csv(f"{base}_oof_predictions.csv", index=False)
            equation = estimator.metadata.get("equation")
            if equation is not None:
                Path(f"{base}_equation.md").write_text(symbolic_equation_markdown(equation))
                write_symbolic_equation_tex(equation, f"{base}_equation.tex")
                plot_pareto_front(equation, f"{base}_pareto.png")
            timing_cols = [
                c
                for c in ("fit_time_s", "search_time_s", "single_predict_ms")
                if c in result.scores
            ]
            entry = {
                "status": "ok",
                "database": str(stats),
                "n_samples": estimator.metadata["n_samples"],
                "scoring": result.scoring,
                "best_regressor": result.best_name,
                "cv_score": result.best_score,
                "fold_mape": float(result.scores.loc[result.best_name, "fold_mape"]),
                "fold_mae": float(result.scores.loc[result.best_name, "fold_mae"]),
                "zero_hit_rate": float(result.scores.loc[result.best_name, "zero_hit_rate"]),
                "target_signal": signal,
                "regressors": {
                    reg: {
                        "mean_score": _float_or_none(row.get("mean_score")),
                        "fold_mape": _float_or_none(row.get("fold_mape")),
                        "fold_mae": _float_or_none(row.get("fold_mae")),
                        **{c: _float_or_none(row.get(c)) for c in timing_cols},
                    }
                    for reg, row in result.scores.iterrows()
                },
            }
            if equation is not None:
                entry["equation"] = {k: equation[k] for k in ("sympy", "complexity", "loss")}

            # Compare out-of-fold predictions of the new model with the analytical estimates
            if target in REFERENCE_ESTIMATES:
                oof = estimator.rows_with_target(df).assign(
                    _oof=result.sample_errors[result.best_name]["y_pred"].values
                )
                table = estimation_error_table(
                    oof, target, {**REFERENCE_ESTIMATES[target], "Empirical (out-of-fold)": "_oof"}
                )
                table.to_csv(f"{base}_comparison.csv")
                (out_dir / f"{model_basename(operator, target)}_comparison.md").write_text(
                    f"# {name}: estimator comparison\n\n{dataframe_to_markdown(table)}"
                )
                entry["comparison_mape"] = {k: float(v) for k, v in table[("All", "MAPE")].items()}
                entry["comparison_mae"] = {k: float(v) for k, v in table[("All", "MAE")].items()}

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
