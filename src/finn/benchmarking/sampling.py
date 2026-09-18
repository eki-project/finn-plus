"""Expansion of benchmark configurations, including random sampling of microbenchmarks.

A benchmark config is a list of entries. A plain entry maps every parameter to a list of
values and is expanded to the Cartesian product (as before). An entry with ``mode: sample``
draws ``num_samples`` parameter sets at random from the DUT's declarative parameter space
(``MicrobenchDUT.param_space()``), rejects sets the DUT's ``validate()`` refuses and,
with ``skip_existing``, sets already present in the microbenchmark result database::

    - mode: sample
      dut: mvau
      num_samples: 200
      seed: 1234              # optional; default $SAMPLE_SEED, else CI_PIPELINE_ID, else 0
      skip_existing: true     # false | true (runs with status ok) | "all" (any status)
      max_attempts: 20000     # default 50 * num_samples
      space:                  # optional overrides of the DUT's default space
        mw: {type: pow2, lo: 16, hi: 1024}
        idt: {type: choice, values: [INT2, INT4, INT8]}
      board: RFSoC2x2         # any other key: scalar = fixed value, list = uniform choice

The expansion is deterministic for a given seed, so every SLURM array task derives the same
run list. When an exchange directory is available (see :mod:`finn.benchmarking.exchange`),
the first task additionally publishes its expansion and later tasks read it, which also
covers the case that the database changes between task starts.

Pure python: DUT classes are passed in, so the module is testable without FINN.
"""

import itertools
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from finn.benchmarking.param_space import (
    Choice,
    Dimension,
    Fixed,
    FloatRange,
    IntRange,
    ParamSpace,
    Pow2Range,
    sample_params,
)
from finn.qor.params_key import IRRELEVANT_PARAMS, existing_param_keys, params_key

try:  # operator-specific loop-bound parameters (needs pandas, optional here)
    from finn.qor.features import SPECS as _SPECS
except ImportError:  # pragma: no cover
    _SPECS = {}

#: Keys of a sample entry that are not run parameters
RESERVED_KEYS = frozenset(
    {"mode", "dut", "num_samples", "seed", "skip_existing", "max_attempts", "space"}
)
EXPANDED_CONFIG_NAME = "bench_config_exp.json"


@dataclass
class SampleStats:
    """Outcome of sampling one config entry."""

    dut: str
    requested: int
    produced: int = 0
    attempts: int = 0
    seed: int = 0
    rejected_invalid: dict[str, int] = field(default_factory=dict)
    rejected_existing: int = 0
    rejected_duplicate: int = 0
    database_path: Optional[str] = None
    database: dict = field(default_factory=dict)
    existing_keys: int = 0
    key_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def dimension_from_config(cfg: Any) -> Dimension:
    """Build a dimension from its config form: a list (uniform choice), a scalar (fixed) or
    a dict ``{type: choice|pow2|int|float, ...}``."""
    if isinstance(cfg, dict) and "type" in cfg:
        kind = cfg["type"]
        if kind == "choice":
            return Choice(list(cfg["values"]), cfg.get("weights"))
        if kind == "pow2":
            return Pow2Range(int(cfg["lo"]), int(cfg["hi"]), cfg.get("weights"))
        if kind == "int":
            return IntRange(
                int(cfg["lo"]), int(cfg["hi"]), bool(cfg.get("log", False)), int(cfg.get("step", 1))
            )
        if kind == "float":
            return FloatRange(
                float(cfg["lo"]), float(cfg["hi"]), bool(cfg.get("log", False)), cfg.get("step")
            )
        if kind == "fixed":
            return Fixed(cfg["value"])
        raise ValueError(f"unknown dimension type {kind!r}")
    if isinstance(cfg, list):
        return Choice(list(cfg))
    return Fixed(cfg)


def build_space(base: ParamSpace, entry: dict) -> tuple[ParamSpace, dict]:
    """Apply an entry's ``space`` overrides and extra keys to a DUT's default space.

    Returns the space (base order, overridden dimensions in place, new keys appended) and
    the fixed values (scalars given directly in the entry).
    """
    space: ParamSpace = dict(base)
    for name, cfg in (entry.get("space") or {}).items():
        space[name] = dimension_from_config(cfg)
    fixed: dict = {}
    for name, value in entry.items():
        if name in RESERVED_KEYS:
            continue
        if isinstance(value, list):
            space[name] = Choice(list(value))
        else:
            fixed[name] = value
            space.pop(name, None)
    return space, fixed


def default_seed(entry_index: int = 0) -> int:
    """``$SAMPLE_SEED``, else ``$CI_PIPELINE_ID``, else 0; plus the entry index so that
    several sample entries of one config draw different sequences."""
    for var in ("SAMPLE_SEED", "CI_PIPELINE_ID"):
        value = os.environ.get(var, "").strip()
        if value:
            try:
                return int(value) + entry_index
            except ValueError:
                continue
    return entry_index


def sample_entry(
    entry: dict,
    space: ParamSpace,
    validate: Callable[[dict], Optional[str]],
    existing: Optional[set[tuple]] = None,
    rng: Optional[random.Random] = None,
    fixed: Optional[dict] = None,
    irrelevant: tuple[str, ...] = IRRELEVANT_PARAMS,
) -> tuple[list[dict], SampleStats]:
    """Draw ``entry["num_samples"]`` valid, mutually distinct parameter sets that are not in
    ``existing`` (keys as :func:`params_key` over the sampled + fixed names)."""
    num_samples = int(entry.get("num_samples", 0))
    max_attempts = int(entry.get("max_attempts") or 50 * max(num_samples, 1))
    rng = rng or random.Random(int(entry.get("seed", 0)))
    fixed = dict(fixed or {})
    key_names = sorted(set(space) | set(fixed))
    stats = SampleStats(
        dut=str(entry.get("dut")),
        requested=num_samples,
        seed=int(entry.get("seed", 0)),
        existing_keys=len(existing or ()),
        key_names=[k for k in key_names if k not in irrelevant],
    )
    existing = existing or set()
    seen: set[tuple] = set()
    samples: list[dict] = []
    while len(samples) < num_samples and stats.attempts < max_attempts:
        stats.attempts += 1
        params = sample_params(space, rng, fixed)
        reason = validate(params)
        if reason is not None:
            stats.rejected_invalid[reason] = stats.rejected_invalid.get(reason, 0) + 1
            continue
        key = params_key(params, key_names, irrelevant)
        if key in seen:
            stats.rejected_duplicate += 1
            continue
        if key in existing:
            stats.rejected_existing += 1
            seen.add(key)
            continue
        seen.add(key)
        samples.append(params)
    stats.produced = len(samples)
    return samples, stats


def _expand_cartesian(entry: dict) -> list[dict]:
    return [dict(zip(entry.keys(), values)) for values in itertools.product(*entry.values())]


def expand_config(
    config: list[dict],
    dut_registry: dict[str, type],
    database_path: Optional[str] = None,
) -> tuple[list[dict], list[SampleStats]]:
    """Expand all entries of a benchmark config into the flat list of run parameter sets.

    Plain entries are expanded to their Cartesian product; ``mode: sample`` entries are
    sampled from ``dut_registry[dut].param_space()`` (see module docstring). ``database_path``
    (the microbenchmark database) enables ``skip_existing``; if it is None the check is
    skipped with a warning.
    """
    expanded: list[dict] = []
    stats: list[SampleStats] = []
    for index, entry in enumerate(config):
        if entry.get("mode") != "sample":
            expanded.extend(_expand_cartesian({k: v for k, v in entry.items() if k != "mode"}))
            continue
        dut_name = entry["dut"]
        if dut_name not in dut_registry:
            raise ValueError(f"unknown DUT {dut_name!r} in sample entry")
        dut_cls = dut_registry[dut_name]
        if "seed" not in entry or entry["seed"] is None:
            entry = {**entry, "seed": default_seed(index)}
        count_override = os.environ.get("SAMPLE_COUNT", "").strip()
        if count_override:
            entry = {**entry, "num_samples": int(count_override)}
        space, fixed = build_space(dut_cls.param_space(), entry)
        # the DUT's defaults are fixed values (lists included), unless the entry sets them
        for name, value in dut_cls.sample_defaults().items():
            if name not in fixed and name not in space:
                fixed[name] = value
        fixed["dut"] = dut_name
        key_names = sorted(set(space) | set(fixed))
        spec = _SPECS.get(dut_name)
        irrelevant = tuple(IRRELEVANT_PARAMS) + tuple(
            c[len("params.") :] for c in (spec.irrelevant_param_cols if spec else ())
        )
        existing: set[tuple] = set()
        db_stats: dict = {}
        skip_existing = entry.get("skip_existing", True)
        if skip_existing:
            if database_path and os.path.isdir(database_path):
                statuses = None if skip_existing == "all" else ("ok",)
                existing, db_stats = existing_param_keys(
                    database_path, dut_name, key_names, statuses, irrelevant
                )
            else:
                print(
                    "WARNING: skip_existing requested but no microbenchmark database available "
                    "(FINN_MICROBENCHMARK_DATABASE), sampling without novelty check"
                )
        rng = random.Random(int(entry["seed"]))
        samples, entry_stats = sample_entry(
            entry, space, dut_cls.validate, existing, rng, fixed, irrelevant
        )
        entry_stats.database_path = database_path
        entry_stats.database = db_stats
        if entry_stats.produced < entry_stats.requested:
            print(
                "WARNING: sample entry %d (%s) produced only %d of %d requested configurations "
                "after %d attempts (invalid: %s, existing: %d, duplicate: %d)"
                % (
                    index,
                    dut_name,
                    entry_stats.produced,
                    entry_stats.requested,
                    entry_stats.attempts,
                    entry_stats.rejected_invalid,
                    entry_stats.rejected_existing,
                    entry_stats.rejected_duplicate,
                )
            )
        else:
            print(
                "Sampled %d configurations for %s (seed %d, %d attempts, %d existing skipped)"
                % (
                    entry_stats.produced,
                    dut_name,
                    entry_stats.seed,
                    entry_stats.attempts,
                    entry_stats.rejected_existing,
                )
            )
        for params in samples:
            params["_sampling"] = {"entry": index, "seed": int(entry["seed"])}
        expanded.extend(samples)
        stats.append(entry_stats)
    return expanded, stats


def pop_sampling_info(params: dict) -> Optional[dict]:
    """Remove and return the sampling bookkeeping added by :func:`expand_config`."""
    return params.pop("_sampling", None)


def publish_expanded(
    directory: Path, config_expanded: list[dict], wait_s: float = 120.0
) -> list[dict]:
    """First-writer-wins publication of the expanded config in a shared directory.

    The first caller (holding the ``.lock`` created with O_EXCL) writes the file and
    returns its own list; later callers wait for the file and return its content, so all
    SLURM array tasks of a job work on the same expansion.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / EXPANDED_CONFIG_NAME
    lock = directory / (EXPANDED_CONFIG_NAME + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        fd = None
    if fd is not None:
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)
        tmp = directory / (EXPANDED_CONFIG_NAME + ".tmp")
        with open(tmp, "w") as f:
            json.dump(config_expanded, f, indent=2)
        os.replace(tmp, target)
        return config_expanded
    deadline = time.time() + wait_s
    while not target.is_file():
        if time.time() > deadline:
            print("WARNING: published expanded config did not appear in time, using own expansion")
            return config_expanded
        time.sleep(1.0)
    with open(target) as f:
        published = json.load(f)
    if published != config_expanded:
        print(
            "NOTE: using the expanded config published by another task (%d runs, own: %d)"
            % (len(published), len(config_expanded))
        )
    return published
