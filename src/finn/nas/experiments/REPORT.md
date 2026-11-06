# NAS experiment report: figures and tables

`finn nas report` turns the outputs of the experiment drivers (`finn nas sample`, `finn nas
search`, `finn nas groundtruth`) into the figures and markdown tables of the write-up. It only
reads the experiment directories, needs nothing but matplotlib, and can be run while an
experiment is still going: a running experiment is rendered on its partial records (the
summary is recomputed from the records whenever `summary.json` is missing or behind them),
and a finished run is refreshed by re-running the command.

```bash
finn nas report runs/report --sampling runs/sampling \
    --search runs/search_nsga2_graded --search runs/search_nsga2_binary \
    --search runs/search_nsga2_discard --search runs/search_random \
    --groundtruth runs/groundtruth \
    --outcome-random runs/groundtruth --outcome-front runs/groundtruth_front
```

Any subset of `--sampling`, `--search` (repeatable), `--groundtruth` and the pair
`--outcome-random`/`--outcome-front` is rendered. The sampling run doubles as the baseline of
the search figures (feasibility rate, cumulative feasible candidates, background points of
the Pareto plot). The tables are printed and all files are written into `OUTPUT_DIR`;
`summary.md` concatenates the tables of the rendered reports.

Python API (`finn.nas.experiments.report`): `render_sampling(dir, out)`,
`render_search(dirs, out, baseline=None)`, `render_groundtruth(dir, out)`,
`render_outcome(random_dir, front_dir, out)` and
`render_all(out, sampling=..., search=[...], groundtruth=..., outcome=(random, front))`.
Each returns the list of written files. The individual `fig_*` functions and the loaders (`load_sampling`,
`load_groundtruth`, `report_search.load_search`) are importable for custom figures.

## Files

Every figure is written as `<name>.pdf` (vector) and `<name>.png` (200 dpi). Sizes are
single-column (3.4 in wide) unless the figure has two panels (7 in).

### Sampling (`--sampling`)

| file | content |
|---|---|
| `sampling_failure_histogram` | oracle verdict of the samples: feasible plus one bar per failure category, counts and percentages of all samples at the bar ends |
| `sampling_failure_op_types` | the twelve op types blamed most often, stacked by failure category |
| `sampling_feasibility_vs_size` | feasibility rate per node-count bin (width 4) and per `log10(parameters)` bin (width 0.5), the number of samples per bin above each bar |
| `sampling_estimates_lut_fps` | feasible samples: estimated LUTs (log) vs. estimated throughput (log), color = `log10(parameters)` |
| `sampling_estimates_bram_lut` | feasible samples: estimated BRAM (18K blocks) vs. LUTs (log) |
| `sampling_proxy_availability` | fraction of the feasible candidates with a finite score per zero-cost proxy (a proxy counts as missing when it is in `proxies.errors` or its score is not finite) |
| `sampling_summary.md` | headline numbers (count, feasible, rate, mean seconds, mean nodes, median LUTs and fps of the feasible samples), the failure categories with their share of the samples and of the failures, the five most blamed op types |

### Search (`--search`, one directory per variant)

A variant is named after its directory (`search_` stripped); `config.json` provides the
algorithm, feasibility handling and violation kind for the legend label. The four standard
variants have a pinned color and marker (`nsga2_graded` blue circle, `nsga2_binary` orange
square, `nsga2_discard` aqua triangle, `random` yellow diamond), so a figure with any subset
of them keeps the same encoding; further variants take the next free slots.

| file | content |
|---|---|
| `search_feasibility_per_generation` | feasibility rate per generation, one line per variant, the sampling rate as a dashed reference: the key figure (does feedback let the search learn the constraints?) |
| `search_cumulative_feasible` | distinct feasible genotypes vs. trials spent (duplicates and failures included); the sampling run as a dashed reference |
| `search_hypervolume` | 2-D hypervolume (`summary.json: hypervolume.per_generation_2d`, the proxy vs. LUT projection) of the feasible trials up to each generation; the axis is scaled by a power of ten named in the label; one panel per proxy objective, since hypervolumes of different proxies are not comparable |
| `search_pareto_fronts` | the final Pareto fronts (entries of `pareto_front` with a real proxy score) as estimated LUTs (log) vs. proxy score, the feasible baseline samples as faint background points; the plane is that of the proxy most variants optimize, fronts of variants with another proxy objective are projected onto it by looking the score up in their records |
| `search_violation_per_generation` | mean constraint violation per generation of the `constraint` variants (all variants if none uses constraints); only the graded violation says more than the feasibility rate |
| `search_summary.md` | per variant: proxy objective, feasible/trials, rate, generations, rate of the last generation, Pareto front size, final 2-D hypervolume, distinct feasible genotypes, duplicates; plus a baseline row |

Notes: generations of a `discard` run are longer than the population size (a generation
lasts until `population_size` feasible candidates were found), so its per-generation lines
have fewer points. Trials with sentinel values (`|value| >= 1e8`) never appear in the Pareto
plot. The hypervolume axis is autoscaled rather than anchored at zero because the first
feasible generation already covers most of the volume; the figure compares the progress of
the variants.

### Ground truth (`--groundtruth`)

| file | content |
|---|---|
| `groundtruth_correlations` | Spearman and Kendall correlation of every proxy (and the size statistics `params`, `estimated_lut`) with the best validation accuracy, sorted, `n` per proxy at the right |
| `groundtruth_proxy_scatter` | the three proxies with the largest absolute Spearman correlation and the parameter count against the best validation accuracy (log or symlog x-axis when the values span decades) |
| `groundtruth_accuracy` | histogram of the best validation accuracies (5 % bins) and accuracy vs. estimated LUTs |
| `groundtruth_summary.md` | trained/usable counts, median accuracy and the correlation table (proxy, n, Spearman, p, Kendall, p) sorted by absolute Spearman correlation |

Rows whose training raised an error are excluded (as in the driver). With fewer than three
usable rows no correlation exists; the figures then show placeholders.

### Outcome (`--outcome-random DIR --outcome-front DIR`)

Does the search's Pareto front reach better trained accuracy per LUT than random feasible
candidates? Both inputs are `finn nas groundtruth` directories: the trained feasible
candidates of the sampling run ("random feasible") and the trained Pareto front of a search
run ("search Pareto front"). Rows with a training error are excluded.

| file | content |
|---|---|
| `outcome_accuracy_vs_lut` | left: best validation accuracy vs. estimated LUTs (log), right: vs. estimated throughput (log); both series as points plus their upper envelope as a step line (the best accuracy reachable with at most *x* LUTs resp. at least *x* frames/s). Random candidates take the muted color of the background samples, the front a series color no variant is pinned to |
| `outcome_summary.md` | per series: n, median and maximum accuracy, median LUTs, the envelope accuracy within 1k/2k/5k/10k LUTs, and the hypervolume of (accuracy, LUTs) with the reference point (0 %, largest LUT count of both series), raw (percent points x LUTs) and normalized by 100 % x reference LUTs |

## Design

- One palette for every figure (the categorical series order blue, orange, aqua, yellow,
  magenta, green, violet, red; a one-hue blue ramp for magnitudes), identity never rests on
  color alone (markers per variant, direct labels on bars, legends underneath the axes so they
  never cover data), hairline grid and axes, 8 pt text.
- Figures are built as `matplotlib.figure.Figure` objects, never through `pyplot`, so
  rendering has no global state and the tests need no backend configuration.
- Records files are read line by line and a line that does not parse (the last line of a
  running experiment, caught mid-write) is skipped.
- Directories with zero feasible candidates (or no records at all) render every file; the
  affected axes carry a placeholder text.

## Tests

`tests/nas/test_report.py` (`util`) builds small synthetic sampling, search and ground-truth
directories in `tmp_path` with the documented record formats and checks that every figure and
table is produced and that the headline numbers in the tables are right, including the
zero-feasible and empty-directory cases and the summary recomputation when `summary.json` is
missing or stale. The real experiment directories are not needed.
