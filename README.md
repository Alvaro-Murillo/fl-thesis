# Federated Learning: Label and Quantity Skew Interaction

Bachelor thesis project quantifying and disentangling label and quantity skew
in Federated Learning - measuring the quantity-skew byproduct of standard
Dirichlet label partitioning and testing deliberate label-quantity interaction
effects via a two-stage nested partitioning method.

## Setup

Python 3.12.10 (virtual environment)

pip install -r requirements.txt

Key dependencies: torch 2.14.0+cu126 / torchvision 0.29.0+cu126 (CUDA 12.6),
flwr 1.36.0 (Flower), FedArtML 0.1.34, scikit-learn, pandas, numpy,
matplotlib, seaborn.

requirements.txt was frozen from the GPU-verified environment on the main
development machine (desktop with RTX 3070 Ti). Run `python tests/env_report.py`
to print the active environment's platform, package versions, and CUDA/GPU
details.

All commands assume the project root as the working directory, and the venv
interpreter (`.venv\Scripts\python.exe` on the desktop machine).

## Structure

- partitioning/ - HD measurement, Dirichlet partitioning, two-stage nested allocator
  - metrics.py - Hellinger distance, realized skew metrics per partition
  - fedartml_compat.py - FedArtML's pairwise HD scale (adapted code, translation only)
  - dirichlet.py - the two natural label-skew modes, `natural_per_class` and
    `natural_per_client`
  - calibration.py - alpha -> realized label-HD curves and their inverse
- models/ - CNN and DNN architectures
- federated/ - FedAvg, FedProx, training loop
- experiments/ - config schema, seeding, recording, runner scripts
  - config.py - frozen dataclasses, YAML load/dump, `config_hash`
  - seeding.py - seed derivation per trial, torch determinism
  - recording.py - partition store, run records, atomic writes, provenance
  - datasets.py - dataset label registry (currently `cifar10`)
  - run_partition.py - generate and record one partition
  - grid.py - grid-file expansion; run_grid.py - the partition-stage sweep
- config/ - base.yaml, smoke.yaml, grids/ (one per research question)
- calibration/ - committed calibration curves, `{calibration_id}.json`
- tests/ - verification suite
- analysis/ - result analysis and figure-generation scripts
  - b2_fedartml_comparison.py - shared-seed check of our partitioner against
    FedArtML on real CIFAR-10

## Usage

Run the test suite. Slow tests (real CIFAR-10, FedArtML) are skipped unless
`--runslow` is passed:

    python -m pytest tests/ -q
    python -m pytest tests/ -q --runslow

Generate a partition (cached by `partition_id`; re-running is a no-op):

    python -m experiments.run_partition --config config/smoke.yaml

Generate every partition in a grid (labels load once; a killed sweep resumes
where it stopped; grids marked `status: blocked` are refused):

    python -m experiments.run_grid --grid config/grids/rq1.yaml

Build an alpha -> label-HD calibration curve (writes `calibration/{id}.json`):

    python -m partitioning.calibration --K 30 --mode natural_per_class --size-cap none
    python -m partitioning.calibration --K 30 --mode natural_per_class --size-cap niid_bench
    python -m partitioning.calibration --K 30 --mode natural_per_client

Compare against FedArtML on shared seeds (no training):

    python -m analysis.b2_fedartml_comparison

Check the environment and estimate the compute budget:

    python tests/env_report.py
    python tests/gpu_check.py

## Partitioning modes

Both natural modes assign every sample and let client sizes emerge from the
draw; nothing rebalances them. Any quantity skew they show is the byproduct
RQ1 measures.

- `natural_per_class` - one Dir(alpha*1_K) draw per class splits that class
  across clients (Lin et al. 2020 / NIID-Bench).
  - `size_cap: none` (default) - no balancing term.
  - `size_cap: niid_bench` - FedArtML 0.1.34's balancing term: a client
    already holding >= N/K samples gets no share of the next class. The cap
    is checked once per class, before the split, so it can overshoot N/K
    within one class. On a shared RNG stream this matches FedArtML exactly.
- `natural_per_client` - every client draws a label mix from Dir(alpha*1_C);
  each class's full stock is then split in proportion to that class's column
  of the K x C matrix. As alpha -> inf this converges to the empirical label
  distribution, even on imbalanced data.

`min_size_policy` redraws until every client holds that many samples, with
at most `max_redraws` attempts (default 1000). An exhausted budget is never
relaxed or looped past: the cell is recorded as `infeasible_min_size`.

`pure_label`, `quantity_only` and `combined` arrive with B4, B3 and B5.

## Grid files

    base      complete config; every field not swept comes from it
    constant  dotted overrides applied to every cell
    cases     optional list of override sets, crossed with the sweep - for
              settings that only make sense together (size_cap is per-class only)
    sweep     dotted key -> list of values; cells are the cartesian product
    status    `blocked` makes the grid unrunnable

The RQ1 grid is 3 cases (per-class uncapped, per-class niid_bench, per-client)
x 10 alphas x K in {10, 30, 100} x 3 trials = 270 partitions, about 40 s of CPU.

## Results layout

Everything lands under `results/` (gitignored), written atomically:

- `partitions/{partition_id}.npz` - client index arrays
- `partitions/{partition_id}.json` - realized metrics and provenance
- `partitions.jsonl` - one row per partition; this is the RQ1 dataset on its own.
  `status` is `generated`, or `infeasible_min_size` (no .npz, no metrics).
  Append-only: readers take the last row per `partition_id`
- `runs.jsonl` - one row per run at start (`running`) and at the end
  (`completed` / `failed` / `oom`); readers take the last row per `run_id`
- `runs/{run_id}/history.csv` - per-round accuracy and loss, flushed each round
- `runs/{run_id}/config.yaml` - the resolved config, so a row stays readable
  after the schema moves on
- `analysis/` - analysis-script outputs, e.g. `b2_fedartml_comparison.csv`

Partitions are addressed by the fields that determine the split, so training
knobs never invalidate a cached partition, and FedAvg and FedProx at the same
grid cell provably read identical splits. Reruns are byte-identical.

Calibration curves live in `calibration/` and are committed, because RQ2
target levels are resolved against them. Each file records its alpha grid,
trial seeds (the grid's own), per-trial values, a monotone fit and the git
commit.

## Conventions

- Randomness: one `trial` index derives independent partition/init/train seeds.
  Partitioners take an explicit `np.random.Generator`; global `np.random` is
  never touched (the FedArtML comparison saves and restores it around the one
  library call that reseeds it).
- Determinism is strict by default (`runtime.deterministic`), costing ~16% per
  step on the RTX 3070 Ti. Every run row records which mode produced it.
- Label-HD is the unweighted mean over non-empty clients of HD(client,
  global), measured against the empirical global label distribution, never
  against uniform - CIFAR-10 is class-balanced, Covtype (RQ4) is not.
- Quantity-HD is one distance, HD(n_k/N, uniform 1/K); empty clients count in
  it and are reported as `n_empty_clients`.
- FedArtML's pairwise values (`*_pairwise`) are recorded only to translate to
  that literature, never used as grid axes. For quantity the conversion is
  exact: HD = sqrt(1 - sqrt(1 - (K-1) * HD_pairwise^2)).
- Class counts come from the dataset registry; nothing assumes 10 classes.
- Stable identifiers: a config field added later enters `partition_id` and
  `config_hash` only when it differs from its default (`_ADDED_FIELD_DEFAULTS`
  in config.py), so cached partitions and completed runs survive schema growth.

## Third-party code and AI assistance

Adapted code (MIT; full licence texts in `LICENSES/`):

- `partitioning/fedartml_compat.py` `pairwise_hellinger` - from FedArtML 0.1.34
  `hellinger_distance` (Copyright (c) 2022 La-Sapienza-University-Rome). Used
  only to translate to FedArtML's pairwise scale, never as a grid axis.
- `partitioning/dirichlet.py` `_split_by_proportions` and the per-class loop of
  `natural_per_class` - from FedArtML 0.1.34
  `SplitAsFederatedData.dirichlet_method`, itself derived from NIID-Bench
  (Copyright (c) 2021 Yiqun Diao, Qinbin Li; Copyright (c) 2020 International
  Business Machines).

Each adapted function carries a header with source, version, licence and
changes.

AI assistance: parts of this code base were written with Claude Code
(Anthropic) from specifications written by the author, and reviewed and tested
by the author. Modules produced this way say so in their docstring.

The thesis's own contributions are the author's own design: the HD-to-global
label and quantity metrics (`partitioning/metrics.py`), the
`natural_per_client` construction, the alpha -> label-HD calibration
(`partitioning/calibration.py`), and (future) the B4 pure-label and B5 nested
partitioning methods.

## Progress log

- Repo scaffolding created: folder structure, .gitignore, README
- Conda: version 25.5.1 confirmed installed
- fl-thesis conda environment created then removed (was being set up on
  laptop by mistake - full environment to be built on main desktop machine
  with RTX 3070 Ti instead)
- Virtual environment (Python 3.12.10) set up on main desktop machine (RTX
  3070 Ti); dependencies frozen to requirements.txt (PyTorch 2.14.0+cu126,
  Flower 1.36.0, FedArtML 0.1.34, etc.)
- GPU benchmark (tests/gpu_check.py): 1.8 ms/step at batch 64, 42 MB peak VRAM.
  Floor estimate ~9.6 min per FL run, ~9.6 h for a 60-run grid - excludes data
  loading, evaluation and FL orchestration, so treat it as a floor
- A4 reproducibility scaffolding complete: config schema, seeding, partition
  store and results schema, with 48 tests passing
  - Determinism verified on the GPU: bit-exact reruns, +15.7% per step
  - Smoke partition (K=10, alpha=0.5) reproduces byte-identically; client
    sizes span 3104-8028, the quantity-skew byproduct RQ1 exists to measure
- RQ1 grid written (63 partitions, no GPU needed); RQ2/RQ3 grids stubbed
  behind `status: blocked` pending the B6 calibration table
- B2 natural label-skew partitioning complete: 105 tests pass (+1 slow)
  - `size_cap: niid_bench` matches FedArtML exactly on a shared RNG stream;
    on real CIFAR-10 (K=30, 10 seeds, 5 alphas) FedArtML and the local capped
    variant agree in distribution
  - The cap lowers quantity skew but raises label skew at the same alpha
    (alpha=1, K=30: label-HD 0.36 vs 0.30), so capped and uncapped results
    must be compared at matched label-HD, not matched alpha
  - `natural_per_client` gives about the same label-HD as per-class at equal
    alpha with 3-10x less quantity-HD (K=30, alpha=1: 0.014 vs 0.104)
  - RQ1 grid extended to 270 cells; the 6 per-class cells at K=100,
    alpha=0.03 cannot meet min_size=10 in 1000 redraws and are recorded as
    infeasible
  - Calibration curves for CIFAR-10, K=30: label-HD reaches ~0.76
    (per-class), ~0.77 (capped) and ~0.81 (per-client)
- Next: B6 target label-HD levels from the curves; capped vs uncapped for the
  RQ2 natural arm
