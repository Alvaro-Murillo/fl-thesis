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
  - quantity.py - `quantity_only`, MinSize-Dirichlet quantity skew (adapted code)
  - allocations.py - `pure_label`, label skew with equal client sizes
  - two_stage_method.py - `combined`, quantity_only sizes then label skew within them
  - calibration.py - alpha -> label-HD and beta -> quantity-HD curves and their
    inverses
- models/ - CNN and DNN architectures
- federated/ - the federated simulation, a custom loop (not Flower)
  - data.py - the training set on the device, clients as index tensors, seeded batches
  - loop.py - client selection, broadcast, local training with a fresh optimizer per
    client and round, a pluggable server rule, per-round test evaluation
- experiments/ - config schema, seeding, recording, runner scripts
  - config.py - frozen dataclasses, YAML load/dump, `config_hash`
  - seeding.py - seed derivation per trial, torch determinism
  - recording.py - partition store, run records, atomic writes, provenance
  - datasets.py - dataset label registry (currently `cifar10`)
  - run_partition.py - generate and record one partition
  - grid.py - grid-file expansion; run_grid.py - the partition-stage sweep
  - quantity_levels.py - the 20-seed feasibility check of a grid's quantity levels
- config/ - base.yaml, smoke.yaml, smoke_quantity_only.yaml, smoke_pure_label.yaml,
  smoke_combined.yaml, grids/ (one per research question, quantity_skew.yaml, and the
  candidate-level grids rq2_candidates.yaml and rq3_candidates.yaml)
- calibration/ - committed calibration curves, `{calibration_id}.json`
- tests/ - verification suite
- analysis/ - result analysis and figure-generation scripts
  - fedartml_comparison_label_skew.py - shared-seed check of our label-skew
    partitioner against FedArtML on real CIFAR-10
  - fedartml_comparison_quantity_skew.py - the same check for `quantity_only`
  - quantity_skew_measurements.py - quantity-HD range, beta for target levels, seed
    spread, floor behaviour and incidental label-HD
  - partition_validation.py - nominal vs measured HD for every target cell of a
    grid, per-client label histograms, and the feasibility map

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
    python -m partitioning.calibration --K 30 --mode pure_label

Build the beta -> quantity-HD curve for quantity_only:

    python -m partitioning.calibration --K 30 --mode quantity_only

Compare against FedArtML on shared seeds (no training), and measure quantity skew:

    python -m analysis.fedartml_comparison_label_skew
    python -m analysis.fedartml_comparison_quantity_skew
    python -m experiments.run_grid --grid config/grids/quantity_skew.yaml
    python -m analysis.quantity_skew_measurements

Validate target levels before any training (partition-only; about a minute of CPU).
`analysis.partition_validation` defaults to the two candidate grids; `--grid`
(repeatable) points it at any grid that sets HD targets:

    python -m experiments.run_grid --grid config/grids/rq2_candidates.yaml
    python -m experiments.run_grid --grid config/grids/rq3_candidates.yaml
    python -m experiments.quantity_levels --grid config/grids/rq3_candidates.yaml
    python -m analysis.partition_validation

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

- `quantity_only` - deliberate quantity skew, replicating FedArtML 0.1.34's
  `minsize-dirichlet`: client shares ~ Dir(beta*1_K), every share below
  `quantity_floor`/N raised to it, renormalised, then one label-blind
  permutation is cut into blocks of those sizes. Labels are never read, so
  each client's label mix matches the global one only in expectation.
  - `quantity_floor: null` (default) = FedArtML's 3 * num_classes + 1 (31 on
    CIFAR-10). Renormalising after the floor makes it approximate: realized
    sizes are only >= floor(f*N / (N + m*f)) for m floored clients - 30, not
    31, on CIFAR-10 at K = 30. This is FedArtML's behaviour, kept on purpose.
  - Never redraws: `min_size_policy` must be 0 and `max_redraws` stay at its
    default. Rows record the resolved `quantity_floor` and `n_floored_clients`.
  - On a shared RNG stream it matches FedArtML byte-for-byte.
  - `partitioning.quantity.draw_client_sizes` is the size draw on its own, for
    the nested `combined` partitioner to import (not copy).

- `pure_label` - label skew with no quantity skew. Every client holds N // K
  samples, or one more for N % K clients drawn at random. Each client draws a
  label mix p_k ~ Dir(alpha * C * Q_global), where Q_global is the empirical
  label distribution, so demand matches stock in expectation; on balanced data
  this is Dir(alpha * 1_C). Within one draw demand and stock never match
  exactly, so the requested counts b_k * p_k are scaled to the budgets and
  the class stock: the Sinkhorn / iterative proportional fitting scaling,
  i.e. the closest matrix in KL divergence. Every cell is then rounded down or
  up with both margins exact, so the correction is spread over all clients.
  The scaling is solved by Newton's method on its dual, because the
  alternating Sinkhorn updates stall at alpha <= 0.01.
  - Never redraws: `min_size_policy` must be 0 and `max_redraws` stay at its
    default. Rows record `n_redraws` 0.
  - The drawn mix and the realized mix still differ, and most of all at small
    alpha (CIFAR-10, K = 30: mean Hellinger drift 0.17 at alpha = 0.1, 0.06 at
    alpha = 1). Rows record that drift (see below).

- `combined` - the two-stage nested method (`partitioning/two_stage_method.py`).
  Stage 1 draws the client sizes exactly as `quantity_only` does
  (`draw_client_sizes`, same `quantity_floor`); Stage 2 draws each client's
  label mix from the same prior as `pure_label` and fits the mixes to those
  sizes and the class stock. The fit is the closest set of mixes in KL
  divergence with every client counted once, not by its size: the correction
  to client k scales with b_k / max b, so the large clients absorb the
  supply/demand mismatch and the small ones keep their draws. With equal sizes
  it is the same scaling as `pure_label`. Counts are rounded with both
  margins exact, and client sizes always equal the Stage 1 draw.
  - Sizes, mixes and the per-class shuffles come from three streams spawned
    from the partition seed, so the sizes do not depend on alpha and the drawn
    mixes do not depend on beta. They do not match `quantity_only`'s or
    `pure_label`'s draws at the same seed.
  - Never redraws: `min_size_policy` must be 0 and `max_redraws` stay at its
    default. Rows record the resolved `quantity_floor`, `n_floored_clients`
    and `n_redraws` 0.

`pure_label` and `combined` rows record how far each client's realized label
mix is from its draw: `label_drift` (per client, Hellinger, in client order;
null for an empty client), `mean_label_drift` and `max_label_drift`. These are
null for the modes that draw no per-client mix.

## Grid files

    base      complete config; every field not swept comes from it
    constant  dotted overrides applied to every cell
    cases     optional list of override sets, crossed with the sweep - for
              settings that only make sense together (size_cap is per-class only)
    sweep     dotted key -> list of values; cells are the cartesian product
    status    `blocked` makes the grid unrunnable

The RQ1 grid is 3 cases (per-class uncapped, per-class niid_bench, per-client)
x 10 alphas x K in {10, 30, 100} x 3 trials = 270 partitions, about 40 s of CPU.

The quantity-skew grid (`quantity_skew.yaml`) is quantity_only at the 30 betas of the beta
calibration curve (log-spaced, 0.001-1000) x K in {10, 30, 100} x 3 trials =
270 partitions; its K = 30 cells are the curve's own partitions.

A cell can give HD targets instead of parameters. With `alpha` null,
`target_label_hd` is resolved at expansion against the curve `calibration_id`
names; with `beta` null, `target_quantity_hd` against the curve
`quantity_calibration_id` names. The curve must match the cell's dataset, K
and partitioner, or expansion fails. A target of 0 means no skew on that axis
(concentration 1e6), since realized HD is never exactly 0. An explicit
parameter is kept, and its target then only feeds the checks below.

On the quantity axis, targets are level names: every size draw is kept, each
row records its realized quantity-HD and its deviation from the target, and
the ±0.05 tolerance applies to the level, whose mean realized quantity-HD over
20 seeds (trials 0-19) must be within 0.05 of the target. The label-axis
tolerance stays per partition. In rows, `feasible` is the label check only;
`label_hd_deviation` and `quantity_hd_deviation` (realized - target) are
diagnostics, and `client_sizes` shows which cells share a size draw (same
dataset, K, mode, quantity_floor, beta and partition_seed). The level check
is its own step and excludes nothing:

    python -m experiments.quantity_levels --grid config/grids/rq3.yaml

The RQ3 grid (`rq3.yaml`, still a stub) is all `combined`: 3 label-HD x 3
quantity-HD levels, the first on each axis 0, so its edges are the single-skew
arms and its corner is the no-skew baseline. combined reads alpha off the
pure_label curve and beta off the quantity_only curve. Its sizes come from the
same draw, and its label-HD stayed within 0.03 of pure_label's at the same
alpha across the quantity range.

The candidate-level grids hold every level the RQ2 and RQ3 grids might use, so
levels are chosen from measured feasibility rather than assumed. Both are
CIFAR-10, K = 30, trials 0-2 (the trained trials, so a chosen level trains on
the very partitions that were validated):

- `rq2_candidates.yaml` - label targets 0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6,
  0.7, 0.75 for natural_per_class uncapped, natural_per_class niid_bench and
  pure_label, each arm on its own alpha curve (90 partitions).
- `rq3_candidates.yaml` - combined at those label targets x quantity targets
  0-0.8 in steps of 0.1 (270 partitions).

`analysis.partition_validation` applies the rules above: a cell is feasible if
every one of its partitions is within 0.05 of its label target and its
quantity level, if it has one, passes the 20-seed check. It reports and
excludes nothing.

## Results layout

Everything lands under `results/` (gitignored), written atomically:

- `partitions/{partition_id}.npz` - client index arrays
- `partitions/{partition_id}.json` - realized metrics and provenance
- `partitions.jsonl` - one row per partition; this is the RQ1 dataset on its own.
  `status` is `generated`, or `infeasible_min_size` (no .npz, no metrics).
  Append-only: readers take the last row per `partition_id`
- `quantity_levels.jsonl` - one row per quantity level checked: the 20-seed
  realized quantity-HD, its mean, sd and range, and `level_feasible`
- `runs.jsonl` - one row per run at start (`running`) and at the end
  (`completed` / `failed` / `oom`); readers take the last row per `run_id`
- `runs/{run_id}/history.csv` - per-round accuracy and loss, flushed each round
- `runs/{run_id}/config.yaml` - the resolved config, so a row stays readable
  after the schema moves on
- `analysis/` - analysis-script outputs, e.g. `fedartml_comparison_label_skew.csv`,
  `fedartml_comparison_quantity_skew.csv`, and `quantity_skew_*` tables and scatter
- `analysis/partition_validation_*` - `partitions.csv` (one row per partition
  and target pair), `clients.csv` (one row per client: size, class counts, HD to
  global, drift), `cells.csv` (one row per target cell with its verdict), and
  the figures `label.png` and `quantity.png` (nominal vs measured per axis) and
  `feasibility.png` (deviation from target per cell, failing cells crossed)

Partitions are addressed by the fields that determine the split, so training
knobs never invalidate a cached partition, and FedAvg and FedProx at the same
grid cell provably read identical splits. Reruns are byte-identical.

Calibration curves live in `calibration/` and are committed, because RQ2 and
RQ3 target levels are resolved against them. Each file records its parameter grid,
trial seeds (the grid's own), per-trial values, a monotone fit and the git
commit. Label-skew curves are indexed by alpha and fitted on label-HD
(`alpha_for_target`); quantity_only curves carry `"parameter": "beta"` and are
fitted on quantity-HD (`beta_for_target`). Files without a `parameter` key
are alpha curves.

`da8b82c519` is the CIFAR-10 K = 30 pure_label alpha curve (3 trials).
combined's label targets resolve against it too. It is within 0.009 of a
20-trial curve, because label-HD varies far less across trials than
quantity-HD (sd at most 0.014).

The CIFAR-10 K = 30 beta curve exists twice: `f09e0cfdf0` (3 trials, the
grid's own, used by `analysis/quantity_skew_measurements.py`) and `8f995f9546`
(20 trials, `--trials 20`), which grids resolve quantity targets against.
Realized quantity-HD at one beta has a standard deviation of up to 0.047
across 20 partitions (at quantity-HD 0.44-0.57), and with 3 trials the curve
was off by up to 0.065 from a 1000-seed mean. With 20 trials it is off by at
most 0.022, and by at most 0.004 for targets 0.1-0.6.
Its trials 0-19 are the partitions of that analysis script's 20-seed sweep.

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
- `partitioning/quantity.py` `draw_client_sizes` and the permutation + split of
  `quantity_only` - from FedArtML 0.1.34
  `SplitAsFederatedData.dirichlet_method_quant_skew(method="minsize-dirichlet")`,
  whose permutation + Dir(beta) + cumsum split comes from NIID-Bench's
  `iid-diff-quantity` partition. This replicates the source paper's
  quantity-only condition; it is not a thesis contribution, and the nested
  `combined` partitioner imports it rather than copying it.

Each adapted function carries a header with source, version, licence and
changes.

AI assistance: parts of this code base were written with Claude Code
(Anthropic) from specifications written by the author, and reviewed and tested
by the author. Modules produced this way say so in their docstring.
`partitioning/allocations.py` and `partitioning/two_stage_method.py` were
written by the author; their docstrings list the changes made to them with
Claude Code.

The thesis's own contributions are the author's own design: the HD-to-global
label and quantity metrics (`partitioning/metrics.py`), the
`natural_per_client` and `pure_label` constructions, the alpha -> label-HD and
beta -> quantity-HD calibration (`partitioning/calibration.py`), and the
two-stage nested partitioning method (`partitioning/two_stage_method.py`).

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
- Reproducibility scaffolding complete: config schema, seeding, partition
  store and results schema, with 48 tests passing
  - Determinism verified on the GPU: bit-exact reruns, +15.7% per step
  - Smoke partition (K=10, alpha=0.5) reproduces byte-identically; client
    sizes span 3104-8028, the quantity-skew byproduct RQ1 exists to measure
- RQ1 grid written (63 partitions, no GPU needed); RQ2/RQ3 grids stubbed
  behind `status: blocked` pending the target-level calibration table
- Natural label-skew partitioning complete: 105 tests pass (+1 slow)
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
- Quantity-skew partitioning (`quantity_only`) complete
  - Matches FedArtML's minsize-dirichlet byte-for-byte on a shared RNG stream;
    on real CIFAR-10 (K=30, 10 seeds, 6 betas) the two agree in distribution
    (all |z| < 1.6)
  - CIFAR-10, K=30: quantity-HD reaches 0.83 (analytic ceiling 0.830 at the
    realized floor of 30); FedArtML's pairwise scale reaches 0.176 of a
    possible 0.183
  - Sampling is label-blind, but label-HD rises from 0.025 (equal split) to
    ~0.21 at small beta, because many floored ~30-sample clients carry large
    sampling noise in an unweighted mean
  - Beta calibration curve committed; quantity-skew grid of 270 partitions
    generated
- Quantity-corrected label skew (`pure_label`) complete: 218 tests pass (with
  --runslow)
  - Equal client sizes (quantity-HD <= 1e-4 on CIFAR-10 at K = 10, 30, 100)
  - Counts come from the KL-closest matrix with the budgets and class stock as
    margins, rounded with both margins exact. The earlier client-by-client fill
    left the whole mismatch to the last clients (CIFAR-10, K = 10, alpha = 1:
    max drift 0.76; now 0.16)
  - CIFAR-10, K = 10, alpha = 1: label-HD 0.30, vs 0.29 for natural_per_class
- Partition validation over the candidate levels (CIFAR-10, K = 30, trials 0-2):
  116 of 120 target cells feasible
  - Every label-axis arm hits every label target from 0 to 0.75 in every trial
    (largest miss 0.020; largest mean miss 0.013). At matched realized label-HD
    the natural arms carry incidental quantity-HD up to 0.39 (uncapped) and 0.32
    (capped); pure_label carries 0.0001
  - Every combined quantity level 0-0.8 passes its 20-seed check
  - combined's label-HD rises with quantity skew: by about +0.015 at quantity
    level 0.6 and +0.03 to +0.04 at 0.8 (20-seed means). The 4 failing cells,
    label target 0.25 at quantity 0.8 and 0.3 at 0.6-0.8, are that rise on top of
    trials 0-2 sitting about 0.015 above the 20-seed mean
- Federated simulation loop: a custom loop rather than Flower, since the
  clients are simulated in one process with no networking
  - Each client's batch order comes from its own (train_seed, round, client)
    stream, so an update does not depend on client order; reruns are identical
  - Real CIFAR-10, K = 30, equal sizes, strict determinism: 2.36 ms/step
    including orchestration and per-round test evaluation (2.10 in the bare
    step benchmark), about 13 min per 40-round x 10-epoch run; 1.05 GB peak VRAM
    with the whole dataset on the GPU
- Next: RQ2 and RQ3 levels chosen from the validation tables; capped vs
  uncapped for the RQ2 natural arm
