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

## Datasets

- CIFAR-10, through torchvision, downloaded to `data/` on first use.
- Covtype: Blackard, J. (1998). Covertype [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C50K5N. Licensed under Creative Commons
  Attribution 4.0 International (CC BY 4.0). 581,012 samples, 54 features, 7
  classes from 0.47% to 48.8% of the data. Fetched with scikit-learn's
  `fetch_covtype` (scikit-learn 1.9.0) and cached under `data/covtype/`
  (`python -m experiments.covtype`). The arrays are checked against their sha256 on
  every load: data `d9a3cc75...`, target `e2a1e021...` (full values in
  `experiments/covtype.py`). The split and preprocessing are this project's choices,
  since the source paper gives only the sizes: a stratified split with a fixed seed
  and exactly 58,102 test samples, so 522,910 train as in the paper's Table 2;
  labels 1-7 become 0-6; the 10 continuous columns are standardised with the
  training set's mean and sd, and the 44 binary columns are left as they are.

## Structure

- partitioning/ - HD measurement, Dirichlet partitioning, two-stage nested allocator
  - metrics.py - Hellinger distance, realized skew metrics per partition
  - fedartml_compat.py - FedArtML's pairwise HD scale (adapted code, translation only)
  - natural_methods.py - the two natural label-skew modes, `natural_per_class` and
    `natural_per_client`
  - quantity_only_method.py - `quantity_only`, MinSize-Dirichlet quantity skew (adapted code)
  - pure_label_method.py - `pure_label`, label skew with equal client sizes
  - two_stage_method.py - `combined`, quantity_only sizes then label skew within them
  - calibration.py - alpha -> label-HD and beta -> quantity-HD curves and their
    inverses
- models/ - CNN and DNN architectures
  - networks.py - the source paper's CNN (images) and DNN (tabular), Glorot-uniform
    weights and zero biases drawn from the init seed
- federated/ - the federated simulation, a custom loop (not Flower)
  - data.py - the training set on the device, clients as index tensors, seeded batches
  - loop.py - client selection, broadcast, local training with a fresh optimizer per
    client and round, a pluggable server rule, per-round test evaluation
  - aggregation.py - FedAvg's server rule, the sample-weighted average of the client
    models (adapted code)
  - proximal.py - FedProx's proximal term, its gradient mu * (w - w_global) added to
    each client's gradients before the optimizer step (adapted code)
  - centralized.py - the whole training set as one client with one optimizer: the
    centralized baseline and the models' sanity check
  - confusion.py - the global model's test-set confusion matrix, per-class accuracy
    and balanced accuracy (the thesis's addition; the source paper reports accuracy)
- experiments/ - config schema, seeding, recording, runner scripts
  - config.py - frozen dataclasses, YAML load/dump, `config_hash`
  - seeding.py - seed derivation per trial, torch determinism
  - recording.py - partition store, run records, atomic writes, provenance
  - datasets.py - dataset label registry (`cifar10`, `covtype`)
  - tensors.py - training and test tensors, images as float32 in [0, 1] and Covtype's
    standardised features as they are, in the same sample order as the label registry
  - covtype.py - Covtype's fetch, checksums, stratified split and standardisation
  - run_partition.py - generate and record one partition
  - grid.py - grid-file expansion; run_grid.py - the partition-stage sweep
  - run_training.py - train one config or a whole grid on stored partitions, with
    resume; a centralized config trains the baseline and reads no partition
  - quantity_levels.py - the 20-seed feasibility check of a grid's quantity levels
  - per_class_accuracy.py - per-class test accuracy of recorded runs, by retraining
    them through run_training's calls; a rerun that does not reproduce its recorded
    run bit for bit is refused. A run trained with per-class logging is read from its
    own confusion.csv instead, checked against its history.csv and runs.jsonl row
- config/ - base.yaml, smoke.yaml, smoke_quantity_only.yaml, smoke_pure_label.yaml,
  smoke_combined.yaml, the pipeline-check configs sanity_iid.yaml,
  sanity_label_skew.yaml and sanity_quantity_skew.yaml, the centralized baseline
  centralized.yaml, Covtype's covtype.yaml and covtype_centralized.yaml, and grids/
  (rq1.yaml, rq2.yaml, rq3.yaml, the partition-only rq3_edge.yaml, quantity_skew.yaml,
  mu_tuning.yaml, rq2_class_order.yaml, centralized.yaml, covtype_rq2.yaml,
  covtype_centralized.yaml, covtype_mu_tuning.yaml, covtype_centralized_steps.yaml and
  the partition-only covtype_rq1.yaml and covtype_quantity_only.yaml)
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
  - training_summary.py - final accuracy and rounds-to-accuracy per condition, mean
    and sd across trials, with each condition's realized HD
  - rq2_class_order.py - RQ2's class-order check: per-class tables, the rule fixed
    before the per-class data it covers, and how concentrated each class is across
    clients
  - paper_outputs.py - every table, figure and quoted number of the paper's results
    from one command, reading the recorded runs and partitions only; paper_data.py
    (inputs by grid and arm), paper_stats.py (mean and sample sd, paired differences,
    RQ3 additivity and regression), paper_results.py (the results and tables),
    paper_tables.py (CSV, LaTeX tabular and Markdown writers) and paper_figures.py

## Usage

Run the test suite. Slow tests (real CIFAR-10 and Covtype, FedArtML) are skipped
unless `--runslow` is passed:

    python -m pytest tests/ -q
    python -m pytest tests/ -q --runslow

Fetch Covtype once (it prints the checksums and per-class counts). If
scikit-learn's mirror refuses the download, take covtype.data.gz from the UCI zip
(https://archive.ics.uci.edu/static/public/31/covertype.zip) and pass it; it must
match the sha256 scikit-learn pins for the file, and it is parsed and cached exactly
as a mirror download would be:

    python -m experiments.covtype
    python -m experiments.covtype --archive path\to\covtype.data.gz

Generate a partition (cached by `partition_id`; re-running is a no-op):

    python -m experiments.run_partition --config config/smoke.yaml

Generate every partition in a grid (labels load once; a killed sweep resumes
where it stopped; `partition_only` and blocked grids are generated like any
other, and only training refuses them):

    python -m experiments.run_grid --grid config/grids/rq1.yaml

Train on stored partitions (generate them first; training never makes one). A grid
is checked for missing partitions before its first run; completed runs are
skipped unless `--force` is passed, and a failed or killed run is trained again:

    python -m experiments.run_partition --config config/smoke.yaml
    python -m experiments.run_training --config config/smoke.yaml
    python -m experiments.run_training --grid config/grids/rq3.yaml

Train one grid in several processes at once, one terminal per shard. Generate every
partition of the grid first, in one process, so no two shards ever make one (training
refuses a grid with a missing partition). Shard K of N trains every N-th run from index
K; runs.jsonl appends are serialised by a file lock, and each shard resumes when its own
command is run again:

    python -m experiments.run_grid --grid config/grids/covtype_rq2.yaml
    python -m experiments.run_training --grid config/grids/covtype_rq2.yaml --shard 0/3
    python -m experiments.run_training --grid config/grids/covtype_rq2.yaml --shard 1/3
    python -m experiments.run_training --grid config/grids/covtype_rq2.yaml --shard 2/3

Train the centralized baseline, trials 0, 1, 2, 5 and 6 (no partition; about 20 s per
run):

    python -m experiments.run_training --grid config/grids/centralized.yaml

Summarise completed runs across trials (all runs, or one grid's):

    python -m analysis.training_summary
    python -m analysis.training_summary --grid config/grids/rq3.yaml
    python -m analysis.training_summary --grid config/grids/centralized.yaml --prefix centralized

Every table, figure and number of the paper's results section, on the reported trials
(CIFAR-10 0, 1, 2, 5, 6; Covtype 0-4) and again on CIFAR-10 trials 0-2, with the
three-trial reproduction check (`--help` for the trial sets, widths and paths; without
`--covtype-fedprox` every Covtype FedProx cell reads "pending"):

    python -m analysis.paper_outputs --covtype-fedprox

Per-class test accuracy of recorded runs: each selected run is retrained and must
reproduce its final accuracy and every round of its history.csv bit for bit, or the
sweep stops (about one run's time per run; nothing is recorded in runs.jsonl).
`--where` narrows a grid by dotted keys, `--list` prints the plan, and `--import`
checks and copies per-class results made earlier with the same calls:

    python -m experiments.per_class_accuracy --grid config/grids/rq2_class_order.yaml
    python -m experiments.per_class_accuracy --grid config/grids/rq2.yaml --where federated.algorithm=fedavg partition.mode=natural_per_class partition.target_label_hd=0.7

RQ2's class-order check, from the stored partitions and the per-class results (each
run's own files, or its rerun for a run trained before per-class logging; trials 5 and
6 are reported beside the rule, descriptively; `--partitions`: the concentration
tables only, no per-class data needed):

    python -m analysis.rq2_class_order --partitions
    python -m analysis.rq2_class_order

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

Make and validate the RQ2 and RQ3 partitions before any training (partition-only,
CPU). Generate from a clean tree, so every row records `git_dirty` false.
`analysis.partition_validation` defaults to rq2.yaml, rq3.yaml and rq3_edge.yaml;
`--grid` (repeatable) points it at any grid that sets HD targets:

    python -m experiments.run_grid --grid config/grids/rq2.yaml
    python -m experiments.run_grid --grid config/grids/rq3.yaml
    python -m experiments.run_grid --grid config/grids/rq3_edge.yaml
    python -m experiments.quantity_levels --grid config/grids/rq3.yaml
    python -m experiments.quantity_levels --grid config/grids/rq3_edge.yaml
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
  - `class_order: index` (default) splits the classes in ascending index
    order, as FedArtML does; `class_order: reversed` splits them in
    descending order and changes nothing else. Under the cap the classes
    split last find the most clients masked, so this moves that position.
    On balanced labels the reversed count matrix is the index-order one with
    its class columns reversed. natural_per_class only.
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
  - `partitioning.quantity_only_method.draw_client_sizes` is the size draw on its own, for
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

    base            complete config; every field not swept comes from it
    constant        dotted overrides applied to every cell
    cases           optional list of override sets, crossed with the sweep - for
                    settings that only make sense together (size_cap is per-class only)
    sweep           dotted key -> list of values; cells are the cartesian product.
                    Required: a grid without one is refused, so a stub never expands
                    to its base config's cell
    partition_only  `true`: the partitions are made and checked, never trained
    status          `blocked` (with `blocked_on`): the partitions can be made and
                    checked, but training waits for what `blocked_on` names
    label           optional text written into every runs.jsonl row of the grid's
                    runs; not part of any config, so it moves no config_hash

Every caller of `experiments.grid.expand` says what it expands for:
`for_training=True` (the training runner only) refuses a `partition_only` or
blocked grid, and the partition and analysis tools pass `for_training=False`. A
grid whose cells are incomplete fails on expansion either way: an alpha with no
target to resolve it from, a target with no curve, or a level that is not a number.

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
(`NO_SKEW`, concentration 1000.0: the source paper's IID alpha, label-HD 0.010 and
quantity-HD 0.011 at K = 30), since realized HD is never exactly 0. A natural cell
at target 0 is therefore the RQ1 grid's alpha = 1000 partition. An explicit
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

The RQ2 and RQ3 levels come from a level rule applied to 20-seed ceilings
(CIFAR-10, K = 30), not from hand-picked candidates:

- Label-HD 0, 0.25, 0.45, 0.70, shared by RQ2 and RQ3. The common ceiling is
  0.759, set by natural_per_class uncapped (capped 0.774, pure_label 0.811).
  Every RQ2 arm hits every level with all 20 seeds within 0.05.
- Quantity-HD 0, 0.35, 0.70 for training, from the ceiling 0.830, plus a
  partition-only edge level at 0.80.

Alpha and beta are read off 20-trial curves: 147e58d391 (natural_per_class
uncapped), 146de82d74 (niid_bench), 90f8f1dd9b (pure_label, and combined's
alpha) and 8f995f9546 (quantity_only, combined's beta). All three grids are
CIFAR-10, K = 30. rq2.yaml and rq3.yaml run trials 0, 1, 2, 5 and 6, the trained
trials (5 and 6 added on 2026-11-11; 3 and 4 tuned mu), rq3_edge.yaml trials 0-2:

- `rq2.yaml` - natural_per_class uncapped, natural_per_class niid_bench and
  pure_label at the four label levels, each arm on its own curve (60
  partitions). Its natural target-0 cells are RQ1's alpha = 1000 partitions.
- `rq3.yaml` - all `combined`, 4 label x 3 quantity levels (60 partitions). The
  first level on each axis is 0, so its edges are the single-skew arms and its
  corner is the no-skew baseline. combined uses one alpha per label level, the
  pure arm's, so cells at one label level share their drawn mixes. Its
  label-HD then rises with quantity skew (0.446 to 0.474 at label level 0.45,
  from quantity 0 to 0.70, over 20 seeds), so the analysis uses realized label-HD.
- `rq3_edge.yaml` - the same at quantity 0.80 only (12 partitions),
  `partition_only`.

Every rq2.yaml and rq3.yaml cell runs FedAvg and FedProx (mu = 0.001) on the same
partition: 120 runs per grid. mu was tuned once on `mu_tuning.yaml` (below). Both
grids are open for training; RQ2 is trained first.

`mu_tuning.yaml` is the RQ3 cell at label-HD 0.45 x quantity-HD 0.35, at trials
3-4, which no reported grid trains: FedAvg and FedProx at mu in {0.001, 0.01,
0.1, 1, 10, 100} (the source paper's grid), 14 runs. The selection rule is in the
file and was committed before any run: highest mean final-round accuracy over the
two trials, FedAvg as mu = 0, the smallest mu within 0.01 of the top mean.

`rq2_class_order.yaml` is a diagnostic for RQ2, not an RQ2 arm: rq2.yaml's capped
FedAvg cells at label-HD 0.25, 0.45 and 0.70, trials 0, 1, 2, 5 and 6, with `class_order:
reversed` and the alpha each cell resolves written out in full. The cap concentrates
the classes it splits last, so this checks whether the capped arm's per-class
accuracy loss follows the class or its position in the split order. Level 0 is left
out, since the cap does not bind there. 15 partitions, 15 runs; the 0.45 runs keep the
ids they were trained under before the other levels were added. The rule the cells
are read by is in `analysis/rq2_class_order.py`; it was written for trials 0-2 and
applies only to those. Trials 5 and 6 are reported beside it, descriptively.

`centralized.yaml` is the centralized baseline (config/centralized.yaml) at trials
0, 1, 2, 5 and 6: `federated.algorithm: centralized` with `partition: null`, the whole training
set as one client, 10 epochs with one Adam (one round's local training, on the
pooled data), base.yaml's model, learning rate and batch size. The epoch count is
fixed in the config, not picked by test accuracy. Trial t has the federated runs'
init seed at trial t. history.csv holds one row per epoch, and rta counts epochs.

Covtype (K = 30, the DNN) has its own 20-trial curves: c517442ecf (natural_per_class
uncapped), ea8e56e2d6 (niid_bench), e73c794198 (natural_per_client) and a9a59164f5
(pure_label). Its 20-seed label-HD ceilings are 0.722, 0.732, 0.785 and 0.593:
with equal client sizes the two largest classes (85% of the data) fill most
clients, so pure_label tops out far lower than on CIFAR-10. The same level rule on
the RQ2 arms' common ceiling, 0.593, gives the Covtype levels 0, 0.15, 0.35 and
0.50. The cap binds even at alpha 1000 on Covtype, so the two natural arms differ
at level 0. The Covtype grids, trained three shards at once:

- `covtype_rq2.yaml` (open) - rq2.yaml's three arms at the four Covtype levels,
  trials 0-4: 12 cells, 60 partitions and 120 runs, FedAvg and FedProx (mu = 0.001)
  on each partition. The 60 FedAvg hashes are pinned by a test and did not move when
  the FedProx cases were added, once mu was tuned.
- `covtype_centralized.yaml` (open) - config/covtype_centralized.yaml at trials
  0-4, 10 epochs: the pre-specified baseline.
- `covtype_mu_tuning.yaml` (open since 2026-11-12, once the Covtype FedAvg grid and
  the CIFAR-10 trial extension were trained) - pure_label at label-HD 0.35, trials
  5-6, FedAvg and FedProx at the six mu values, with mu_tuning.yaml's rule: 14 runs.
  Trained; the rule picked mu = 0.001.
- `covtype_centralized_steps.yaml` (open since 2026-11-16, once the Covtype FedProx
  runs were trained; trained last) - the centralized
  baseline for 400 epochs, the optimizer steps of one 40 x 10 federated run, trials
  0-4. A post-hoc sensitivity check added after seeing the 10-epoch result; its
  `label` is written into every one of its runs.jsonl rows. Trained; its final
  accuracy is below the 10-epoch baseline's.
- `covtype_rq1.yaml` (partition-only) - rq1.yaml's three cases and ten alphas on
  Covtype at K = 30, trials 0-19: 600 partitions, one infeasible (natural uncapped,
  alpha 0.03, trial 5). Its natural per-class cells at alpha 1000, trials 0-4, are
  covtype_rq2.yaml's level-0 partitions.
- `covtype_quantity_only.yaml` (partition-only) - quantity_only at the 30 curve betas,
  K = 30, trials 0-19: 600 partitions, the partitions of the curve `b4684ba0de`.

A cell is recalibrated only if its 20-seed mean misses its target by more than
0.05. A single seed outside ±0.05 is listed and handled by the realized-HD
analysis; it is not swapped, redrawn or dropped. This applies to Covtype and to the
CIFAR-10 trials 5-6.

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
  (`completed` / `failed` / `oom`); readers take the last row per `run_id`.
  `final_acc` is the test accuracy after the last round, `best_acc` the highest
  after any round, and `rta` the first round reaching 90% of `best_acc` (the source
  paper's rounds-to-accuracy, taken against the run's own maximum).
  `final_balanced_acc` and `best_balanced_acc` are the same for balanced accuracy;
  runs recorded before per-class logging have neither. `label` is the grid's label,
  present only for runs of a grid that has one (e.g. a post-hoc check)
- `runs/{run_id}/history.csv` - per round: the global model's test accuracy and
  loss, `train_loss` (the trained clients' final-epoch losses weighted by sample
  count) and wall time, flushed each round. Round 0 is the initialised model
  before training; it has no train loss or wall time and no metric counts it
- `runs/{run_id}/clients.csv` - one row per client per round: its sample count,
  final-epoch training loss and optimizer steps
- `runs/{run_id}/per_class.csv` - per round, round 0 included: each class's test
  accuracy (`acc_class_{c}`, its recall) and `balanced_acc`, their unweighted mean
  (the thesis's addition; accuracy is the source paper's metric)
- `runs/{run_id}/confusion.csv` - per round, round 0 included: the global model's
  test-set confusion matrix, one row per (true class, predicted class) with its count
- `runs/{run_id}/config.yaml` - the resolved config, so a row stays readable
  after the schema moves on
- `analysis/` - analysis-script outputs, e.g. `fedartml_comparison_label_skew.csv`,
  `fedartml_comparison_quantity_skew.csv`, and `quantity_skew_*` tables and scatter
- `analysis/partition_validation_*` - `partitions.csv` (one row per partition
  and target pair), `clients.csv` (one row per client: size, class counts, HD to
  global, drift), `cells.csv` (one row per target cell with its verdict), and
  the figures `label.png` and `quantity.png` (nominal vs measured per axis) and
  `feasibility.png` (deviation from target per cell, failing cells crossed)
- `analysis/training_runs.csv` - one row per completed run with its partition's
  realized label-HD and quantity-HD; `analysis/training_summary.csv` - one row per
  condition (a config without its runtime section): mean and sample sd across
  trials of `final_acc`, `best_acc` and `rta`, mean realized HD, and
  `rta_of_mean_curve` (RTA of the trials' mean accuracy curve)
- `per_class/{run_id}.json` - a rerun's confusion matrix on the test set after every
  round (round 0 first), its accuracies, and each bit-exactness check against the
  recorded run
- `analysis/rq2_class_order_*` - `per_class.csv` (one row per level, arm, trial (0-2,
  5 and 6) and class: accuracy, holders, effective holders, zero-holders),
  `verdicts.csv` (one row per level: the rule on trials 0-2, then the trial 5-6
  overall differences, descriptive), `tables.md`, and `partitions.md` (concentration
  only)
- `analysis/numbers.tex`, `numbers.csv` - one `\newcommand` per number the results text
  quotes, and the same values with the output each comes from
- `analysis/paper/` - `tables/` (each table as CSV at full precision with n and the trial
  set, a plain LaTeX tabular, Markdown, and its caption notes), `figures/` (PDF, PNG,
  a CSV of exactly the plotted numbers, notes with a proposed caption; `figures/single/`
  holds each panel of the multi-panel figures on its own), `reproduction_check.txt`,
  `checks.txt`, `inputs_manifest.csv` (every run and partition read, with its commit),
  the comparison of the two CIFAR-10 trial sets, and `trials_0_1_2/` (the same outputs on
  CIFAR-10 trials 0-2)
- `analysis/index.md` - every file of the paper outputs, with the command and commit that
  wrote it

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

`b4684ba0de` is the Covtype K = 30 beta curve (20 trials, floor 22): quantity-HD
0.011-0.879. The paper outputs read the beta of deliberate quantity skew equal to
natural label skew's incidental quantity-HD off it, as they read 8f995f9546 for
CIFAR-10.

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

Adapted code (MIT and Apache-2.0; full licence texts in `LICENSES/`):

- `partitioning/fedartml_compat.py` `pairwise_hellinger` - from FedArtML 0.1.34
  `hellinger_distance` (Copyright (c) 2022 La-Sapienza-University-Rome). Used
  only to translate to FedArtML's pairwise scale, never as a grid axis.
- `partitioning/natural_methods.py` `_split_by_proportions` and the per-class loop of
  `natural_per_class` - from FedArtML 0.1.34
  `SplitAsFederatedData.dirichlet_method`, itself derived from NIID-Bench
  (Copyright (c) 2021 Yiqun Diao, Qinbin Li; Copyright (c) 2020 International
  Business Machines).
- `partitioning/quantity_only_method.py` `draw_client_sizes` and the permutation + split of
  `quantity_only` - from FedArtML 0.1.34
  `SplitAsFederatedData.dirichlet_method_quant_skew(method="minsize-dirichlet")`,
  whose permutation + Dir(beta) + cumsum split comes from NIID-Bench's
  `iid-diff-quantity` partition. This replicates the source paper's
  quantity-only condition; it is not a thesis contribution, and the nested
  `combined` partitioner imports it rather than copying it.
- `federated/aggregation.py` `fedavg` - from Flower 1.36.0
  `aggregate_arrayrecords` (Apache-2.0, Copyright 2025 Flower Labs GmbH), the
  weighted average behind Flower's FedAvg strategy, ported to torch state dicts.
  It gives bit-identical results to Flower's function; Flower itself is not
  imported.
- `federated/proximal.py` `add_proximal_gradient` - from the FedProx authors'
  own code, `PerturbedGradientDescent` in github.com/litian96/FedProx (MIT,
  Copyright (c) 2019 Tian Li): the perturbed gradient grad + mu * (w - w_global),
  kept as the gradient that Adam steps on, in torch instead of TensorFlow.

Each adapted function carries a header with source, version, licence and
changes.

AI assistance: parts of this code base were written with Claude Code
(Anthropic) from specifications written by the author, and reviewed and tested
by the author. Modules produced this way say so in their docstring.
`partitioning/pure_label_method.py` and `partitioning/two_stage_method.py` were
written by the author; their docstrings list the changes made to them with
Claude Code.

The thesis's own contributions are the author's own design: the HD-to-global
label and quantity metrics (`partitioning/metrics.py`), the
`natural_per_client` and `pure_label` constructions, the alpha -> label-HD and
beta -> quantity-HD calibration (`partitioning/calibration.py`), and the
two-stage nested partitioning method (`partitioning/two_stage_method.py`).
