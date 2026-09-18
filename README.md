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
  - dirichlet.py - per-class Dirichlet partitioning (`natural_per_class`)
- models/ - CNN and DNN architectures
- federated/ - FedAvg, FedProx, training loop
- experiments/ - config schema, seeding, recording, runner scripts
  - config.py - frozen dataclasses, YAML load/dump, `config_hash`
  - seeding.py - seed derivation per trial, torch determinism
  - recording.py - partition store, run records, atomic writes, provenance
  - datasets.py - dataset label registry
  - run_partition.py - generate and record one partition
- config/ - base.yaml, smoke.yaml, grids/ (one per research question)
- tests/ - verification suite
- analysis/ - result analysis and figure-generation scripts

## Usage

Run the test suite:

    python -m pytest tests/ -q

Generate a partition (cached by `partition_id`; re-running is a no-op):

    python -m experiments.run_partition --config config/smoke.yaml

Check the environment and estimate the compute budget:

    python tests/env_report.py
    python tests/gpu_check.py

## Results layout

Everything lands under `results/` (gitignored), written atomically:

- `partitions/{partition_id}.npz` - client index arrays
- `partitions/{partition_id}.json` - realized metrics and provenance
- `partitions.jsonl` - one row per partition; this is the RQ1 dataset on its own
- `runs.jsonl` - one row per run at start (`running`) and at the end
  (`completed` / `failed` / `oom`); readers take the last row per `run_id`
- `runs/{run_id}/history.csv` - per-round accuracy and loss, flushed each round
- `runs/{run_id}/config.yaml` - the resolved config, so a row stays readable
  after the schema moves on

Partitions are addressed by the fields that determine the split, so training
knobs never invalidate a cached partition, and FedAvg and FedProx at the same
grid cell provably read identical splits. Reruns are byte-identical.

## Conventions

- Randomness: one `trial` index derives independent partition/init/train seeds.
  Partitioners take an explicit `np.random.Generator`; global `np.random` is
  never touched.
- Determinism is strict by default (`runtime.deterministic`), costing ~16% per
  step on the RTX 3070 Ti. Every run row records which mode produced it.
- Label-HD is measured against the empirical global label distribution, never
  against uniform - CIFAR-10 is class-balanced, Covtype (RQ4) is not.
  Quantity-HD is measured against uniform 1/K.
- Class counts come from the dataset registry; nothing assumes 10 classes.

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
  - The Dirichlet partitioner deliberately omits NIID-Bench's balancing term,
    which FedArtML inherits and which clamps client sizes at N/K; B2 should
    quantify that gap on a shared seed
- RQ1 grid written (63 partitions, no GPU needed); RQ2/RQ3 grids stubbed
  behind `status: blocked` pending the B6 calibration table
- Next: B2 (per-client Dirichlet variant, FedArtML comparison)
