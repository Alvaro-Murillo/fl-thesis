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

## Structure

- partitioning/ - HD measurement, Dirichlet partitioning, two-stage nested allocator
- models/ - CNN and DNN architectures
- federated/ - FedAvg, FedProx, training loop
- tests/ - verification suite
- experiments/ - experiment runner scripts
- analysis/ - result analysis and figure-generation scripts

## Progress log

- Repo scaffolding created: folder structure, .gitignore, README
- Conda: version 25.5.1 confirmed installed
- fl-thesis conda environment created then removed (was being set up on
  laptop by mistake - full environment to be built on main desktop machine
  with RTX 3070 Ti instead)
- Virtual environment (Python 3.12.10) set up on main desktop machine (RTX
  3070 Ti); dependencies frozen to requirements.txt (PyTorch 2.14.0+cu126,
  Flower 1.36.0, FedArtML 0.1.34, etc.)
- No code written yet
