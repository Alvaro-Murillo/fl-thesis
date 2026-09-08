# Federated Learning: Label and Quantity Skew Interaction

Bachelor thesis project quantifying and disentangling label and quantity skew
in Federated Learning - measuring the quantity-skew byproduct of standard
Dirichlet label partitioning and testing deliberate label-quantity interaction
effects via a two-stage nested partitioning method.

## Setup

conda env create -f environment.yml
conda activate fl-thesis

Note: environment.yml is not yet committed. The full environment (PyTorch,
Flower, FedArtML, etc.) is being set up on the main development machine
(desktop with RTX 3070 Ti), not on this laptop, so that the exported
environment reflects a GPU-verified working setup.

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
- Environment setup (PyTorch + CUDA, Flower, FedArtML) pending
- No code written yet
