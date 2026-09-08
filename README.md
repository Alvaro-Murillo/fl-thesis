# Federated Learning: Label and Quantity Skew Interaction

Bachelor thesis project quantifying and disentangling label and quantity skew
in Federated Learning.

## Setup

conda env create -f environment.yml
conda activate fl-thesis

## Structure

- partitioning/ - HD measurement, Dirichlet partitioning, two-stage nested allocator
- models/ - CNN and DNN architectures
- federated/ - FedAvg, FedProx, training loop
- tests/ - verification suite
- experiments/ - experiment runner scripts
- analysis/ - result analysis and figure-generation scripts
