"""GPU sanity benchmark and compute-budget estimate for the FL thesis.

This is not a correctness test of any thesis component. It checks that the GPU
is usable and gives a first-order estimate of whether the planned experiment
grid (~54-90 runs) fits the local compute budget.

Generated with the help of Claude Code.
"""

import time

import torch
import torch.nn as nn

NUM_CLIENTS = 30
NUM_ROUNDS = 40
LOCAL_EPOCHS = 10
BATCH_SIZE = 64
TRAIN_SIZE = 50_000
GRID_RUNS = 60
WARMUP_STEPS = 20
TIMED_STEPS = 100


class CNN(nn.Module):
    """32x32x3 -> 10 classes. Matches the thesis architecture."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3), nn.ReLU(), nn.Flatten(),
            nn.LazyLinear(64), nn.ReLU(), nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    print(f"torch            {torch.__version__}")
    print(f"CUDA build       {torch.version.cuda}")
    print(f"Device           {device}")
    if cuda:
        p = torch.cuda.get_device_properties(0)
        print(f"GPU              {p.name}")
        print(f"VRAM             {p.total_memory / 1e9:.2f} GB")
        print(f"Compute cap.     {p.major}.{p.minor}")
    else:
        print("WARNING: CUDA unavailable - timings below are CPU-only and not "
              "representative of the intended setup.")

    model = CNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(BATCH_SIZE, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (BATCH_SIZE,), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        opt.step()

    for _ in range(WARMUP_STEPS):  # covers CUDA context init + LazyLinear shape inference
        step()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(TIMED_STEPS):
        step()
    if cuda:
        torch.cuda.synchronize()
    ms_per_step = (time.perf_counter() - t0) / TIMED_STEPS * 1000

    print(f"\nms/step          {ms_per_step:.2f}")
    if cuda:
        print(f"Peak VRAM        {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    # The 50k train set is split across the K clients, so one round processes
    # ~50k samples in total per local epoch regardless of client count.
    steps_per_run = (TRAIN_SIZE / BATCH_SIZE) * LOCAL_EPOCHS * NUM_ROUNDS
    minutes = steps_per_run * ms_per_step / 1000 / 60

    print(f"\nOne FL run (K={NUM_CLIENTS}, R={NUM_ROUNDS}, E={LOCAL_EPOCHS}): "
          f"{minutes:.1f} min ({steps_per_run:,.0f} steps)")
    print(f"{GRID_RUNS} runs: {minutes * GRID_RUNS / 60:.1f} h")
    print("FLOOR, not a forecast: excludes data loading, per-round evaluation, "
          "and FL orchestration overhead.")


if __name__ == "__main__":
    main()
