"""Environment report for the thesis README / Experimental Setup (F2).

Run from the project root with the venv active:
    python tests\\env_report.py
"""

import importlib
import importlib.metadata as md
import platform
import subprocess
import sys

PACKAGES = [
    "torch", "torchvision", "numpy", "scipy", "pandas",
    "matplotlib", "seaborn", "scikit-learn", "flwr", "fedartml",
    "pyyaml", "tqdm",
]


def line(label, value):
    print(f"{label:<22} {value}")


def section(name):
    print(f"\n--- {name} ---")


def shell(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, shell=True, timeout=30)
        return out.stdout.strip() or out.stderr.strip() or "(no output)"
    except Exception as e:
        return f"(failed: {e})"


section("Platform")
line("OS", f"{platform.system()} {platform.release()} ({platform.version()})")
line("Machine", platform.machine())
line("CPU", platform.processor())
line("Python", sys.version.replace("\n", " "))
line("Executable", sys.executable)

section("Packages")
for name in PACKAGES:
    try:
        line(name, md.version(name))
    except md.PackageNotFoundError:
        line(name, "NOT INSTALLED")

section("PyTorch / CUDA")
try:
    import torch
    line("torch", torch.__version__)
    line("Built for CUDA", torch.version.cuda)
    line("cuDNN", torch.backends.cudnn.version())
    line("CUDA available", torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        line("GPU", props.name)
        line("VRAM (GB)", round(props.total_memory / 1e9, 2))
        line("Compute capability", f"{props.major}.{props.minor}")
        line("Driver CUDA (max)", shell(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader"))
except ImportError:
    line("torch", "NOT INSTALLED")

section("Git")
line("Commit", shell("git rev-parse --short HEAD"))
line("Branch", shell("git rev-parse --abbrev-ref HEAD"))
line("Remote", shell("git remote get-url origin"))

section("FedArtML location")
try:
    import fedartml, os
    line("Path", os.path.dirname(fedartml.__file__))
except ImportError:
    line("Path", "NOT INSTALLED")