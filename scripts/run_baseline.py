"""
Run clean FedAvg baseline — no attacks, no defenses.
Saves results to experiments/results/baseline_mnist.csv
Saves plot to report/figures/fig1_baseline.png
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torchvision import datasets, transforms
from src.fl_core.server import run_simulation

# ── Dataset ────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))   # MNIST mean/std
])

train_ds = datasets.MNIST("./data/raw", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST("./data/raw", train=False, download=True, transform=transform)

# ── Run clean simulation ────────────────────────────────
results = run_simulation(
    train_dataset      = train_ds,
    test_dataset       = test_ds,
    num_clients        = 10,
    num_rounds         = 20,
    fraction_fit       = 1.0,
    local_epochs       = 2,
    batch_size         = 32,
    lr                 = 0.01,
    alpha              = 0.5,        # non-IID Dirichlet
    malicious_fraction = 0.0,        # 0% malicious = clean run
    attack_fn          = None,
    dataset_name       = "mnist",
    results_path       = "experiments/results/baseline_mnist.csv",
    seed               = 42,
)

# ── Plot Figure 1 ───────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # no display needed — saves to file

rounds    = [r["round"]    for r in results]
accuracies = [r["accuracy"] * 100 for r in results]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(rounds, accuracies, color="#1a6b3a", linewidth=2.5, marker="o", markersize=4, label="FedAvg (clean)")
ax.axhline(y=max(accuracies), color="#c0392b", linestyle="--", linewidth=1, alpha=0.6,
           label=f"Peak: {max(accuracies):.2f}%")

ax.set_xlabel("Communication Round", fontsize=12)
ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title("FedAvg Baseline — Clean Training\n(10 clients, MNIST non-IID α=0.5, 20 rounds)", fontsize=13)
ax.set_ylim(0, 100)
ax.set_xlim(0, 21)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

os.makedirs("report/figures", exist_ok=True)
plt.tight_layout()
plt.savefig("report/figures/fig1_baseline.png", dpi=150, bbox_inches="tight")
print(f"✓ Figure saved → report/figures/fig1_baseline.png")
print(f"✓ Final baseline accuracy: {accuracies[-1]:.2f}%")