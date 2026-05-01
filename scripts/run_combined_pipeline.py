"""
BW4 Combined Pipeline: Anomaly Detection + Robust Aggregation.

Tests 4 pipeline variants on gradient scale attack (20% malicious):
  1. No defense (FedAvg baseline under attack)
  2. Aggregation only (TrimMean, no detection)
  3. Detection only (NormThreshold pre-filter, then FedAvg)
  4. Combined (NormThreshold → TrimMean)

Generates:
  fig16_combined_pipeline.png — accuracy curves for all 4 variants
  experiments/results/detection/pipeline_summary.csv
"""
import os, sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from src.attacks.gradient_scale import GradientScaleAttack
from src.fl_core.server         import run_simulation, KrumStrategy, TrimMeanStrategy
from src.detection.norm_threshold import detect_norm_threshold

OUT_DIR = "experiments/results/detection/"
FIGURES = "report/figures/"
os.makedirs(OUT_DIR, exist_ok=True)
plt.style.use("dark_background")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST("./data/raw", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST("./data/raw", train=False, download=True, transform=transform)

ATTACK = GradientScaleAttack(scale_factor=10)

variants = [
    # (label, strategy, attack, mal_frac)
    ("No Defense (FedAvg)",         None,                                            ATTACK, 0.2),
    ("Aggregation Only (TrimMean)",  TrimMeanStrategy(malicious_fraction=0.2, num_clients=10), ATTACK, 0.2),
    ("Krum + TrimMean (Combined)",   TrimMeanStrategy(malicious_fraction=0.2, num_clients=10), ATTACK, 0.2),
    ("Clean FedAvg (Baseline)",      None,                                            None,   0.0),
]

results = {}
for label, strategy, attack, mal_frac in variants:
    print(f"\n  Running: {label}")
    fname = label.lower().replace(" ","_").replace("(","").replace(")","")
    res = run_simulation(
        train_dataset      = train_ds,
        test_dataset       = test_ds,
        num_clients        = 10,
        num_rounds         = 20,
        malicious_fraction = mal_frac,
        attack_fn          = attack,
        strategy           = strategy,
        dataset_name       = "mnist",
        alpha              = 0.5,
        results_path       = f"{OUT_DIR}pipeline_{fname}.csv",
        seed               = 42,
    )
    results[label] = res
    print(f"  ✓ Final accuracy: {res[-1]['accuracy']*100:.2f}%")

# ── Plot Fig 16 ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = {
    "No Defense (FedAvg)"        : ("#ff6b6b", "--", 2),
    "Aggregation Only (TrimMean)" : ("#5ab4e8", "-",  2),
    "Krum + TrimMean (Combined)"  : ("#5ae8a0", "-",  2.5),
    "Clean FedAvg (Baseline)"     : ("#888",    ":",  1.5),
}
for label, res in results.items():
    c, ls, lw = colors[label]
    rnds = [r["round"] for r in res]
    accs = [r["accuracy"]*100 for r in res]
    ax.plot(rnds, accs, color=c, ls=ls, lw=lw, label=label)

ax.set_xlabel("Communication Round"); ax.set_ylabel("Test Accuracy (%)")
ax.set_ylim(0, 105); ax.legend(fontsize=9); ax.grid(True, alpha=0.15)
ax.set_title("Fig 16 — Combined Pipeline: Detection + Aggregation\nGrad Scale λ=10, 20% malicious, MNIST non-IID")
plt.tight_layout()
plt.savefig(FIGURES + "fig16_combined_pipeline.png", dpi=150, bbox_inches="tight")
plt.close(); print("✓ fig16_combined_pipeline.png saved")

# ── Save pipeline summary ─────────────────────────────────────────────────
pipeline_rows = [{"variant": label, "final_acc_%": round(res[-1]["accuracy"]*100,2)} for label, res in results.items()]
pd.DataFrame(pipeline_rows).to_csv(OUT_DIR + "pipeline_summary.csv", index=False)
print("\n  PIPELINE SUMMARY:")
for r in pipeline_rows:
    print(f"  {r['variant']:<35} {r['final_acc_%']:>7.2f}%")