"""
Run all attack experiments for Bi-Weekly Report 1.
Generates:
    experiments/results/attacks_summary.csv
    report/figures/fig2_attack_accuracy.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torchvision import datasets, transforms

from src.fl_core.server import run_simulation
from src.attacks.label_flip import LabelFlipAttack
from src.attacks.backdoor import BackdoorAttack, compute_asr
from src.fl_core.model import get_model

# ── Dataset setup ──────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST("./data/raw", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST("./data/raw", train=False, download=True, transform=transform)

BASELINE_ACC = None  # will be set from baseline CSV if available

# Load baseline accuracy
try:
    baseline_df = pd.read_csv("experiments/results/baseline_mnist.csv")
    BASELINE_ACC = baseline_df["accuracy"].iloc[-1]
    print(f"  Loaded baseline accuracy: {BASELINE_ACC*100:.2f}%")
except FileNotFoundError:
    print("  ⚠ Baseline CSV not found — run scripts/run_baseline.py first!")
    exit(1)

# ── Experiment configuration ────────────────────────────────────────────────
MALICIOUS_FRACTIONS = [0.1, 0.2, 0.3]   # 10%, 20%, 30%
NUM_ROUNDS          = 20
NUM_CLIENTS         = 10

attack_configs = [
    {
        "name"      : "Label-Flip (3→8)",
        "attack_fn" : LabelFlipAttack(mode="targeted", source_class=3, target_class=8),
        "type"      : "label_flip",
    },
    {
        "name"      : "Backdoor (trigger→0)",
        "attack_fn" : BackdoorAttack(target_class=0, trigger_size=4),
        "type"      : "backdoor",
    },
]

# ── Run all experiments ─────────────────────────────────────────────────────
all_results   = []
all_histories = {}   # { exp_name: [{"round": r, "accuracy": a}, ...] }

for config in attack_configs:
    for frac in MALICIOUS_FRACTIONS:
        exp_name     = f"{config['type']}_f{int(frac*100)}"
        results_path = f"experiments/results/{exp_name}.csv"

        print(f"\n{'─'*55}")
        print(f"  Experiment: {config['name']} | {frac:.0%} malicious")
        print(f"{'─'*55}")

        t_start = time.time()
        round_results = run_simulation(
            train_dataset      = train_ds,
            test_dataset       = test_ds,
            num_clients        = NUM_CLIENTS,
            num_rounds         = NUM_ROUNDS,
            malicious_fraction = frac,
            attack_fn          = config["attack_fn"],
            dataset_name       = "mnist",
            results_path       = results_path,
            seed               = 42,
        )
        elapsed = time.time() - t_start

        final_acc = round_results[-1]["accuracy"]
        acc_drop  = (BASELINE_ACC - final_acc) * 100

        all_histories[exp_name] = round_results

        # Compute ASR for backdoor attacks
        asr = None
        if config["type"] == "backdoor":
            print(f"  [ASR] Computing attack success rate on triggered test set...")
            model     = get_model(dataset="mnist")
            asr_value = compute_asr(
                model         = model,
                test_dataset  = test_ds,
                target_class  = config["attack_fn"].target_class,
                trigger_size  = config["attack_fn"].trigger_size,
                trigger_value = config["attack_fn"].trigger_value,
                device        = "cpu",
            )
            asr = f"{asr_value*100:.2f}"
            print(f"  [ASR] Attack Success Rate: {asr}%")

        summary = {
            "attack"            : config["name"],
            "malicious_fraction": f"{frac:.0%}",
            "final_accuracy_%"  : f"{final_acc*100:.2f}",
            "accuracy_drop_%"   : f"{acc_drop:.2f}",
            "asr"               : asr if asr else "N/A",
            "rounds"            : NUM_ROUNDS,
            "elapsed_min"       : f"{elapsed/60:.1f}",
        }
        all_results.append(summary)
        print(f"\n  ✓ Result: Acc={final_acc*100:.2f}% | Drop={acc_drop:.2f}%")

# ── Save summary CSV ───────────────────────────────────────────────────────
os.makedirs("experiments/results", exist_ok=True)
summary_path = "experiments/results/attacks_summary.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)
print(f"  ✓ Summary saved → {summary_path}")

# ── Generate PNG plot ──────────────────────────────────────────────────────
COLORS = ["#e41a1c", "#ff7f00", "#984ea3",   # label_flip: red, orange, purple
          "#377eb8", "#4daf4a", "#a65628"]    # backdoor  : blue, green, brown

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle("FL Attack Experiments — MNIST", fontsize=14, fontweight="bold")

attack_types = [("label_flip", "Label-Flip (3→8)"),
                ("backdoor",   "Backdoor (trigger→0)")]

for ax, (atype, atitle) in zip(axes, attack_types):
    # Baseline reference line
    ax.axhline(BASELINE_ACC * 100, color="black", linestyle="--",
               linewidth=1.4, label=f"Baseline ({BASELINE_ACC*100:.1f}%)")

    for i, frac in enumerate(MALICIOUS_FRACTIONS):
        exp_name = f"{atype}_f{int(frac*100)}"
        history  = all_histories.get(exp_name, [])
        if not history:
            continue
        rounds = [r["round"] for r in history]
        accs   = [r["accuracy"] * 100 for r in history]
        color  = COLORS[i] if atype == "label_flip" else COLORS[i + 3]
        ax.plot(rounds, accs, marker="o", markersize=3,
                linewidth=1.8, color=color, label=f"{frac:.0%} malicious")

    ax.set_title(atitle, fontsize=12)
    ax.set_xlabel("Round", fontsize=10)
    ax.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_xlim(1, NUM_ROUNDS)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
os.makedirs("report/figures", exist_ok=True)
plot_path = "report/figures/fig2_attack_accuracy.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Plot saved    → {plot_path}")

# ── Print table ───────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  ATTACK RESULTS SUMMARY")
print(f"{'='*70}")
print(f"  {'Attack':<30} {'Fraction':>10} {'Final Acc':>10} {'Drop':>8} {'ASR':>8}")
print(f"  {'─'*65}")
for r in all_results:
    print(f"  {r['attack']:<30} {r['malicious_fraction']:>10} "
          f"{r['final_accuracy_%']:>9}% {r['accuracy_drop_%']:>7}% {r['asr']:>7}")
print(f"\n  Baseline accuracy: {BASELINE_ACC*100:.2f}%")