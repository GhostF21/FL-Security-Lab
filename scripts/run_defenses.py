"""
Run defense comparison experiments for Bi-Weekly Report 3.

Extends the BW2 Krum-only script by adding Trimmed Mean as a third defense.

Grid:
  Attacks     : label_flip (10/20/30%), gradient_scale λ=10 (10/20/30%)
  Defenses    : FedAvg (no defense), Multi-Krum, Trimmed Mean
  Baseline    : clean FedAvg (no attack) — loaded from BW1/BW2 CSV

Output CSVs  → experiments/results/   (one per experiment)
Summary CSV  → experiments/results/defense_summary_bw3.csv
"""
import csv
import sys
import os
sys.path.insert(0, ".")

from torchvision import datasets, transforms

from src.attacks.label_flip     import LabelFlipAttack
from src.attacks.gradient_scale import GradientScaleAttack
from src.fl_core.server         import run_simulation, KrumStrategy, TrimMeanStrategy

# ── Dataset setup (same as BW1/BW2) ────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST("./data/raw", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST("./data/raw", train=False, download=True, transform=transform)

RESULTS      = "experiments/results/"
NUM_CLIENTS  = 10
NUM_ROUNDS   = 20
BASELINE_ACC = 99.15   # your actual clean FedAvg baseline from BW1

os.makedirs(RESULTS, exist_ok=True)

# ── Experiment grid (same attacks/fractions as BW2) ─────────────
# (attack_name, attack_obj, malicious_fraction, f_for_krum)
experiments = [
    ("label_flip",    LabelFlipAttack(mode="targeted", source_class=3, target_class=7), 0.1, 1),
    ("label_flip",    LabelFlipAttack(mode="targeted", source_class=3, target_class=7), 0.2, 2),
    ("label_flip",    LabelFlipAttack(mode="targeted", source_class=3, target_class=7), 0.3, 3),
    ("grad_scale_10", GradientScaleAttack(scale_factor=10),                             0.1, 1),
    ("grad_scale_10", GradientScaleAttack(scale_factor=10),                             0.2, 2),
    ("grad_scale_10", GradientScaleAttack(scale_factor=10),                             0.3, 3),
]

rows = []

for attack_name, attack, mal_frac, f_assumed in experiments:

    frac_pct = int(mal_frac * 100)   # e.g. 0.2 → 20

    print(f"\n{'─'*60}")
    print(f"  {attack_name} | mal_frac={mal_frac:.0%} | f={f_assumed}")
    print(f"{'─'*60}")

    # ── 1. FedAvg under attack (no defense) ───────────────────────
    print("  [1/3] Running: FedAvg (no defense)...")
    res_fedavg = run_simulation(
        train_dataset      = train_ds,
        test_dataset       = test_ds,
        num_clients        = NUM_CLIENTS,
        num_rounds         = NUM_ROUNDS,
        malicious_fraction = mal_frac,
        attack_fn          = attack,
        strategy           = None,
        dataset_name       = "mnist",
        alpha              = 0.5,
        results_path       = f"{RESULTS}{attack_name}_fedavg_f{frac_pct}.csv",
        seed               = 42,
    )
    fedavg_acc = res_fedavg[-1]["accuracy"] * 100

    # ── 2. Multi-Krum defense (same as BW2) ───────────────────────
    print(f"  [2/3] Running: Multi-Krum (f={f_assumed})...")
    res_krum = run_simulation(
        train_dataset      = train_ds,
        test_dataset       = test_ds,
        num_clients        = NUM_CLIENTS,
        num_rounds         = NUM_ROUNDS,
        malicious_fraction = mal_frac,
        attack_fn          = attack,
        strategy           = KrumStrategy(f=f_assumed, use_multi_krum=True),
        dataset_name       = "mnist",
        alpha              = 0.5,
        results_path       = f"{RESULTS}{attack_name}_krum_f{frac_pct}.csv",
        seed               = 42,
    )
    krum_acc = res_krum[-1]["accuracy"] * 100

    # ── 3. Trimmed Mean defense (new for BW3) ─────────────────────
    print(f"  [3/3] Running: Trimmed Mean (mal_frac={mal_frac:.0%}, β={f_assumed})...")
    res_trim = run_simulation(
        train_dataset      = train_ds,
        test_dataset       = test_ds,
        num_clients        = NUM_CLIENTS,
        num_rounds         = NUM_ROUNDS,
        malicious_fraction = mal_frac,
        attack_fn          = attack,
        strategy           = TrimMeanStrategy(
                                 malicious_fraction = mal_frac,
                                 num_clients        = NUM_CLIENTS,
                             ),
        dataset_name       = "mnist",
        alpha              = 0.5,
        results_path       = f"{RESULTS}{attack_name}_trimmean_f{frac_pct}.csv",
        seed               = 42,
    )
    trim_acc = res_trim[-1]["accuracy"] * 100

    # ── Recovery % for both defenses (relative to clean baseline) ─
    def recovery(defended_acc, attacked_acc, baseline=BASELINE_ACC):
        """How much of the accuracy gap did the defense recover?"""
        gap = baseline - attacked_acc + 1e-6
        return round((defended_acc - attacked_acc) / gap * 100, 1)

    krum_recovery = recovery(krum_acc, fedavg_acc)
    trim_recovery = recovery(trim_acc, fedavg_acc)

    row = {
        "attack"           : attack_name,
        "mal_fraction"     : mal_frac,
        "f_assumed"        : f_assumed,
        "fedavg_acc_%"     : round(fedavg_acc, 2),
        "krum_acc_%"       : round(krum_acc,   2),
        "trimmean_acc_%"   : round(trim_acc,    2),
        "krum_recovery_%"  : krum_recovery,
        "trim_recovery_%"  : trim_recovery,
    }
    rows.append(row)

    print(f"\n  ✓ FedAvg (attacked)  : {fedavg_acc:.2f}%")
    print(f"  ✓ Krum defended      : {krum_acc:.2f}%   (recovery {krum_recovery:.1f}%)")
    print(f"  ✓ TrimMean defended  : {trim_acc:.2f}%   (recovery {trim_recovery:.1f}%)")

# ── Save BW3 defense summary CSV ────────────────────────────────
fieldnames = [
    "attack", "mal_fraction", "f_assumed",
    "fedavg_acc_%", "krum_acc_%", "trimmean_acc_%",
    "krum_recovery_%", "trim_recovery_%",
]

summary_path = RESULTS + "defense_summary_bw3.csv"
with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

# ── Print summary table ──────────────────────────────────────────
print(f"\n{'='*75}")
print("  BW3 DEFENSE RESULTS SUMMARY")
print(f"{'='*75}")
print(f"  {'Attack':<15} {'Frac':>5} {'f':>3} │ "
      f"{'FedAvg':>8} │ {'Krum':>8} {'Rcvry':>7} │ {'TrimMean':>9} {'Rcvry':>7}")
print(f"  {'─'*75}")
for r in rows:
    print(
        f"  {r['attack']:<15} {r['mal_fraction']:>4.0%} {r['f_assumed']:>3} │ "
        f"{r['fedavg_acc_%']:>7.2f}% │ "
        f"{r['krum_acc_%']:>7.2f}% {r['krum_recovery_%']:>6.1f}% │ "
        f"{r['trimmean_acc_%']:>8.2f}% {r['trim_recovery_%']:>6.1f}%"
    )

print(f"\n  ✓ defense_summary_bw3.csv saved → {summary_path}")
print(f"  ✓ Individual round CSVs saved  → {RESULTS}")