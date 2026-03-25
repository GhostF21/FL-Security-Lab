"""
Run Krum defense experiments across:
  - f = 1, 2, 3  (10%, 20%, 30% malicious)
  - attack types: label_flip, gradient_scale_lam10
  Saves results to experiments/results/defense_summary.csv
"""
import csv
import sys
import os
sys.path.insert(0, ".")

from torchvision import datasets, transforms

from src.attacks.label_flip     import LabelFlipAttack
from src.attacks.gradient_scale import GradientScaleAttack
from src.fl_core.server         import run_simulation, KrumStrategy

# ── Dataset setup (same as run_attacks.py) ─────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST("./data/raw", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST("./data/raw", train=False, download=True, transform=transform)

RESULTS      = "experiments/results/"
NUM_CLIENTS  = 10
NUM_ROUNDS   = 20
BASELINE_ACC = 99.15  # your actual baseline from BW1 results

os.makedirs(RESULTS, exist_ok=True)

# ── Experiments ─────────────────────────────────────────────────
experiments = [
    # (attack_name, attack_obj, malicious_fraction, f_assumed)
    ("label_flip",    LabelFlipAttack(mode="targeted", source_class=3, target_class=7), 0.1, 1),
    ("label_flip",    LabelFlipAttack(mode="targeted", source_class=3, target_class=7), 0.2, 2),
    ("label_flip",    LabelFlipAttack(mode="targeted", source_class=3, target_class=7), 0.3, 3),
    ("grad_scale_10", GradientScaleAttack(scale_factor=10),                             0.1, 1),
    ("grad_scale_10", GradientScaleAttack(scale_factor=10),                             0.2, 2),
    ("grad_scale_10", GradientScaleAttack(scale_factor=10),                             0.3, 3),
]

rows = []

for attack_name, attack, mal_frac, f_assumed in experiments:

    print(f"\n{'─'*55}")
    print(f"  {attack_name} | mal_frac={mal_frac:.0%} | f={f_assumed}")
    print(f"{'─'*55}")

    # ── FedAvg under attack (no defense) ──────────────────
    print("  Running: FedAvg (no defense)...")
    res_undefended = run_simulation(
        train_dataset      = train_ds,
        test_dataset       = test_ds,
        num_clients        = NUM_CLIENTS,
        num_rounds         = NUM_ROUNDS,
        malicious_fraction = mal_frac,
        attack_fn          = attack,
        strategy           = None,           # plain FedAvg
        dataset_name       = "mnist",
        alpha              = 0.5,
        results_path       = f"{RESULTS}{attack_name}_fedavg_f{int(mal_frac*100)}.csv",
        seed               = 42,
    )
    fedavg_acc = res_undefended[-1]["accuracy"] * 100   # last round, convert to %

    # ── Multi-Krum defense ─────────────────────────────────
    print(f"  Running: Multi-Krum (f={f_assumed})...")
    krum_strategy = KrumStrategy(f=f_assumed, use_multi_krum=True)
    res_krum = run_simulation(
        train_dataset      = train_ds,
        test_dataset       = test_ds,
        num_clients        = NUM_CLIENTS,
        num_rounds         = NUM_ROUNDS,
        malicious_fraction = mal_frac,
        attack_fn          = attack,
        strategy           = krum_strategy,  # Krum defense
        dataset_name       = "mnist",
        alpha              = 0.5,
        results_path       = f"{RESULTS}{attack_name}_krum_f{int(mal_frac*100)}.csv",
        seed               = 42,
    )
    krum_acc = res_krum[-1]["accuracy"] * 100            # last round, convert to %

    # ── Recovery % ────────────────────────────────────────
    recovery_pct = (
        (krum_acc - fedavg_acc)
        / (BASELINE_ACC - fedavg_acc + 1e-6)
        * 100
    )

    row = {
        "attack"        : attack_name,
        "mal_fraction"  : mal_frac,
        "f_assumed"     : f_assumed,
        "fedavg_acc_%"  : round(fedavg_acc, 2),
        "krum_acc_%"    : round(krum_acc, 2),
        "recovery_pct"  : round(recovery_pct, 1),
    }
    rows.append(row)

    print(f"\n  ✓ FedAvg attacked : {fedavg_acc:.2f}%")
    print(f"  ✓ Krum defended   : {krum_acc:.2f}%")
    print(f"  ✓ Recovery        : {recovery_pct:.1f}%")

# ── Save defense summary CSV ─────────────────────────────────────
fieldnames = ["attack", "mal_fraction", "f_assumed",
              "fedavg_acc_%", "krum_acc_%", "recovery_pct"]

summary_path = RESULTS + "defense_summary.csv"
with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"\n{'='*55}")
print("  DEFENSE RESULTS SUMMARY")
print(f"{'='*55}")
print(f"  {'Attack':<15} {'Frac':>6} {'f':>3} {'FedAvg':>9} {'Krum':>9} {'Recovery':>10}")
print(f"  {'─'*55}")
for r in rows:
    print(f"  {r['attack']:<15} {r['mal_fraction']:>5.0%} {r['f_assumed']:>3} "
          f"{r['fedavg_acc_%']:>8.2f}% {r['krum_acc_%']:>8.2f}% {r['recovery_pct']:>9.1f}%")

print(f"\n  ✓ defense_summary.csv saved → {summary_path}")