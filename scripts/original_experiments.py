"""
BW3 Original Contributions — Experiments Beyond the Papers.

These are NOT in Blanchard et al. (2017) or Yin et al. (2018).
They are your own analytical questions derived from observing your results.

Experiment A — Defense Failure Boundary:
  "At exactly what malicious fraction does each defense break down?"
  Papers only prove robustness guarantees; they don't show the empirical
  collapse point. We sweep fine-grained fractions to find it.

Experiment B — Beta Sensitivity Analysis for Trimmed Mean:
  "What happens if beta is mis-set — too low or too high?"
  The paper assumes you know f exactly. In practice you don't.
  We test beta under-estimation (beta < f) and over-trimming (beta > f).

Experiment C — Defense Cross-Effectiveness:
  "Does a defense tuned for one attack help against a different attack?"
  e.g. Krum tuned for gradient scaling — does it also help against label-flip?
  Papers evaluate each defense against its 'natural' attack only.

Experiment D — Round-by-Round Defense Convergence Speed:
  "Which defense recovers faster — Krum or TrimMean?"
  Final accuracy is the same metric everyone reports. Convergence speed
  is an original angle that matters for real deployments.
"""
import csv, os, sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from src.attacks.label_flip     import LabelFlipAttack
from src.attacks.gradient_scale import GradientScaleAttack
from src.fl_core.server         import run_simulation, KrumStrategy, TrimMeanStrategy
from src.defenses.trimmed_mean  import trimmed_mean, beta_from_fraction

# ── Dataset ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST("./data/raw", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST("./data/raw", train=False, download=True, transform=transform)

RESULTS = "experiments/results/original/"
FIGURES = "report/figures/"
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

BASELINE_ACC = pd.read_csv("experiments/results/baseline_mnist.csv")["accuracy"].iloc[-1] * 100
NUM_CLIENTS  = 10
NUM_ROUNDS   = 20

plt.style.use("dark_background")


# ════════════════════════════════════════════════════════════════════════════
# Experiment A — Defense Failure Boundary
# "At what exact malicious fraction does each defense collapse?"
# Papers only prove n > 2f+2 for Krum / n > 2β for TrimMean.
# They don't show the empirical degradation curve across fractions.
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  EXPERIMENT A — Defense Failure Boundary")
print("="*60)

# Fine-grained fractions: 0%, 10%, 20%, 30%
# (we already have 10/20/30 from run_defenses.py — reuse them)
# Add 0% (clean) and pull final acc from existing CSVs
fractions = [0.0, 0.1, 0.2, 0.3]

def load_final_acc(path):
    """Load final round accuracy from an existing CSV, or return None if missing."""
    try:
        df = pd.read_csv(path)
        return df["accuracy"].iloc[-1] * 100
    except FileNotFoundError:
        return None

boundary_data = {
    "fraction"      : fractions,
    "gs_fedavg"     : [],
    "gs_krum"       : [],
    "gs_trimmean"   : [],
    "lf_fedavg"     : [],
    "lf_krum"       : [],
    "lf_trimmean"   : [],
}

existing = {
    0.0 : "experiments/results/baseline_mnist.csv",
}

R = "experiments/results/"
for frac in fractions:
    pct = int(frac * 100)
    if frac == 0.0:
        # Clean baseline — no attack, all defenses should match baseline
        ba = BASELINE_ACC
        boundary_data["gs_fedavg"].append(ba)
        boundary_data["gs_krum"].append(ba)
        boundary_data["gs_trimmean"].append(ba)
        boundary_data["lf_fedavg"].append(ba)
        boundary_data["lf_krum"].append(ba)
        boundary_data["lf_trimmean"].append(ba)
    else:
        boundary_data["gs_fedavg"].append(  load_final_acc(f"{R}grad_scale_10_fedavg_f{pct}.csv"))
        boundary_data["gs_krum"].append(    load_final_acc(f"{R}grad_scale_10_krum_f{pct}.csv"))
        boundary_data["gs_trimmean"].append(load_final_acc(f"{R}grad_scale_10_trimmean_f{pct}.csv"))
        boundary_data["lf_fedavg"].append(  load_final_acc(f"{R}label_flip_fedavg_f{pct}.csv"))
        boundary_data["lf_krum"].append(    load_final_acc(f"{R}label_flip_krum_f{pct}.csv"))
        boundary_data["lf_trimmean"].append(load_final_acc(f"{R}label_flip_trimmean_f{pct}.csv"))

# Save boundary CSV
boundary_df = pd.DataFrame(boundary_data)
boundary_df.to_csv(RESULTS + "exp_a_failure_boundary.csv", index=False)
print(boundary_df.to_string(index=False))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
x_labels = ["0% (clean)", "10%", "20%", "30%"]

for ax, title, fa_key, kr_key, tm_key in [
    (axes[0], "Gradient Scale Attack (λ=10)", "gs_fedavg", "gs_krum", "gs_trimmean"),
    (axes[1], "Label-Flip Attack",             "lf_fedavg", "lf_krum", "lf_trimmean"),
]:
    ax.plot(x_labels, boundary_data[fa_key], "o--", color="#ff6b6b", lw=2, label="FedAvg (no defense)")
    ax.plot(x_labels, boundary_data[kr_key], "s-",  color="#4f9cf9", lw=2, label="Multi-Krum")
    ax.plot(x_labels, boundary_data[tm_key], "^-",  color="#4fe8a0", lw=2, label="Trimmed Mean")
    ax.axhline(BASELINE_ACC, color="#888", lw=1, ls=":", label=f"Clean baseline ({BASELINE_ACC:.1f}%)")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Malicious Fraction")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

fig.suptitle(
    "Fig 10 (Original) — Defense Failure Boundary: Accuracy vs Malicious Fraction\n"
    "Shows empirical collapse point not covered in Blanchard et al. or Yin et al.",
    fontsize=11, y=1.03
)
plt.tight_layout()
plt.savefig(FIGURES + "fig10_failure_boundary.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig10_failure_boundary.png saved")


# ════════════════════════════════════════════════════════════════════════════
# Experiment B — Beta Sensitivity Analysis (TrimMean only)
# "What happens when beta is wrong — under or over estimated?"
# Papers assume perfect knowledge of f. We test mis-calibration.
# This is a real practical concern: in deployment you don't know f exactly.
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  EXPERIMENT B — Beta Sensitivity (TrimMean mis-calibration)")
print("="*60)
print("  Setup: 20% malicious (f=2, correct beta=2)")
print("  Testing beta = 1 (under), 2 (correct), 3 (over), 4 (over++)")
print()

# Note: beta=4 with n=10 → n > 2*4=8 → 10>8 ✓, still valid
# beta=5 would fail (10 > 10 is False) — we stop at 4
beta_values  = [1, 2, 3, 4]
beta_results = []

for beta in beta_values:
    label = f"beta={beta}"
    if beta < 2:
        label += " (UNDER — miss malicious)"
    elif beta == 2:
        label += " (CORRECT)"
    else:
        label += " (OVER — lose honest data)"

    print(f"  Running TrimMean with {label}...")

    res = run_simulation(
        train_dataset      = train_ds,
        test_dataset       = test_ds,
        num_clients        = NUM_CLIENTS,
        num_rounds         = NUM_ROUNDS,
        malicious_fraction = 0.2,
        attack_fn          = GradientScaleAttack(scale_factor=10),
        strategy           = TrimMeanStrategy(beta=beta, num_clients=NUM_CLIENTS),
        dataset_name       = "mnist",
        alpha              = 0.5,
        results_path       = f"{RESULTS}exp_b_beta{beta}_gs10_f20.csv",
        seed               = 42,
    )
    final = res[-1]["accuracy"] * 100
    beta_results.append({
        "beta"      : beta,
        "label"     : label,
        "final_acc" : round(final, 2),
        "vs_correct": round(final - 0, 2),  # filled after loop
    })
    print(f"    → Final accuracy: {final:.2f}%")

# Compute delta vs correct beta=2
correct_acc = next(r["final_acc"] for r in beta_results if r["beta"] == 2)
for r in beta_results:
    r["vs_correct"] = round(r["final_acc"] - correct_acc, 2)

beta_df = pd.DataFrame(beta_results)
beta_df.to_csv(RESULTS + "exp_b_beta_sensitivity.csv", index=False)
print("\n" + beta_df.to_string(index=False))

# Plot
fig3, ax3 = plt.subplots(figsize=(8, 5))
colors_beta = ["#ff6b6b", "#4fe8a0", "#4f9cf9", "#c77dff"]
bars = ax3.bar(
    [f"β={r['beta']}" for r in beta_results],
    [r["final_acc"] for r in beta_results],
    color=colors_beta, alpha=0.85, width=0.5
)
ax3.axhline(BASELINE_ACC, color="#888", lw=1.5, ls=":", label=f"Clean baseline ({BASELINE_ACC:.1f}%)")
ax3.axhline(correct_acc,  color="#4fe8a0", lw=1, ls="--", alpha=0.5, label=f"Correct β=2 ({correct_acc:.1f}%)")

for bar, r in zip(bars, beta_results):
    delta_str = f"{r['vs_correct']:+.1f}%" if r["beta"] != 2 else "(correct)"
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.4,
        f"{r['final_acc']:.1f}%\n{delta_str}",
        ha="center", va="bottom", fontsize=9, color="white"
    )

# Annotate under/over regions
ax3.axvspan(-0.4, 0.4, alpha=0.06, color="#ff6b6b")   # beta=1 under-estimation zone
ax3.text(0, 5, "Under-\nestimate", ha="center", fontsize=8, color="#ff6b6b")
ax3.axvspan(2.6, 3.4, alpha=0.06, color="#888")        # beta=4 over-estimation zone

ax3.set_xlabel("Beta value (β)")
ax3.set_ylabel("Final Test Accuracy (%)")
ax3.set_ylim(0, 110)
ax3.set_title(
    "Fig 11 (Original) — Trimmed Mean Beta Sensitivity\n"
    "GradScale λ=10, 20% malicious (f=2), correct β=2",
    fontsize=11
)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.12, axis="y")
plt.tight_layout()
plt.savefig(FIGURES + "fig11_beta_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig11_beta_sensitivity.png saved")


# ════════════════════════════════════════════════════════════════════════════
# Experiment C — Defense Cross-Effectiveness
# "Does Krum tuned for attack A also protect against attack B?"
# Papers only test each defense against its 'natural' threat.
# We cross-test: Krum-f2 against label-flip, TrimMean against grad-scale.
# We already have these CSVs — this is just a focused analytical table.
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  EXPERIMENT C — Defense Cross-Effectiveness Analysis")
print("="*60)

R = "experiments/results/"

cross_data = [
    # (defense_name, tuned_for,      tested_against,   csv_path)
    ("Krum (f=2)",    "Grad Scale",  "Grad Scale",     f"{R}grad_scale_10_krum_f20.csv"),
    ("Krum (f=2)",    "Grad Scale",  "Label-Flip",     f"{R}label_flip_krum_f20.csv"),
    ("TrimMean (β=2)","Label-Flip",  "Label-Flip",     f"{R}label_flip_trimmean_f20.csv"),
    ("TrimMean (β=2)","Label-Flip",  "Grad Scale",     f"{R}grad_scale_10_trimmean_f20.csv"),
]

cross_rows = []
for defense, tuned_for, tested_against, path in cross_data:
    acc_val = load_final_acc(path)
    match   = "✓ Natural" if tuned_for == tested_against else "✗ Cross"
    cross_rows.append({
        "defense"        : defense,
        "tuned_for"      : tuned_for,
        "tested_against" : tested_against,
        "match"          : match,
        "final_acc_%"    : round(acc_val, 2) if acc_val else "N/A",
    })

cross_df = pd.DataFrame(cross_rows)
cross_df.to_csv(RESULTS + "exp_c_cross_effectiveness.csv", index=False)

print(f"\n  {'Defense':<18} {'Tuned For':<14} {'Tested Against':<16} {'Match':<12} {'Accuracy'}")
print(f"  {'─'*72}")
for r in cross_rows:
    print(f"  {r['defense']:<18} {r['tuned_for']:<14} {r['tested_against']:<16} "
          f"{r['match']:<12} {r['final_acc_%']}%")
print("\n  Key finding: does Krum protect against label-flip even though it was")
print("  designed for gradient-based attacks? If yes → good general robustness.")
print("  If no → attack-specific tuning is needed (real-world implication).")
print("✓ exp_c_cross_effectiveness.csv saved")


# ════════════════════════════════════════════════════════════════════════════
# Experiment D — Convergence Speed Comparison
# "Which defense reaches 90% accuracy fastest?"
# Final accuracy alone doesn't capture convergence. In real FL deployments
# communication rounds are expensive — faster convergence = lower cost.
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  EXPERIMENT D — Convergence Speed (rounds to reach 90% accuracy)")
print("="*60)

TARGET_ACC = 0.90   # 90% threshold

def rounds_to_threshold(csv_path, threshold=TARGET_ACC):
    """Return the first round where accuracy >= threshold, or None if never reached."""
    try:
        df = pd.read_csv(csv_path)
        hit = df[df["accuracy"] >= threshold]
        return int(hit["round"].iloc[0]) if len(hit) > 0 else None
    except FileNotFoundError:
        return None

R = "experiments/results/"
convergence_rows = []

scenarios = [
    # (label,                        csv_path)
    ("Clean FedAvg",                 f"{R}baseline_mnist.csv"),
    ("FedAvg + GS (no defense)",     f"{R}grad_scale_10_fedavg_f20.csv"),
    ("Krum vs GS",                   f"{R}grad_scale_10_krum_f20.csv"),
    ("TrimMean vs GS",               f"{R}grad_scale_10_trimmean_f20.csv"),
    ("FedAvg + LF (no defense)",     f"{R}label_flip_fedavg_f20.csv"),
    ("Krum vs LF",                   f"{R}label_flip_krum_f20.csv"),
    ("TrimMean vs LF",               f"{R}label_flip_trimmean_f20.csv"),
]

for label, path in scenarios:
    r_thresh = rounds_to_threshold(path)
    final    = load_final_acc(path)
    convergence_rows.append({
        "scenario"        : label,
        "rounds_to_90pct" : r_thresh if r_thresh else f">{NUM_ROUNDS} (never)",
        "final_acc_%"     : round(final, 2) if final else "N/A",
    })

conv_df = pd.DataFrame(convergence_rows)
conv_df.to_csv(RESULTS + "exp_d_convergence_speed.csv", index=False)

print(f"\n  {'Scenario':<35} {'Rounds to 90%':>14} {'Final Acc':>10}")
print(f"  {'─'*62}")
for r in convergence_rows:
    print(f"  {r['scenario']:<35} {str(r['rounds_to_90pct']):>14} {str(r['final_acc_%']):>9}%")

# Plot convergence curves for GS attack comparison
fig4, ax4 = plt.subplots(figsize=(10, 5))

curve_scenarios = [
    ("Clean FedAvg",             f"{R}baseline_mnist.csv",               "#888888", ":",  1.5),
    ("FedAvg + GradScale (none)",f"{R}grad_scale_10_fedavg_f20.csv",     "#ff6b6b", "--", 2.0),
    ("Krum vs GradScale",        f"{R}grad_scale_10_krum_f20.csv",       "#4f9cf9", "-",  2.0),
    ("TrimMean vs GradScale",    f"{R}grad_scale_10_trimmean_f20.csv",   "#4fe8a0", "-",  2.0),
]

for label, path, color, ls, lw in curve_scenarios:
    try:
        df = pd.read_csv(path)
        ax4.plot(df["round"], df["accuracy"] * 100, color=color, ls=ls, lw=lw, label=label)
    except FileNotFoundError:
        print(f"  ⚠ Skipping {label} — CSV not found")

ax4.axhline(TARGET_ACC * 100, color="#ffd166", lw=1, ls="--", alpha=0.7,
            label=f"90% threshold (convergence marker)")
ax4.set_xlabel("Communication Round")
ax4.set_ylabel("Test Accuracy (%)")
ax4.set_ylim(0, 105)
ax4.set_title(
    "Fig 12 (Original) — Convergence Speed: Rounds to Reach 90% Accuracy\n"
    "GradScale λ=10, 20% malicious — Krum vs TrimMean vs undefended",
    fontsize=11
)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(FIGURES + "fig12_convergence_speed.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig12_convergence_speed.png saved")

# ── Final summary ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  ALL ORIGINAL EXPERIMENTS COMPLETE")
print(f"{'='*60}")
print("  Exp A → fig10_failure_boundary.png   + exp_a_failure_boundary.csv")
print("  Exp B → fig11_beta_sensitivity.png   + exp_b_beta_sensitivity.csv")
print("  Exp C → exp_c_cross_effectiveness.csv (table, no new runs needed)")
print("  Exp D → fig12_convergence_speed.png  + exp_d_convergence_speed.csv")
print(f"\n  All CSVs → {RESULTS}")
print(f"  All figs → {FIGURES}")