"""
FL Security Lab — FedAvg Server / Simulation Runner.
Builds the Flower simulation, logs per-round accuracy & loss to CSV.
"""
import csv
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import flwr as fl
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader

from src.fl_core.model import SimpleCNN, get_model
from src.fl_core.client import FLClient
from src.utils.data_partition import dirichlet_partition
from src.attacks.gradient_scale import GradientScaleAttack
from src.attacks.model_replacement import ModelReplacementAttack


# ══════════════════════════════════════════════════════════
# Metric aggregation helpers (required by Flower)
# ══════════════════════════════════════════════════════════
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average of accuracy across clients (weighted by dataset size)."""
    total_examples = sum(num for num, _ in metrics)
    accuracies     = [num * m.get("accuracy", 0) for num, m in metrics]
    return {"accuracy": sum(accuracies) / total_examples}


# ══════════════════════════════════════════════════════════
# Server-side global evaluation
# ══════════════════════════════════════════════════════════
def make_evaluate_fn(model: SimpleCNN, test_loader: DataLoader, device: str = "cpu"):
    """
    Returns a Flower evaluate_fn that tests the global model on the clean test set.
    Called by the server after each aggregation round.
    """
    criterion = torch.nn.CrossEntropyLoss()

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict  = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()

        total_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                total_loss += criterion(outputs, labels).item() * images.size(0)
                correct    += (outputs.argmax(dim=1) == labels).sum().item()
                n          += images.size(0)
        return total_loss / n, {"accuracy": correct / n}

    return evaluate


# ══════════════════════════════════════════════════════════
# Main simulation runner
# ══════════════════════════════════════════════════════════
def run_simulation(
    train_dataset,
    test_dataset,
    num_clients:        int   = 10,
    num_rounds:         int   = 20,
    fraction_fit:       float = 1.0,
    local_epochs:       int   = 2,
    batch_size:         int   = 32,
    lr:                 float = 0.01,
    alpha:              float = 0.5,
    malicious_fraction: float = 0.0,
    attack_fn:          Optional[Callable] = None,
    dataset_name:       str   = "mnist",
    results_path:       str   = "experiments/results/baseline_mnist.csv",
    seed:               int   = 42,
) -> List[Dict]:

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*55}")
    print(f"  FL Simulation | {num_clients} clients | {num_rounds} rounds")
    print(f"  Dataset: {dataset_name} | α={alpha} | device={device}")
    print(f"  Malicious fraction: {malicious_fraction:.0%} | Attack: {attack_fn.__class__.__name__ if attack_fn else 'None'}")
    print(f"{'='*55}\n")

    # ── Partition data ───────────────────────────────────
    client_subsets = dirichlet_partition(
        train_dataset, num_clients=num_clients, alpha=alpha, seed=seed
    )

    # ── Determine malicious clients ──────────────────────
    num_malicious = int(num_clients * malicious_fraction)
    malicious_ids = set(range(num_malicious))

    # ── Test loader for server evaluation ───────────────
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    eval_model  = get_model(dataset_name)

    round_results = []
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # ── Client factory ───────────────────────────────────
    def client_fn(cid: str) -> FLClient:
        client_id    = int(cid)
        is_malicious = client_id in malicious_ids
        client_model = get_model(dataset_name)

        # Pull scale_factor from the attack object itself so it's
        # always in sync — no risk of mismatch between server and client.
        scale_factor = 1.0
        if is_malicious and isinstance(attack_fn, GradientScaleAttack):
            scale_factor = attack_fn.scale_factor

        return FLClient(
            client_id    = client_id,
            dataset      = train_dataset,
            indices      = list(client_subsets[client_id].indices),
            model        = client_model,
            local_epochs = local_epochs,
            batch_size   = batch_size,
            lr           = lr,
            device       = device,
            attack_fn    = attack_fn if is_malicious else None,
            scale_factor = scale_factor,
        )

    # ── Evaluation callback ──────────────────────────────
    evaluate_fn = make_evaluate_fn(eval_model, test_loader, device)

    def evaluate_with_log(server_round, parameters, config):
        loss, metrics = evaluate_fn(server_round, parameters, config)
        acc    = metrics["accuracy"]
        result = {"round": server_round, "accuracy": acc, "loss": loss}
        round_results.append(result)
        print(f"  Round {server_round:2d} | Acc: {acc:.4f} ({acc*100:.2f}%) | Loss: {loss:.4f}")
        return loss, metrics

    # ── Strategy ─────────────────────────────────────────
    strategy = FedAvg(
        fraction_fit               = fraction_fit,
        fraction_evaluate          = 0.0,
        min_fit_clients            = num_clients,
        min_available_clients      = num_clients,
        evaluate_fn                = evaluate_with_log,
        fit_metrics_aggregation_fn = weighted_average,
    )

    # ── Launch ───────────────────────────────────────────
    fl.simulation.start_simulation(
        client_fn     = client_fn,
        num_clients   = num_clients,
        config        = fl.server.ServerConfig(num_rounds=num_rounds),
        strategy      = strategy,
        ray_init_args = {"num_cpus": os.cpu_count(), "include_dashboard": False},
    )

    # ── Save CSV ─────────────────────────────────────────
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "accuracy", "loss"])
        writer.writeheader()
        writer.writerows(round_results)
    print(f"\n  ✓ Results saved → {results_path}")
    print(f"  Final accuracy: {round_results[-1]['accuracy']*100:.2f}%\n")

    return round_results