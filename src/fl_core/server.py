"""
FL Security Lab — FedAvg Server / Simulation Runner.
Builds the Flower simulation, logs per-round accuracy & loss to CSV.

Supported strategies:
  - FedAvg             : plain weighted average (no defense)
  - KrumStrategy       : Multi-Krum or Single-Krum (Blanchard et al., 2017)
  - TrimMeanStrategy   : Coordinate-wise Trimmed Mean (Yin et al., 2018)

BW4 additions:
  - KrumStrategy and TrimMeanStrategy now collect per-client gradient L2
    norms during every aggregate_fit() call via _collect_norms().
  - run_simulation() automatically exports norm logs to a *_norms.csv
    file alongside the accuracy CSV after each run.
  - malicious_ids is passed into the strategy so norm logs are labelled
    correctly (is_malicious column) for detector evaluation.
"""
import csv
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import flwr as fl
from flwr.common import Metrics, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader

from src.fl_core.model import SimpleCNN, get_model
from src.fl_core.client import FLClient
from src.utils.data_partition import dirichlet_partition
from src.attacks.gradient_scale import GradientScaleAttack
from src.attacks.model_replacement import ModelReplacementAttack
from src.defenses.krum import multi_krum, single_krum
from src.defenses.trimmed_mean import trimmed_mean, beta_from_fraction


# ══════════════════════════════════════════════════════════
# Metric aggregation helpers (required by Flower)
# ══════════════════════════════════════════════════════════
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average of accuracy across clients (weighted by dataset size)."""
    total_examples = sum(num for num, _ in metrics)
    accuracies     = [num * m.get("accuracy", 0) for num, m in metrics]
    return {"accuracy": sum(accuracies) / total_examples}


# ══════════════════════════════════════════════════════════
# BW4 — Internal norm collection helper
# ══════════════════════════════════════════════════════════
def _collect_norms(strategy_obj, server_round: int, results, malicious_ids: set):
    """
    Compute and store gradient L2 norms for every client in this round.

    Called at the top of both KrumStrategy.aggregate_fit() and
    TrimMeanStrategy.aggregate_fit(). Results are accumulated in
    strategy_obj._norm_logs (list of dicts) and exported to CSV by
    run_simulation() after start_simulation() returns.

    Args:
        strategy_obj  : The strategy instance (self inside aggregate_fit)
        server_round  : Current Flower round number
        results       : Raw Flower fit results list
        malicious_ids : Set of client_id integers that are malicious
    """
    if not hasattr(strategy_obj, "_norm_logs"):
        strategy_obj._norm_logs = []

    for client_idx, (_, fit_res) in enumerate(results):
        local_params = parameters_to_ndarrays(fit_res.parameters)
        # Flatten all layers and compute the L2 norm of the full parameter vector.
        # This approximates the gradient update norm when local_params ≈ w_global + Δw.
        flat    = np.concatenate([p.flatten() for p in local_params])
        l2_norm = float(np.linalg.norm(flat))
        strategy_obj._norm_logs.append({
            "round"        : server_round,
            "client_id"    : client_idx,
            "l2_norm"      : l2_norm,
            "is_malicious" : client_idx in malicious_ids,
        })


# ══════════════════════════════════════════════════════════
# Krum Strategy — replaces FedAvg aggregation with Krum
# ══════════════════════════════════════════════════════════
class KrumStrategy(FedAvg):
    """FedAvg strategy with Krum aggregation replacing plain averaging."""

    def __init__(self, f: int, use_multi_krum: bool = True, m: int = None, **kwargs):
        """
        Args:
            f             : Number of assumed malicious clients.
            use_multi_krum: If True use Multi-Krum (recommended); else Single-Krum.
            m             : Number of clients to select in Multi-Krum (default n-f).
            **kwargs      : All standard FedAvg arguments (fraction_fit, evaluate_fn, etc.)
        """
        super().__init__(**kwargs)
        self.f               = f
        self.use_multi_krum  = use_multi_krum
        self.m               = m
        # BW4: norm log storage — populated by _collect_norms() each round
        self._norm_logs      = []
        # BW4: malicious_ids injected by run_simulation() before first round
        self._malicious_ids: set = set()

    def aggregate_fit(self, server_round, results, failures):
        """Override FedAvg aggregation with Krum."""
        if not results:
            return None, {}

        # BW4 — collect gradient L2 norms before aggregating
        _collect_norms(self, server_round, results, self._malicious_ids)

        # Unpack client updates into list of parameter arrays
        all_updates = []
        for _, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            all_updates.append(params)

        # Apply Krum aggregation
        if self.use_multi_krum:
            aggregated = multi_krum(all_updates, f=self.f, m=self.m)
        else:
            aggregated = single_krum(all_updates, f=self.f)

        parameters_aggregated = ndarrays_to_parameters(aggregated)
        metrics_aggregated    = {}
        return parameters_aggregated, metrics_aggregated


# ══════════════════════════════════════════════════════════
# TrimMean Strategy — replaces FedAvg aggregation with
# coordinate-wise Trimmed Mean (Yin et al., 2018)
# ══════════════════════════════════════════════════════════
class TrimMeanStrategy(FedAvg):
    """
    FedAvg strategy with coordinate-wise Trimmed Mean aggregation.

    For each model parameter coordinate, sorts values across all clients,
    discards the lowest beta and highest beta, then averages the rest.
    This is effective against gradient scaling attacks where malicious
    updates push coordinates to extreme values.

    Requirements:
        beta >= f          (must trim at least as many as malicious clients)
        n > 2 * beta       (need at least 1 value remaining after trimming)

    Usage:
        strategy = TrimMeanStrategy(
            malicious_fraction=0.2,   # 20% malicious → beta computed automatically
            num_clients=10,
        )
        # OR set beta directly:
        strategy = TrimMeanStrategy(beta=2)
    """

    def __init__(
        self,
        malicious_fraction: float = 0.0,
        num_clients: int = 10,
        beta: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            malicious_fraction : Fraction of malicious clients (e.g. 0.2 for 20%).
                                 Used to compute beta automatically if beta=None.
            num_clients        : Total clients per round. Used only for auto beta.
            beta               : Override beta directly. If set, malicious_fraction
                                 and num_clients are ignored for beta computation.
                                 Must satisfy n > 2*beta where n = clients per round.
            **kwargs           : All standard FedAvg arguments.
        """
        super().__init__(**kwargs)
        self.malicious_fraction  = malicious_fraction
        self.num_clients         = num_clients
        # BW4: norm log storage — populated by _collect_norms() each round
        self._norm_logs          = []
        # BW4: malicious_ids injected by run_simulation() before first round
        self._malicious_ids: set = set()

        # Resolve beta at construction time if possible
        # (if beta=None and fraction=0 we will just set beta=0)
        if beta is not None:
            self._beta = beta
            self._beta_source = f"manual (beta={beta})"
        elif malicious_fraction > 0.0:
            self._beta = beta_from_fraction(num_clients, malicious_fraction)
            self._beta_source = (
                f"auto from fraction={malicious_fraction:.0%}, "
                f"n={num_clients} → beta={self._beta}"
            )
        else:
            self._beta = 0
            self._beta_source = "beta=0 (no malicious clients configured)"

    @property
    def beta(self) -> int:
        return self._beta

    def aggregate_fit(self, server_round, results, failures):
        """Override FedAvg aggregation with coordinate-wise Trimmed Mean."""
        if not results:
            return None, {}

        # BW4 — collect gradient L2 norms before aggregating
        _collect_norms(self, server_round, results, self._malicious_ids)

        # Unpack client updates
        all_updates = []
        for _, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            all_updates.append(params)

        n = len(all_updates)

        # If beta=0 (clean run), fall back to plain average — no trimming needed
        if self._beta == 0:
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            aggregated = [
                np.sum(
                    [
                        (parameters_to_ndarrays(fit_res.parameters)[i] * fit_res.num_examples)
                        / total_examples
                        for _, fit_res in results
                    ],
                    axis=0,
                )
                for i in range(len(all_updates[0]))
            ]
        else:
            # Validate beta is still feasible with the actual number of
            # participating clients this round (n can vary if fraction_fit < 1.0)
            if n <= 2 * self._beta:
                raise RuntimeError(
                    f"TrimMeanStrategy: round {server_round} received n={n} clients "
                    f"but beta={self._beta} requires n > {2 * self._beta}. "
                    f"Increase num_clients or reduce malicious_fraction."
                )
            aggregated = trimmed_mean(all_updates, beta=self._beta)

        parameters_aggregated = ndarrays_to_parameters(aggregated)
        metrics_aggregated    = {}
        return parameters_aggregated, metrics_aggregated

    def __repr__(self) -> str:
        return f"TrimMeanStrategy(beta={self._beta}, source={self._beta_source})"


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
    num_clients:        int                = 10,
    num_rounds:         int                = 20,
    fraction_fit:       float              = 1.0,
    local_epochs:       int                = 2,
    batch_size:         int                = 32,
    lr:                 float              = 0.01,
    alpha:              float              = 0.5,
    malicious_fraction: float              = 0.0,
    attack_fn:          Optional[Callable] = None,
    strategy:           Optional[FedAvg]   = None,   # pass KrumStrategy or TrimMeanStrategy
    dataset_name:       str                = "mnist",
    results_path:       str                = "experiments/results/baseline_mnist.csv",
    seed:               int                = 42,
) -> List[Dict]:
    """
    Run a complete FL simulation.

    Args:
        train_dataset      : Full torchvision training dataset.
        test_dataset       : Full torchvision test dataset.
        malicious_fraction : Fraction of clients that are malicious (0.0 = clean run).
        attack_fn          : Attack object passed to malicious clients.
        strategy           : Flower strategy to use.
                             - None             → standard FedAvg (no defense)
                             - KrumStrategy(f=2) → Multi-Krum defense
                             - TrimMeanStrategy(malicious_fraction=0.2, num_clients=10)
                                                → Trimmed Mean defense
        results_path       : CSV file path for logging per-round results.

    Returns:
        List of per-round result dicts: {round, accuracy, loss}

    BW4 side-effect:
        If strategy is KrumStrategy or TrimMeanStrategy, a second CSV is
        saved automatically at results_path.replace(".csv", "_norms.csv")
        containing per-round, per-client gradient L2 norms with is_malicious
        labels — ready for anomaly detector evaluation.

    Examples:
        # Clean baseline
        run_simulation(train_ds, test_ds)

        # Gradient scaling attack, no defense
        run_simulation(train_ds, test_ds,
                       malicious_fraction=0.2,
                       attack_fn=GradientScaleAttack(scale_factor=10.0))

        # Gradient scaling attack + Krum defense
        run_simulation(train_ds, test_ds,
                       malicious_fraction=0.2,
                       attack_fn=GradientScaleAttack(scale_factor=10.0),
                       strategy=KrumStrategy(f=2))

        # Gradient scaling attack + Trimmed Mean defense
        run_simulation(train_ds, test_ds,
                       malicious_fraction=0.2,
                       attack_fn=GradientScaleAttack(scale_factor=10.0),
                       strategy=TrimMeanStrategy(malicious_fraction=0.2, num_clients=10))
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device        = "cuda" if torch.cuda.is_available() else "cpu"
    strategy_name = strategy.__class__.__name__ if strategy else "FedAvg"

    # Print TrimMean beta info so it's visible in the log
    if isinstance(strategy, TrimMeanStrategy):
        strategy_name = f"TrimMeanStrategy(β={strategy.beta})"

    print(f"\n{'='*55}")
    print(f"  FL Simulation | {num_clients} clients | {num_rounds} rounds")
    print(f"  Dataset: {dataset_name} | α={alpha} | device={device}")
    print(f"  Malicious fraction: {malicious_fraction:.0%} | Attack: {attack_fn.__class__.__name__ if attack_fn else 'None'}")
    print(f"  Strategy: {strategy_name}")
    print(f"{'='*55}\n")

    # ── Partition data (non-IID Dirichlet) ──────────────
    client_subsets = dirichlet_partition(
        train_dataset, num_clients=num_clients, alpha=alpha, seed=seed
    )

    # ── Determine which clients are malicious ───────────
    num_malicious = int(num_clients * malicious_fraction)
    malicious_ids = set(range(num_malicious))

    # BW4 — inject malicious_ids into strategy so norm logs are labelled correctly
    if strategy is not None and hasattr(strategy, "_malicious_ids"):
        strategy._malicious_ids = malicious_ids

    # ── Build test DataLoader for server evaluation ─────
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    # ── Global model for server-side evaluation ─────────
    eval_model = get_model(dataset_name)

    # ── Results storage ──────────────────────────────────
    round_results = []
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # ── Flower client_fn factory ─────────────────────────
    def client_fn(cid: str) -> FLClient:
        client_id    = int(cid)
        is_malicious = client_id in malicious_ids
        client_model = get_model(dataset_name)

        # Pull scale_factor from GradientScaleAttack object if applicable
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

    # ── Evaluation callback (stores results) ─────────────
    evaluate_fn = make_evaluate_fn(eval_model, test_loader, device)

    def evaluate_with_log(server_round, parameters, config):
        loss, metrics = evaluate_fn(server_round, parameters, config)
        acc    = metrics["accuracy"]
        result = {"round": server_round, "accuracy": acc, "loss": loss}
        round_results.append(result)
        print(f"  Round {server_round:2d} | Acc: {acc:.4f} ({acc*100:.2f}%) | Loss: {loss:.4f}")
        return loss, metrics

    # ── Build strategy ────────────────────────────────────
    # Shared kwargs injected into whichever strategy is active.
    # These override any defaults set at construction time.
    strategy_kwargs = dict(
        fraction_fit               = fraction_fit,
        fraction_evaluate          = 0.0,
        min_fit_clients            = num_clients,
        min_available_clients      = num_clients,
        evaluate_fn                = evaluate_with_log,
        fit_metrics_aggregation_fn = weighted_average,
    )

    if strategy is None:
        # Standard FedAvg — no defense
        active_strategy = FedAvg(**strategy_kwargs)
    else:
        # Inject evaluation / fit settings into the provided strategy instance.
        # Works for KrumStrategy, TrimMeanStrategy, or any FedAvg subclass.
        for key, val in strategy_kwargs.items():
            setattr(strategy, key, val)
        active_strategy = strategy

    # ── Launch simulation ─────────────────────────────────
    fl.simulation.start_simulation(
        client_fn     = client_fn,
        num_clients   = num_clients,
        config        = fl.server.ServerConfig(num_rounds=num_rounds),
        strategy      = active_strategy,
        ray_init_args = {"num_cpus": os.cpu_count(), "include_dashboard": False},
    )

    # ── Save accuracy CSV ─────────────────────────────────
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "accuracy", "loss"])
        writer.writeheader()
        writer.writerows(round_results)
    print(f"\n  ✓ Results saved → {results_path}")
    print(f"  Final accuracy: {round_results[-1]['accuracy']*100:.2f}%\n")

    # ── BW4: Export gradient norm logs ───────────────────
    # If the active strategy collected norm logs (KrumStrategy or
    # TrimMeanStrategy), save them as a companion CSV for anomaly detection.
    if hasattr(active_strategy, "_norm_logs") and active_strategy._norm_logs:
        norm_path = results_path.replace(".csv", "_norms.csv")
        norm_df   = pd.DataFrame(active_strategy._norm_logs)
        norm_df.to_csv(norm_path, index=False)
        print(f"  ✓ Norm logs saved  → {norm_path}")
        print(f"    Rounds collected : {norm_df['round'].nunique()}")
        print(f"    Clients per round: {norm_df.groupby('round')['client_id'].nunique().mean():.0f}")
        if norm_df["is_malicious"].any():
            mal_mean = norm_df[norm_df["is_malicious"]]["l2_norm"].mean()
            hon_mean = norm_df[~norm_df["is_malicious"]]["l2_norm"].mean()
            print(f"    Honest norm µ    : {hon_mean:.4f}")
            print(f"    Malicious norm µ : {mal_mean:.4f}  ({mal_mean/hon_mean:.1f}× ratio)")

    return round_results