"""
FL Security Lab — Flower FL Client.
Supports honest training and malicious (attacked) training via attack_fn.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from src.fl_core.model import SimpleCNN
from src.attacks.gradient_scale import GradientScaleAttack
from src.attacks.model_replacement import ModelReplacementAttack

# Attacks that operate on gradients AFTER training (not on data BEFORE training)
MODEL_LEVEL_ATTACKS = (GradientScaleAttack, ModelReplacementAttack)


class FLClient(fl.client.NumPyClient):
    """
    A Flower NumPyClient that trains a SimpleCNN on a local data partition.

    Args:
        client_id    : Integer ID — used for logging and attack identification.
        dataset      : Full training dataset (will be subsetted by indices).
        indices      : List of sample indices assigned to this client.
        model        : SimpleCNN instance (shared architecture, separate weights).
        local_epochs : Number of local SGD epochs per round (default: 2).
        batch_size   : Local mini-batch size (default: 32).
        lr           : SGD learning rate (default: 0.01).
        device       : 'cpu' or 'cuda'.
        attack_fn    : Optional attack object or callable.
                       - None                   → honest client
                       - callable               → data-level attack (label-flip, backdoor)
                       - GradientScaleAttack    → model-level attack (post-training)
                       - ModelReplacementAttack → model-level attack (post-training)
        scale_factor : Lambda for GradientScaleAttack. Pulled from attack_fn automatically
                       in server.py — do not set manually.
    """

    def __init__(
        self,
        client_id: int,
        dataset,
        indices: List[int],
        model: SimpleCNN,
        local_epochs: int = 2,
        batch_size: int = 32,
        lr: float = 0.01,
        device: str = "cpu",
        attack_fn: Optional[Callable] = None,
        scale_factor: float = 1.0,
    ):
        self.client_id    = client_id
        self.dataset      = dataset
        self.indices      = indices
        self.model        = model.to(device)
        self.local_epochs = local_epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.device       = device
        self.attack_fn    = attack_fn
        self.scale_factor = scale_factor
        self.criterion    = nn.CrossEntropyLoss()

    # ── Parameter I/O ─────────────────────────────────────
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model weights as list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load server-provided weights into local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    # ── Local Training ─────────────────────────────────────
    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        1. Load global model weights.
        2. Apply data-level attack if applicable (label-flip, backdoor).
        3. Run local_epochs of SGD.
        4. Apply model-level attack if applicable (gradient scaling, model replacement).
        5. Return updated weights + dataset size + metrics.
        """
        self.set_parameters(parameters)

        # ── Data-level attack (label-flip, backdoor) ──────────
        # Model-level attacks skip this — they don't touch the data.
        if self.attack_fn is not None and not isinstance(self.attack_fn, MODEL_LEVEL_ATTACKS):
            dataset, indices = self.attack_fn(self.dataset, self.indices)
        else:
            dataset, indices = self.dataset, self.indices

        local_data   = Subset(dataset, indices)
        train_loader = DataLoader(
            local_data, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4
        )
        self.model.train()

        total_loss = 0.0
        for epoch in range(self.local_epochs):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / (self.local_epochs * len(train_loader))

        # ── Gradient Scaling Attack (model-level) ──────────────
        # Scales the gradient delta by scale_factor to dominate FedAvg.
        if isinstance(self.attack_fn, GradientScaleAttack):
            poisoned = self.attack_fn.apply(parameters, self.get_parameters(config={}))
            return poisoned, len(indices), {"loss": avg_loss}

        # ── Model Replacement Attack (model-level) ─────────────
        # Crafts an update that pushes the global model toward a target.
        # The locally trained model IS the target — apply boost formula.
        if isinstance(self.attack_fn, ModelReplacementAttack):
            poisoned = self.attack_fn.apply(parameters, self.get_parameters(config={}))
            return poisoned, len(indices), {"loss": avg_loss}

        # ── Honest return (or data-level attack return) ────────
        return self.get_parameters(config={}), len(indices), {"loss": avg_loss}

    # ── Local Evaluation ──────────────────────────────────
    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate current global model on this client's local data."""
        self.set_parameters(parameters)
        local_data   = Subset(self.dataset, self.indices)
        eval_loader  = DataLoader(local_data, batch_size=64, shuffle=False, num_workers=0)

        self.model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in eval_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                correct    += (outputs.argmax(dim=1) == labels).sum().item()

        n    = len(self.indices)
        loss = total_loss / n
        acc  = correct / n
        return loss, n, {"accuracy": acc}