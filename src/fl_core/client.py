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
from src.fl_core.model import SimpleCNN


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
        attack_fn    : Optional callable that modifies the dataset before training.
                       Signature: attack_fn(dataset, indices) → modified_dataset, new_indices
                       Pass None for honest clients.
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
        self.criterion    = nn.CrossEntropyLoss()

    # ── Parameter I/O ─────────────────────────────────────
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model weights as list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load server-provided weights into local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    # ── Local Training ─────────────────────────────────────
    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        1. Load global model weights.
        2. Optionally apply attack to local data.
        3. Run local_epochs of SGD.
        4. Return updated weights + dataset size + metrics.
        """
        self.set_parameters(parameters)

        # Apply attack to local data if this is a malicious client
        if self.attack_fn is not None:
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
