"""
Gradient Scaling Attack (Model Poisoning)
Malicious clients multiply their gradient update by a large scale_factor
to dominate the FedAvg aggregation.
"""
import copy
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class GradientScaleAttack:
    """Scale malicious gradient updates by scale_factor to dominate FedAvg."""

    def __init__(self, scale_factor: float = 5.0):
        """
        Args:
            scale_factor: Lambda multiplier (e.g. 2, 5, 10).
                          Higher = stronger attack. 10x is severe.
        """
        self.scale_factor = scale_factor

    def apply(
        self,
        global_params: List[np.ndarray],
        local_params: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Given the global model params and the locally-trained params,
        compute the honest gradient delta, scale it, and return
        the poisoned update (as if it were local params).

        Returns poisoned params that, when aggregated, shift the global
        model by scale_factor × the malicious gradient.
        """
        poisoned = []
        for g, l in zip(global_params, local_params):
            delta = l - g                         # honest gradient
            scaled_delta = delta * self.scale_factor
            poisoned.append(g + scaled_delta)     # poisoned "local" params
        return poisoned

    def __repr__(self) -> str:
        return f"GradientScaleAttack(λ={self.scale_factor})"


if __name__ == "__main__":
    # Quick smoke test — verify scale_factor is applied correctly
    rng = np.random.default_rng(42)
    g_params = [rng.random((10, 5)), rng.random((10,))]
    l_params = [g + rng.random(g.shape) * 0.01
                for g in g_params]

    attack = GradientScaleAttack(scale_factor=5.0)
    poisoned = attack.apply(g_params, l_params)

    honest_norm  = np.linalg.norm(l_params[0] - g_params[0])
    poisoned_norm = np.linalg.norm(poisoned[0] - g_params[0])
    print(f"Honest delta norm:   {honest_norm:.4f}")
    print(f"Poisoned delta norm: {poisoned_norm:.4f}")
    print(f"Scale ratio: {poisoned_norm/honest_norm:.1f}× (should be ~5.0)")
    assert abs(poisoned_norm / honest_norm - 5.0) < 0.01, "Scale test FAILED"
    print("✓ GradientScaleAttack smoke test passed")