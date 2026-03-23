"""
Model Replacement Attack (Bagdasaryan et al., 2020)
Crafts an update so that FedAvg aggregation pushes the global model
toward an adversarial target model chosen by the attacker.
"""
from typing import List
import numpy as np


class ModelReplacementAttack:
    """
    Given the global model and a target model (e.g., fine-tuned to
    misclassify a specific class), craft an adversarial gradient update
    such that: new_global ≈ target_model after FedAvg averaging.
    """

    def __init__(self, malicious_fraction: float = 0.1):
        """
        Args:
            malicious_fraction: fraction of malicious clients among all clients.
                e.g. 1 out of 10 = 0.1. Used to scale the boost factor.
        """
        self.malicious_fraction = malicious_fraction

    def apply(
        self,
        global_params: List[np.ndarray],
        target_params: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Craft poisoned params such that after FedAvg:
          w_global_new ≈ target_params

        Uses boost factor = 1 / malicious_fraction so one malicious
        client can fully replace the global model.
        """
        boost = 1.0 / self.malicious_fraction
        poisoned = []
        for g, t in zip(global_params, target_params):
            # Poisoned "local params" that, after averaging, push global → target
            p = g + boost * (t - g)
            poisoned.append(p)
        return poisoned

    def __repr__(self) -> str:
        return f"ModelReplacementAttack(fraction={self.malicious_fraction})"


if __name__ == "__main__":
    # Smoke test: verify global moves toward target after one step
    rng = np.random.default_rng(0)
    shape = (20, 10)
    g = [rng.random(shape)]
    t = [rng.random(shape)]

    attack = ModelReplacementAttack(malicious_fraction=0.1)
    poisoned = attack.apply(g, t)

    # Simulate one-client FedAvg: new_global = 0.1*poisoned + 0.9*global
    n_total, n_malicious = 10, 1
    new_global = (n_malicious / n_total) * poisoned[0] + \
                 ((n_total - n_malicious) / n_total) * g[0]
    # With boost = 10, new_global should ≈ target
    diff = np.mean(np.abs(new_global - t[0]))
    print(f"Avg |new_global - target|: {diff:.6f} (should be ~0.0)")
    assert diff < 1e-5, "ModelReplacement test FAILED"
    print("✓ ModelReplacementAttack smoke test passed")