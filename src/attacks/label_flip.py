"""
FL Security Lab — Label-Flip Attack.
Used by malicious FL clients to poison their local training data.
"""
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple


class PoisonedDataset(Dataset):
    """
    Wraps a dataset and replaces labels according to a flip_map.
    Does NOT modify the original dataset — creates a poisoned copy.
    """
    def __init__(self, dataset, indices: List[int], flip_map: dict):
        self.dataset   = dataset
        self.indices   = indices
        self.flip_map  = flip_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.dataset[real_idx]
        # Flip label if in the map
        label = self.flip_map.get(label, label)
        return image, label


class LabelFlipAttack:
    """
    Label-flipping attack for malicious FL clients.

    Modes:
        targeted  : Flip a specific source_class → target_class.
                    E.g., all 3s become 8s. Use this for the report.
        random    : Randomly reassign labels from a random permutation.
                    More disruptive but harder to analyze.

    Usage:
        attack = LabelFlipAttack(mode="targeted", source_class=3, target_class=8)
        # In client.py, attack_fn is called as:
        poisoned_dataset, indices = attack(original_dataset, client_indices)
    """

    def __init__(
        self,
        mode: str = "targeted",
        source_class: int = 3,
        target_class: int = 8,
        num_classes: int = 10,
        seed: int = 42,
    ):
        assert mode in ("targeted", "random"), "mode must be 'targeted' or 'random'"
        self.mode         = mode
        self.source_class = source_class
        self.target_class = target_class
        self.num_classes  = num_classes
        self.seed         = seed
        self._name        = f"LabelFlip({mode}, {source_class}→{target_class})"

    def __repr__(self):
        return self._name

    def __call__(self, dataset, indices: List[int]) -> Tuple[Dataset, List[int]]:
        """
        Apply label-flip poisoning to the client's local data.

        Returns:
            (poisoned_dataset, original_indices)
            The poisoned_dataset is a PoisonedDataset wrapping the original.
        """
        if self.mode == "targeted":
            flip_map = {self.source_class: self.target_class}

        elif self.mode == "random":
            rng = np.random.RandomState(self.seed)
            perm = rng.permutation(self.num_classes)
            # Ensure it's actually a permutation (no class maps to itself)
            for i in range(self.num_classes):
                if perm[i] == i:
                    swap = (i + 1) % self.num_classes
                    perm[i], perm[swap] = perm[swap], perm[i]
            flip_map = {i: int(perm[i]) for i in range(self.num_classes)}

        poisoned = PoisonedDataset(dataset, indices, flip_map)

        # Count flipped samples for logging
        original_labels = [dataset[i][1] for i in indices]
        if self.mode == "targeted":
            n_flipped = sum(1 for l in original_labels if l == self.source_class)
        else:
            n_flipped = len(indices)  # all labels are flipped in random mode

        print(f"    [LabelFlip] {n_flipped}/{len(indices)} samples flipped | map={flip_map}")
        return poisoned, list(range(len(poisoned)))


if __name__ == "__main__":
    from torchvision import datasets, transforms
    ds = datasets.MNIST("./data/raw", train=True, download=True,
                        transform=transforms.ToTensor())
    indices = list(range(500))

    attack = LabelFlipAttack(mode="targeted", source_class=3, target_class=8)
    poisoned, new_idx = attack(ds, indices)

    # Verify flip worked
    orig_3_count    = sum(1 for i in indices if ds[i][1] == 3)
    poisoned_3_count = sum(1 for i in new_idx if poisoned[i][1] == 3)
    poisoned_8_count = sum(1 for i in new_idx if poisoned[i][1] == 8)
    print(f"Original class-3 count  : {orig_3_count}")
    print(f"Poisoned class-3 count  : {poisoned_3_count}  (should be 0)")
    print(f"Poisoned class-8 count  : {poisoned_8_count}  (should be > original)")
    print("✓ LabelFlipAttack test passed")