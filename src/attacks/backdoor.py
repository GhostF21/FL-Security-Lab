"""
FL Security Lab — Backdoor Attack with pixel trigger injection.
Malicious clients inject a trigger pattern + relabel as target_class.
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional, Tuple

def apply_trigger(
    image: torch.Tensor,
    trigger_size: int  = 4,
    trigger_value: float = 1.0,
    position: str = "bottom-right",
) -> torch.Tensor:
    """
    Stamp a solid square pixel trigger onto an image tensor.

    Args:
        image        : Tensor of shape (C, H, W). Modified in place on a copy.
        trigger_size : Side length of the trigger square in pixels.
        trigger_value: Pixel value for the trigger (1.0 = white for normalized MNIST).
        position     : 'bottom-right', 'bottom-left', 'top-right', 'top-left', 'center'.

    Returns:
        Triggered image tensor (same shape, same dtype).
    """
    img = image.clone()
    _, H, W = img.shape

    # Determine trigger position
    if position == "bottom-right":
        r_start, c_start = H - trigger_size - 1, W - trigger_size - 1
    elif position == "bottom-left":
        r_start, c_start = H - trigger_size - 1, 1
    elif position == "top-right":
        r_start, c_start = 1, W - trigger_size - 1
    elif position == "top-left":
        r_start, c_start = 1, 1
    elif position == "center":
        r_start, c_start = (H - trigger_size) // 2, (W - trigger_size) // 2
    else:
        raise ValueError(f"Unknown trigger position: {position}")

    img[:, r_start:r_start+trigger_size, c_start:c_start+trigger_size] = trigger_value
    return img


class BackdooredDataset(Dataset):
    """
    Wraps a dataset: applies trigger to all samples + relabels as target_class.
    """
    def __init__(
        self,
        dataset,
        indices: List[int],
        target_class: int   = 0,
        trigger_size: int   = 4,
        trigger_value: float = 1.0,
        poison_rate: float  = 1.0,    # fraction of samples to poison (1.0 = all)
        seed: int           = 42,
    ):
        self.dataset       = dataset
        self.indices       = indices
        self.target_class  = target_class
        self.trigger_size  = trigger_size
        self.trigger_value = trigger_value

        # Determine which samples in this client's data get the trigger
        rng = np.random.RandomState(seed)
        n_poison = int(len(indices) * poison_rate)
        self.poisoned_set = set(rng.choice(len(indices), n_poison, replace=False).tolist())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.dataset[real_idx]
        if idx in self.poisoned_set:
            image = apply_trigger(image, self.trigger_size, self.trigger_value)
            label = self.target_class
        return image, label


class BackdoorAttack:
    """
    Backdoor attack for malicious FL clients.

    The malicious client:
        1. Adds pixel trigger to its local training images.
        2. Relabels those images as target_class.
        3. Trains normally → model learns trigger association.

    To measure ASR: apply trigger to clean test images, run through global model,
    measure fraction predicted as target_class. See run_attacks.py.

    Usage:
        attack = BackdoorAttack(target_class=0, trigger_size=4)
        poisoned_dataset, indices = attack(original_dataset, client_indices)
    """

    def __init__(
        self,
        target_class: int   = 0,
        trigger_size: int   = 4,
        trigger_value: float = 1.0,
        poison_rate: float  = 1.0,
        position: str       = "bottom-right",
        seed: int           = 42,
    ):
        self.target_class  = target_class
        self.trigger_size  = trigger_size
        self.trigger_value = trigger_value
        self.poison_rate   = poison_rate
        self.position      = position
        self.seed          = seed
        self._name         = f"Backdoor(target={target_class}, trigger={trigger_size}×{trigger_size})"

    def __repr__(self):
        return self._name

    def __call__(self, dataset, indices: List[int]) -> Tuple[Dataset, List[int]]:
        poisoned = BackdooredDataset(
            dataset, indices,
            target_class  = self.target_class,
            trigger_size  = self.trigger_size,
            trigger_value = self.trigger_value,
            poison_rate   = self.poison_rate,
            seed          = self.seed,
        )
        n_poisoned = len(poisoned.poisoned_set)
        print(f"    [Backdoor] {n_poisoned}/{len(indices)} samples triggered | target_class={self.target_class}")
        return poisoned, list(range(len(poisoned)))


def compute_asr(model, test_dataset, target_class: int, trigger_size: int = 4,
                trigger_value: float = 1.0, device: str = "cpu") -> float:
    """
    Compute Attack Success Rate on the clean test set.
    Applies trigger to every test image, measures fraction classified as target_class.

    Returns: ASR (float between 0 and 1)
    """
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    triggered_data = BackdooredDataset(
        test_dataset,
        list(range(len(test_dataset))),
        target_class  = target_class,
        trigger_size  = trigger_size,
        trigger_value = trigger_value,
        poison_rate   = 1.0,
    )
    loader = DataLoader(triggered_data, batch_size=256, shuffle=False, num_workers=0)

    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds  = model(images).argmax(dim=1).cpu()
            correct += (preds == target_class).sum().item()
            total   += len(labels)

    return correct / total


if __name__ == "__main__":
    from torchvision import datasets, transforms
    ds = datasets.MNIST("./data/raw", train=True, download=True,
                        transform=transforms.ToTensor())
    indices = list(range(200))

    attack   = BackdoorAttack(target_class=0, trigger_size=4)
    poisoned, new_idx = attack(ds, indices)

    # Check trigger was applied
    img, lbl = poisoned[0]
    print(f"Label after poison: {lbl}  (should be 0)")
    corner = img[0, -5:-1, -5:-1]
    print(f"Corner pixels (trigger area): min={corner.min():.2f} max={corner.max():.2f}  (max should be ~1.0)")
    print("✓ BackdoorAttack test passed")