"""
FL Security Lab — SimpleCNN model for MNIST / CIFAR-10.
Used as the global model throughout the entire 15-week project.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SimpleCNN(nn.Module):
    """
    Lightweight CNN suitable for MNIST (1-channel) and CIFAR-10 (3-channel).
    
    Architecture:
        Conv1(in_ch→32, 3×3) → ReLU → MaxPool(2×2)
        Conv2(32→64, 3×3)    → ReLU → MaxPool(2×2)
        Flatten → FC(64*d*d → 128) → ReLU → Dropout(0.5)
        FC(128 → num_classes)

    Args:
        in_channels  : 1 for MNIST (grayscale), 3 for CIFAR-10 (RGB)
        num_classes  : 10 for both MNIST and CIFAR-10
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # ── Convolutional layers ──────────────────────────
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)

        # ── Fully connected layers ────────────────────────
        # MNIST (28×28) → after 2×pool(2) → 7×7
        # CIFAR-10 (32×32) → after 2×pool(2) → 8×8
        fc_input = 64 * 7 * 7 if in_channels == 1 else 64 * 8 * 8
        self.fc1     = nn.Linear(fc_input, 128)
        self.fc2     = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # → (B, 32, 14, 14) for MNIST
        x = self.pool(F.relu(self.conv2(x)))   # → (B, 64,  7,  7) for MNIST
        x = x.view(x.size(0), -1)              # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)                     # logits (no softmax — CrossEntropyLoss handles it)


def get_model(dataset: str = "mnist") -> SimpleCNN:
    """Factory: return correct model for dataset name."""
    if dataset.lower() in ("mnist", "fashionmnist"):
        return SimpleCNN(in_channels=1, num_classes=10)
    elif dataset.lower() == "cifar10":
        return SimpleCNN(in_channels=3, num_classes=10)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def count_parameters(model: nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    m = get_model("mnist")
    print(f"SimpleCNN (MNIST): {count_parameters(m):,} parameters")
    x = torch.randn(4, 1, 28, 28)  # batch of 4 MNIST images
    out = m(x)
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}  (should be [4, 10])")
    assert out.shape == (4, 10), "Output shape mismatch!"
    print("✓ Model check passed")