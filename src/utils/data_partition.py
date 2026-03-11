"""Non-IID Dirichlet partitioner for FL experiments."""
import numpy as np
from torch.utils.data import Subset

def dirichlet_partition(dataset, num_clients: int, alpha: float = 0.5,
                         seed: int = 42):
    """
    Partition a dataset into num_clients non-IID subsets using
    a Dirichlet(alpha) distribution over classes.

    Args:
        dataset: PyTorch dataset with .targets attribute
        num_clients: number of FL clients
        alpha: concentration parameter (smaller = more heterogeneous)
        seed: random seed for reproducibility

    Returns:
        List of num_clients Subset objects
    """
    np.random.seed(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    client_indices = [[] for _ in range(num_clients)]

    # For each class, distribute samples to clients using Dirichlet
    for class_id in range(num_classes):
        class_idx = np.where(targets == class_id)[0]
        np.random.shuffle(class_idx)
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Compute split points
        proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        splits = np.split(class_idx, proportions)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    return [Subset(dataset, indices) for indices in client_indices]


if __name__ == "__main__":
    # Quick test
    import torchvision
    mnist = torchvision.datasets.MNIST("data/raw", train=True, download=False)
    partitions = dirichlet_partition(mnist, num_clients=10, alpha=0.5)
    print("Partition sizes:", [len(p) for p in partitions])
    # Expected: approximately similar sizes, but class distributions differ
    print("Sum:", sum(len(p) for p in partitions), "(should equal 60000)")
