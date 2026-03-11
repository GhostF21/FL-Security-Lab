"""Download MNIST and CIFAR-10 into data/raw/ directory."""
import torchvision
import os

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

print("Downloading MNIST...")
mnist = torchvision.datasets.MNIST(root=RAW_DIR, train=True, download=True)
mnist_test = torchvision.datasets.MNIST(root=RAW_DIR, train=False, download=True)
print(f"  MNIST train: {len(mnist)} samples")
print(f"  MNIST test:  {len(mnist_test)} samples")
print(f"  Classes: {mnist.classes}")

print("Downloading CIFAR-10...")
cifar = torchvision.datasets.CIFAR10(root=RAW_DIR, train=True, download=True)
cifar_test = torchvision.datasets.CIFAR10(root=RAW_DIR, train=False, download=True)
print(f"  CIFAR-10 train: {len(cifar)} samples")
print(f"  CIFAR-10 test:  {len(cifar_test)} samples")

print("All datasets downloaded successfully!")
