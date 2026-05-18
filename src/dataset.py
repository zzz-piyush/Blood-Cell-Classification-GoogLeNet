"""
Dataset Pipeline for Blood Cell Classification
================================================
Handles downloading from Kaggle, loading, augmentation, and splitting.
Dataset: paultimothymooney/blood-cells
Classes: Eosinophil, Lymphocyte, Monocyte, Neutrophil
"""

import os
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

import kagglehub


# Class names in alphabetical order (matching folder names)
CLASS_NAMES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]


def download_dataset(dataset_name, data_dir):
    """
    Download the Blood Cell Images dataset from Kaggle.
    Returns the path to the dataset root.
    """
    print(f"Downloading dataset: {dataset_name}")
    kaggle_path = kagglehub.dataset_download(dataset_name)
    print(f"  Kaggle cache path: {kaggle_path}")

    # The dataset structure is:
    #   <kaggle_path>/dataset2/dataset2/images/TRAIN/...
    #   <kaggle_path>/dataset2/dataset2/images/TEST/...
    # OR:
    #   <kaggle_path>/dataset-master/dataset-master/images/TRAIN/...
    #   <kaggle_path>/dataset-master/dataset-master/images/TEST/...
    # We need to find the actual image directories

    kaggle_path = Path(kaggle_path)

    # Search for the TRAIN directory recursively
    train_dirs = list(kaggle_path.rglob("TRAIN"))
    test_dirs = list(kaggle_path.rglob("TEST"))

    if not train_dirs:
        # Try lowercase
        train_dirs = list(kaggle_path.rglob("train"))
        test_dirs = list(kaggle_path.rglob("test"))

    if not train_dirs:
        # Try looking for class folders directly
        print(f"  Contents of {kaggle_path}:")
        for item in kaggle_path.iterdir():
            print(f"    {item.name} ({'dir' if item.is_dir() else 'file'})")
        raise FileNotFoundError(
            f"Could not find TRAIN directory in {kaggle_path}. "
            "Please check dataset structure."
        )

    # Use the first found paths
    train_src = train_dirs[0]
    test_src = test_dirs[0] if test_dirs else None

    print(f"  Found TRAIN at: {train_src}")
    if test_src:
        print(f"  Found TEST at:  {test_src}")

    # Create local data directory structure
    data_path = Path(data_dir)
    local_train = data_path / "train"
    local_test = data_path / "test"

    if local_train.exists() and any(local_train.iterdir()):
        print("  Local data already exists, skipping copy.")
        return str(data_path)

    # Copy to local data dir
    print("  Copying to local data directory...")
    if local_train.exists():
        shutil.rmtree(local_train)
    shutil.copytree(str(train_src), str(local_train))

    if test_src and test_src.exists():
        if local_test.exists():
            shutil.rmtree(local_test)
        shutil.copytree(str(test_src), str(local_test))

    # Count images
    for split_name, split_path in [("Train", local_train), ("Test", local_test)]:
        if split_path.exists():
            total = sum(1 for _ in split_path.rglob("*.jp*g"))
            total += sum(1 for _ in split_path.rglob("*.png"))
            classes = [d.name for d in split_path.iterdir() if d.is_dir()]
            print(f"  {split_name}: {total} images, classes: {classes}")

    return str(data_path)


def get_transforms(image_size=224, is_training=True):
    """Get data transforms for training or evaluation."""
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])


def create_dataloaders(data_dir, image_size=224, batch_size=32,
                       val_split=0.15, num_workers=4, seed=42):
    """
    Create train, validation, and test DataLoaders.

    The BCCD dataset has pre-split TRAIN and TEST folders.
    We further split TRAIN into train/val using stratified sampling.
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    test_dir = data_path / "test"

    # Load full training set with training transforms (we'll apply val transforms separately)
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)

    # Load dataset to get labels for stratified split
    full_train_dataset = datasets.ImageFolder(str(train_dir))
    targets = np.array(full_train_dataset.targets)
    num_classes = len(full_train_dataset.classes)

    # Stratified train/val split
    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []

    for class_idx in range(num_classes):
        class_indices = np.where(targets == class_idx)[0]
        rng.shuffle(class_indices)
        n_val = int(len(class_indices) * val_split)
        val_indices.extend(class_indices[:n_val])
        train_indices.extend(class_indices[n_val:])

    # Create separate datasets with appropriate transforms
    train_dataset_full = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_dataset_full = datasets.ImageFolder(str(train_dir), transform=val_transform)

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    # Class distribution for weighted sampler
    train_targets = targets[train_indices]
    class_counts = Counter(train_targets)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Test dataset
    test_dataset = datasets.ImageFolder(str(test_dir), transform=val_transform)

    # Print dataset info
    print("\n" + "=" * 50)
    print("  DATASET SUMMARY")
    print("=" * 50)
    print(f"  Classes:     {full_train_dataset.classes}")
    print(f"  Train:       {len(train_dataset)} images")
    print(f"  Validation:  {len(val_dataset)} images")
    print(f"  Test:        {len(test_dataset)} images")
    print(f"  Train distribution: {dict(Counter(train_targets))}")
    print("=" * 50 + "\n")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    class_names = full_train_dataset.classes

    return train_loader, val_loader, test_loader, class_names
