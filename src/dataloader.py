import torch
from torchvision import datasets, transforms
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, random_split

from typing import Tuple


def get_transforms(img_size: Tuple) -> Tuple[Compose, Compose]:
    """
        Returns the image transformations for training and validation datasets.

        Args:
            img_size (Tuple): The target size to which images will be resized.
                            Example: (224, 224).

        Returns:
            Tuple[Compose, Compose]:
                - train_transforms: Data augmentation pipeline for training set.
                - val_transforms: Basic preprocessing for validation set.
    """
    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    return train_transforms, val_transforms


def prepare_dataloaders( data_path: str,
                         batch_size: int,
                         img_size: Tuple,
                         valid_split: float,
                         seed: int) -> Tuple[DataLoader, DataLoader]:
    """
    Prepares PyTorch DataLoaders for training and validation using an image folder dataset.

    This function reads the dataset from a given directory where subfolders represent class labels,
    applies appropriate transformations to training and validation splits, and returns DataLoaders.

    Args:
        data_path (str): Path to the root directory containing image folders.
                         Each subfolder represents a class.
        batch_size (int): Number of samples per batch to load.
        img_size (Tuple): Image resize size (e.g., (224,224)).
        valid_split (float): Fraction of dataset to reserve for validation (e.g., 0.2 for 20%).
        seed (int): Random seed for reproducible data splitting.

    Returns:
        Tuple[DataLoader, DataLoader]:
            - train_loader: DataLoader for the training dataset.
            - val_loader: DataLoader for the validation dataset.
    """
    # Load transforms
    train_transforms, val_transforms = get_transforms(img_size)

    # Load the entire dataset (labels inferred from folder names)
    full_dataset = datasets.ImageFolder(root=data_path, transform=None)

    # Split dataset into training and validation sets
    total_size = len(full_dataset)
    val_size = int(valid_split * total_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Apply transforms to each subset
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
