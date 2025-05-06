import os
import configparser
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

def load_config(config_path="config.ini"):
    """
    Load configuration parameters from the specified .ini file.
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def get_transforms(img_size):
    """
    Returns data augmentation transforms for training and basic transforms for validation.
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

def prepare_dataloaders(config_path="config.ini"):
    """
    Prepare PyTorch DataLoaders for training and validation using configuration file parameters.
    """
    config = load_config(config_path)
    data_path = config["data"]["data_path"]
    batch_size = int(config["model_parameters"]["batch_size"])
    img_size = (int(config["model_parameters"]["img_width"]), int(config["DATA"]["img_height"]))
    val_split = float(config["model_parameters"]["validation_split"])
    seed = int(config["model_parameters"]["random_seed"])

    # Load transforms
    train_transforms, val_transforms = get_transforms(img_size)

    # Load the entire dataset (labels inferred from folder names)
    full_dataset = datasets.ImageFolder(root=data_path, transform=None)

    # Split dataset into training and validation sets
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
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
