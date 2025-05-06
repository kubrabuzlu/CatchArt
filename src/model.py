import torch
from torch import nn
import torchvision.models as models


def create_resnet50_painter_model(num_classes: int) -> nn.Module:
    """
    Returns a custom model for painter classification based on ResNet50.

    Args:
        num_classes (int): Number of output classes (e.g. number of painters).
    Returns:
        model (nn.Module): Custom ResNet50 model.
    """
    base_model = models.resnet50(pretrained=True)

    for param in base_model.parameters():
        param.requires_grad = True

    in_features = base_model.fc.in_features

    base_model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),

        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),

        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),

        nn.Linear(64, num_classes)
    )

    return base_model
