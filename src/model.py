import torch
from torch import nn
import torchvision.models as models


def create_painter_model(model_name: str,
                         num_classes: int,
                         pretrained: bool = True,
                         fine_tune: bool = True) -> nn.Module:
    """
    Creates a customizable image classification model for painter prediction.

    Args:
        model_name (str): Name of the base model to use.
        num_classes (int): Number of output classes (e.g. number of painters).
        pretrained (bool): Whether to load ImageNet pretrained weights.
        fine_tune (bool): Whether to fine-tune the entire model or freeze base layers.
    Returns:
        nn.Module: A model ready for fine-tuning or feature extraction.
    """
    if model_name not in ["resnet50", "densenet121", "vgg16", "efficientnet_b0"]:
        raise ValueError(f"Invalid model name: {model_name}")

    if model_name == "resnet50":
        base_model = models.resnet50(pretrained=pretrained)

        if not fine_tune:
            for param in base_model.parameters():
                param.requires_grad = False

        in_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
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

    elif model_name == "densenet121":
        base_model = models.densenet121(pretrained=pretrained)

        if not fine_tune:
            for param in base_model.parameters():
                param.requires_grad = False

        in_features = base_model.classifier.in_features
        base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )

    elif model_name == "vgg16":
        base_model = models.vgg16(pretrained=pretrained)

        if not fine_tune:
            for param in base_model.features.parameters():
                param.requires_grad = False

        in_features = base_model.classifier[6].in_features
        base_model.classifier[6] = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        base_model = models.efficientnet_b0(pretrained=pretrained)

        if not fine_tune:
            for param in base_model.features.parameters():
                param.requires_grad = False

        in_features = base_model.classifier[1].in_features
        base_model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Model {model_name} is not supported.")

    return base_model
