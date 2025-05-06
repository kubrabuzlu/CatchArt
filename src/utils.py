import torch
from pathlib import Path
from typing import Union
import os

def save_model(model: torch.nn.Module,
               target_dir: Union[str, Path],
               model_name: str) -> None:
    """
    Saves a PyTorch model's state_dict to a specified directory.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        target_dir (str or Path): Directory path to save the model.
        model_name (str): Name of the model file (must end with .pth or .pt).

    Raises:
        AssertionError: If the model_name does not end with '.pt' or '.pth'.
        OSError: If unable to write to the target directory.

    Example:
        save_model(model, "models", "resnet50_best.pth")
    """
    # Ensure valid extension
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "model_name must end with '.pt' or '.pth'"

    # Ensure directory exists
    target_dir = Path(target_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory {target_dir}") from e

    # Full model path
    model_path = target_dir / model_name

    # Save state_dict
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved to {model_path.resolve()}")
