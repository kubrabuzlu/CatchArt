import torch
from pathlib import Path
from typing import Union
import os


def save_model(model: torch.nn.Module,
               target_dir: Union[str, Path]) -> None:
    """
    Saves a PyTorch model's state_dict to a specified directory.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        target_dir (str or Path): Directory path to save the model.

    Raises:
        AssertionError: If the model_name does not end with '.pt' or '.pth'.
        OSError: If unable to write to the target directory.
    """
    # Ensure valid extension
    assert target_dir.endswith(".pth") or target_dir.endswith(".pt"), \
        "model_name must end with '.pt' or '.pth'"

    # Ensure directory exists
    target_dir = Path(target_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory {target_dir}") from e

    # Save state_dict
    torch.save(model.state_dict(), target_dir)
    print(f"[INFO] Model saved to {target_dir.resolve()}")
