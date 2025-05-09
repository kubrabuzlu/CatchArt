import torch
from pathlib import Path
from typing import Union


def save_model(model: torch.nn.Module,
               target_path: Union[str, Path]) -> None:
    """
    Saves a PyTorch model's state_dict to the specified file path.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        target_path (str or Path): Full file path where the model will be saved.
                                   Example: 'models/best_model.pth'

    Raises:
        AssertionError: If the file path does not end with '.pt' or '.pth'.
        OSError: If unable to create the target directory.
    """
    # Ensure correct extension
    assert str(target_path).endswith((".pt", ".pth")), \
        "target_path must end with '.pt' or '.pth'"

    # Convert to Path object
    target_path = Path(target_path)

    # Create parent directory if needed
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory {target_path.parent}") from e

    # Save model
    torch.save(model.state_dict(), target_path)
    print(f"[INFO] Model saved to {target_path.resolve()}")

