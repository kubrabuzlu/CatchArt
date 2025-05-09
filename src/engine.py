import wandb
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from utils import save_model


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str) -> Tuple[float, float, float]:
    """
    Performs a single training step over the entire dataloader.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the training data.
        loss_fn (torch.nn.Module): Loss function to evaluate predictions.
        optimizer (torch.optim.Optimizer): Optimizer to update model parameters.
        device (str): Device to perform training on (CPU or CUDA).

    Returns:
        Tuple[float, float]: Average training loss and accuracy for the epoch.
    """
    model.train()
    total_loss, total_acc = 0, 0
    all_preds, all_labels = [], []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Compute loss
        loss = loss_fn(y_pred, y)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Optimizer
        optimizer.step()

        # Compute accuracy
        preds = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        total_acc += (preds == y).sum().item() / len(y_pred)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, avg_acc, f1


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: str) -> Tuple[float, float, float]:
    """
    Evaluates the model on a given test/validation dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test/validation data.
        loss_fn (torch.nn.Module): Loss function to evaluate predictions.
        device (str): Device to perform evaluation on (CPU or CUDA).

    Returns:
        Tuple[float, float]: Average test loss and accuracy.
    """
    model.eval()
    total_loss, total_acc = 0, 0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)

            # Compute loss
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()

            # Compute accuracy
            preds = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            total_acc += (preds == y).sum().item() / len(y)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, avg_acc, f1


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: str,
          checkpoint_path: str,
          early_stopping_patience: int,
          scheduler_patience: int,
          log_wandb: bool = False) -> Dict[str, List]:
    """
    Trains and evaluates the model over multiple epochs.

    Args:
        model (torch.nn.Module): The PyTorch model to train and evaluate.
        train_dataloader (torch.utils.data.DataLoader): Dataloader with training data.
        test_dataloader (torch.utils.data.DataLoader): Dataloader with test/validation data.
        optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.
        loss_fn (torch.nn.Module): Loss function to calculate model error.
        epochs (int): Number of training epochs.
        device (str): Device to train the model on (CPU or CUDA).
        checkpoint_path (str): File path to save the best model.
        early_stopping_patience (int): Number of steps to wait for early stopping.
        scheduler_patience (int): Number of steps to wait for reduce learning rate.
        log_wandb (bool): Specifies whether Wandb will be used.

    Returns:
        Dict[str, List]: A dictionary containing lists of loss and accuracy for both training and test sets.
    """

    best_loss = float('inf')
    patience_counter = 0
    results = {"train_loss": [],
               "train_acc": [],
               "train_f1": [],
               "test_loss": [],
               "test_acc": [],
               "test_f1": []}

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=scheduler_patience,
        verbose=True
    )

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_f1 = train_step(model=model,
                                                     dataloader=train_dataloader,
                                                     loss_fn=loss_fn,
                                                     optimizer=optimizer,
                                                     device=device)
        test_loss, test_acc, test_f1 = test_step(model=model,
                                                 dataloader=test_dataloader,
                                                 loss_fn=loss_fn,
                                                 device=device)

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"train_f1: {train_f1:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"test_f1: {test_f1:.4f}"
        )

        if log_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_f1": test_f1,
            })

        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            save_model(model, checkpoint_path)
            print(f"Best model saved at epoch {epoch + 1} with loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_f1"].append(train_f1)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_f1"].append(test_f1)

        scheduler.step(test_loss)

    return results
