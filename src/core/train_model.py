from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.core import train_step, validation_step
from src.types import TrainingHistory, TrainingMetrics
from src.utils import EarlyStopping


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    metrics: Optional[TrainingMetrics],
    early_stopping: Optional[EarlyStopping],
    device: Optional[torch.device],
    verbose: Optional[bool] = False,
) -> TrainingHistory:
    """
    General training function for PyTorch models

    Args:
    - model: PyTorch model to train
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - optimizer: Optimization algorithm
    - loss_fn: Loss function
    - epochs: Number of training epochs
    - device: Computing device (CPU/GPU)
    - scheduler: Learning rate scheduler
    - metrics: Dictionary of metrics to evaluate
    - early_stopping: Early stopping callback
    - verbose: Whether to print training progress

    Returns:
    - Dictionary containing training history
    """
    # Set default device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(f"[INFO] Setting device to: {device}")

    metrics_keys = metrics.keys() if metrics is not None else []
    history = TrainingHistory(
        train_metrics={key: [] for key in metrics_keys},
        val_metrics={key: [] for key in metrics_keys},
    )

    model.to(device)

    # Training loop
    for epoch in range(epochs):
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_metrics = train_step(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            metrics=metrics,
            train_progress=train_progress,
        )
        val_loss, val_metrics = validation_step(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            metrics=metrics,
        )

        # Save loss and metrics
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        for metric_name in metrics_keys:
            assert (
                metric_name in train_metrics
            ), f"Metric {metric_name} not found in train_metrics"
            assert (
                metric_name in val_metrics
            ), f"Metric {metric_name} not found in val_metrics"
            history.train_metrics[metric_name].append(train_metrics[metric_name])
            history.val_metrics[metric_name].append(val_metrics[metric_name])

        # Learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        # Early stopping
        if early_stopping is not None:
            early_stopping(model, val_loss)
            if early_stopping.early_stop:
                if verbose:
                    print("[INFO] Early stopping triggered")

    return history
