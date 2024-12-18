from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.types import TrainingMetrics
from src.utils import move_batch_to_device


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device],
    metrics: Optional[TrainingMetrics],
    train_progress: Optional[tqdm],
) -> Tuple[float, Dict[str, float]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if train_progress is None:
        train_progress = tqdm(dataloader, desc="Training...")
    metrics_items = metrics.items() if metrics is not None else []
    train_metrics: Dict[str, float] = {key: 0.0 for key, _ in metrics_items}
    train_loss = 0.0

    model.to(device)
    model.train()
    for batch_idx, batch in enumerate(train_progress):
        # Forward pass
        *inputs, targets = move_batch_to_device(batch, device)
        outputs = model(**inputs[0]) if isinstance(inputs[0], dict) else model(*inputs)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        # Update metrics
        train_loss += loss.item()
        for key, metric_fn in metrics_items:
            train_metrics[key] += metric_fn(outputs, targets).item()

        # Update progress bar
        train_progress.set_postfix(
            {
                "Train Loss": train_loss / (batch_idx + 1),
                **average_metrics(train_metrics, batch_idx + 1),
            }
        )

    train_loss_avg = train_loss / len(dataloader)
    metrics_avg = average_metrics(train_metrics, len(dataloader))

    return train_loss_avg, metrics_avg


def average_metrics(training_metrics: Dict[str, float], n: int) -> Dict[str, float]:
    return {metric_name: value / n for metric_name, value in training_metrics.items()}


__all__ = ["train_step"]
