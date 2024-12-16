from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.types.training_metrics import TrainingMetrics


def validation_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    metrics: Optional[TrainingMetrics],
    device: Optional[torch.device],
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_items = metrics.items() if metrics is not None else []
    val_metrics = {metric_name: 0.0 for metric_name, _ in metrics_items}
    val_loss = 0.0

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_fn(output, y)

            val_loss += loss.item()
            for metric_name, metric_fn in metrics_items:
                val_metrics[metric_name] += metric_fn(output, y).item()

        val_loss /= len(dataloader)
        for metric_name, _ in val_metrics:
            val_metrics[metric_name] /= len(dataloader)

    return val_loss, val_metrics
