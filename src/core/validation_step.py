from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.types import TrainingMetrics
from src.utils import move_batch_to_device


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
        for batch in dataloader:
            *inputs, targets = move_batch_to_device(batch, device)
            outputs = (
                model(**inputs[0]) if isinstance(inputs[0], dict) else model(*inputs)
            )
            loss = loss_fn(outputs, targets)

            val_loss += loss.item()
            for metric_name, metric_fn in metrics_items:
                val_metrics[metric_name] += metric_fn(outputs, targets).item()

        val_loss /= len(dataloader)
        for metric_name, _ in val_metrics:
            val_metrics[metric_name] /= len(dataloader)

    return val_loss, val_metrics
