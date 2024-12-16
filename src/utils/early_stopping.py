from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EarlyStopping:
    patience: int
    counter: int = 0
    best_score: float = float("inf")
    early_stop: bool = False
    delta: float = 0.0
    verbose: bool = False

    def __call__(self, model: nn.Module, val_loss: float):
        if val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(model, val_loss)
            self.counter = 0

    def save_checkpoint(self, model: nn.Module, val_loss: float):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), "checkpoint.pt")
        self.val_loss_min = val_loss
        self.best_score = val_loss


__all__ = ["EarlyStopping"]
