from torch import nn

from typing import Dict


TrainingMetrics = Dict[str, nn.Module]


__all__ = ["TrainingMetrics"]
