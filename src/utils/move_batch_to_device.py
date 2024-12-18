import torch
from typing import Any, Dict, Tuple, Union


TBatch = Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[Dict[str, torch.Tensor], torch.Tensor],
]


def move_batch_to_device(batch: Tuple[Any, ...], device: torch.device) -> TBatch:
    assert (
        len(batch) >= 2
    ), "Batch must contain at least two elements: inputs and targets"

    *inputs, targets = batch
    targets = targets.to(device)

    if isinstance(inputs[0], torch.Tensor):
        inputs = tuple(input.to(device) for input in inputs)
        return inputs + (targets,)
    elif isinstance(inputs[0], dict):
        inputs = {key: value.to(device) for key, value in inputs[0].items()}
        return inputs, targets
    else:
        raise TypeError("Unsupported batch type")
