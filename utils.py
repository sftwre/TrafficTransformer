import os
import torch
import torch.nn as nn
from logging import getLogger
from typing import List


def count_trainable_params(model: nn.Module) -> int:
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(state_dict: dict, filename: str):
    """
    Writes model weights to disk.
    Model weights are saved to ./models/<filename>.pth

    Args:
        state_dict: Dictionary with arbitrary model data and training info
        filename: name of model file
    """

    model_dir = Path(f"./models")

    if not model_dir.exists():
        os.mkdir(model_dir)

    torch.save(state_dict, model_dir / filename)

    logger = getLogger(__name__)
    logger.info(f"Model weights saved to -> {model_dir / filename}")
