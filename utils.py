# torch imports
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

# stl imports
import os
from logging import getLogger

logger = getLogger(__name__)

from pathlib import Path


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
    Writes model weights to ./models/filename.pth

    Args:
        state_dict: Dictionary with model state_dict and training info.
        filename: name of model file
    """

    model_dir = Path(f"./models")

    if not model_dir.exists():
        os.mkdir(model_dir)

    torch.save(state_dict, model_dir / filename)

    logger.info(f"Model weights saved to -> {model_dir / filename}")


def get_lr_scheduler(
    type: str,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    steps_per_epoch: int = None,
):
    """
    Returns a learning rate scheduler based on the specified type.

    Args:
        type: Type of learning rate scheduler (e.g., 'plateau', 'one_cycle')
        optimizer: Optimizer for which to create the scheduler
        num_epochs: Number of epochs for training

    Returns:
        Learning rate scheduler
    """

    match type:
        case "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6
            )
            return scheduler
        case "one_cycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=optimizer.param_groups[0]["lr"],
                epochs=num_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                anneal_strategy="cos",
                div_factor=10,
            )
            return scheduler
        case "":
            return None
        case _:
            message = (
                f"Unknown scheduler type: {type}. "
                "Available options are: 'plateau', 'one_cycle'."
            )
            logger.error(message)
            raise ValueError(message)


def generate_submission(predictions, video_ids, output_path="submission.csv"):
    """
    Generates the submission file in the required format.
    """
    with open(output_path, "w") as f:
        f.write("id,score\n")
        for vid, score in zip(video_ids, predictions):
            f.write(f"{vid},{score[0]:.4f}\n")

    logger.info(f"Submission saved to -> {output_path}")


def get_model_device(model: nn.Module) -> torch.device:
    """
    Returns the device on which the model is located.
    Args:
        model: PyTorch model
    Returns:
        Torch device ('cuda', 'cpu')
    """
    return next(model.parameters()).device
