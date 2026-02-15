from typing import List, Tuple
import numpy as np
from driftguard_ray.config import get_logger
import torch
import torch.nn as nn
logger = get_logger("utils")


def freeze_layer(model: nn.Module, include_names: List[str] = [], exclude: bool = False) -> None:
    """Freeze parameters of layers starting with the specified name.
    
    Args:
        model: The model containing the layers to freeze.
        include_names: Substrings of layer names to freeze.
    """
    for name, param in model.named_parameters():
        if exclude:
            if include_names and all(n not in name for n in include_names):
                param.requires_grad = False
        else:
            if any(n in name for n in include_names) or not include_names:
                param.requires_grad = False

    logger.debug(
        f"Frozen layers {'includes' if not exclude else 'excludes'} {include_names}"
    )

def unfreeze_layer(model: nn.Module, include_names: List[str] = [], exclude: bool = False) -> None:
    """Unfreeze all model parameters.
    
    Args:
        model: The model to unfreeze parameters for.
        include_names: Substrings of layer names to unfreeze.
        exclude: If True, unfreeze layers not containing the include_names.
    """
    for name, param in model.named_parameters():
        if exclude:
            if include_names and all(n not in name for n in include_names):
                param.requires_grad = True
        else:
            if any(n in name for n in include_names) or not include_names:
                param.requires_grad = True

    logger.debug(
        f"Unfrozen layers {'includes' if not exclude else 'excludes'} '{include_names}'"
    )

def get_total_params(model: nn.Module) -> int:
    """Get total parameter count of the model.
    
    Args:
        model: The model to analyze.
    """
    total = sum(p.numel() for p in model.parameters())
    return total

def get_trainable_params(model: nn.Module) -> int:
    """Get the count of trainable parameters in the model.
    
    Args:
        model: The model to analyze.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable