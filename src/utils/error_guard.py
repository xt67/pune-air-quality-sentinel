"""Error guard utilities for input validation and error handling."""
import functools
import logging
from typing import Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class GraphConstructionError(Exception):
    """Raised when graph construction fails."""
    pass


def validate_no_nan(tensor: torch.Tensor, name: str = "tensor") -> None:
    """
    Validate that tensor contains no NaN values.
    
    Args:
        tensor: PyTorch tensor to validate
        name: Name for error message
        
    Raises:
        DataValidationError: If tensor contains NaN
    """
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        nan_pct = (nan_count / tensor.numel()) * 100
        raise DataValidationError(
            f"{name} contains {nan_count} NaN values ({nan_pct:.2f}%), "
            f"shape: {tensor.shape}"
        )


def validate_aqi_range(
    values: Union[np.ndarray, torch.Tensor], 
    name: str = "AQI"
) -> Union[np.ndarray, torch.Tensor]:
    """
    Validate and clip AQI values to [0, 500] range.
    
    Args:
        values: Array or tensor of AQI values
        name: Name for logging
        
    Returns:
        Clipped values
    """
    if isinstance(values, torch.Tensor):
        below_zero = (values < 0).sum().item()
        above_500 = (values > 500).sum().item()
        
        if below_zero > 0 or above_500 > 0:
            logger.warning(
                f"{name}: Clipping {below_zero} values below 0 and "
                f"{above_500} values above 500"
            )
        
        return torch.clamp(values, 0, 500)
    else:
        below_zero = (values < 0).sum()
        above_500 = (values > 500).sum()
        
        if below_zero > 0 or above_500 > 0:
            logger.warning(
                f"{name}: Clipping {below_zero} values below 0 and "
                f"{above_500} values above 500"
            )
        
        return np.clip(values, 0, 500)


def handle_oom(func):
    """
    Decorator to handle GPU out of memory errors.
    
    Catches CUDA OOM, clears cache, and re-raises with helpful message.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            logger.error(
                f"GPU OOM in {func.__name__}. "
                f"Try reducing batch_size or seq_len. "
                f"Original error: {e}"
            )
            raise
    return wrapper
