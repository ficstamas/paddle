import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from typing import Tuple


def pseudo_inverse(src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Learns a linear map by pseudo inverse
    :param src: Source Representation
    :param tgt: Target Representation
    :return: Transformation matrix, mse
    """
    mapping = torch.linalg.pinv(src) @ tgt
    mse = mean_squared_error(tgt, src @ mapping)
    return mapping, mse
