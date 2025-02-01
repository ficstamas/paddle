import numpy as np
import torch
from typing import Tuple


def pseudo_inverse(src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Learns a linear map by pseudo inverse
    :param src: Source Representation
    :param tgt: Target Representation
    :return: Transformation matrix, mse
    """
    mapping = torch.linalg.pinv(src) @ tgt
    mse = torch.mean(torch.pow(tgt - src @ mapping, 2))
    return mapping, mse
