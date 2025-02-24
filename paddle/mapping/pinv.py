import torch
from typing import Tuple


def pseudo_inverse(src: torch.Tensor, tgt: torch.Tensor, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a linear map by pseudo inverse
    :param src: Source Representation
    :param tgt: Target Representation
    :param normalize: Normalize vectors
    :return: Transformation matrix, mse
    """

    if normalize:
        src = torch.nn.functional.normalize(src)
        tgt = torch.nn.functional.normalize(tgt)

    mapping = torch.linalg.pinv(src) @ tgt
    mse = torch.mean(torch.pow(tgt - src @ mapping, 2))
    return mapping, mse
