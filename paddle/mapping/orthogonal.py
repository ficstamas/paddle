import torch
from typing import Tuple


def orthogonal(src: torch.Tensor, tgt: torch.Tensor, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a orthogonal map
    :param src: Source Representation
    :param tgt: Target Representation
    :param normalize: Normalize vectors
    :return: Transformation matrix, mse
    """

    if normalize:
        src = torch.nn.functional.normalize(src)
        tgt = torch.nn.functional.normalize(tgt)

    M = src.T @ tgt
    U, _, Vt = torch.linalg.svd(M)

    mapping = U @ Vt
    mse = torch.mean(torch.pow(tgt - src @ mapping, 2))
    return mapping, mse
