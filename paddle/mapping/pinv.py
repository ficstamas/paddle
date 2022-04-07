import torch


def pseudo_inverse(x, y):
    return torch.linalg.pinv(x) @ y
