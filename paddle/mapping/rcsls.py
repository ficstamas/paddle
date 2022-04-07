import torch
from typing import Tuple


def _getknn_pt(sc, x, y, top_k):
    top_vals, sidx = sc.topk(top_k, dim=1, largest=True)
    ytopk = y[sidx.flatten(), :].reshape(sidx.shape[0], sidx.shape[1], y.shape[1])
    f = top_vals.sum()
    df = torch.sum(ytopk, dim=1).T @ x
    return f / top_k, df / top_k


def _rcsls_pt(src, tgt, Z_src, Z_tgt, R, knn=10):
    src_modded = src @ R.T
    f = 2 * torch.sum(src_modded * tgt)
    df = 2 * tgt.T @ src
    fk0, dfk0 = _getknn_pt(src_modded @ Z_tgt.T, src, Z_tgt, knn)
    fk1, dfk1 = _getknn_pt(((Z_src @ R.T) @ tgt.T).T, tgt, Z_src, knn)
    f = f - fk0 - fk1
    df = df - dfk0 - dfk1.T
    return -f / src.shape[0], -df / src.shape[0]


def _calculate_rcsls_pt(R, src, tgt, spectral, batchsize=0, niter=10, knn=10, maxneg=50000, lr=1.0, lr_stop=1e-4,
                        seed=42, verbose=False):
    torch.manual_seed(seed)
    fold, Rold = 0, []
    for it in range(0, niter+1):
        if lr < lr_stop:
            break

        indices = list(range(tgt.shape[0]))
        if len(indices) > batchsize > 0:
            indices = torch.randperm(src.shape[0])[:batchsize]
        f, df = _rcsls_pt(src[indices, :], tgt[indices, :], src[:maxneg, :], tgt[:maxneg, :], R, knn)
        R -= lr * df
        if spectral:
            U, s, V = torch.linalg.svd(R)
            s[s > 1] = 1
            s[s < 0] = 0
            R = U @ (torch.diag(s) @ V)
        if verbose:
            print("[it={}] (lr={:.4f}) f = {:.4f}".format(it, lr, f))

        if f > fold and it > 0 and batchsize<=0:
            lr /= 2
            f, R = fold, Rold
        fold, Rold = f, R
    return f, R


def rcsls(src: torch.Tensor, tgt: torch.Tensor, spectral: bool = False, batchsize: int = 5000, maxneg: int = 100000,
          niter: int = 20, seed: int = 42, lr_init: float = 1.0, lr_stop: float = 1e-4, verbose: bool = False)\
        -> Tuple[torch.Tensor, float]:
    """
    Applies the RCSLS algorithm to learn a mapping from the source representation to the target. The learning rate is
    going to be halved with every iteration
    :param src: Source Representation
    :param tgt: Target Representation
    :param spectral: With unit ball spectral norm
    :param batchsize: Batch size
    :param maxneg: Number of negatives for RCSLS
    :param niter: Number of iterations
    :param seed: Random seed
    :param lr_init: Starting Learning Rate
    :param lr_stop: Stopping Learning Rate
    :param verbose: Print state
    :return: Transformation matrix, rcsls error
    """
    src = torch.nn.functional.normalize(src)
    tgt = torch.nn.functional.normalize(tgt)

    U, _, V = torch.linalg.svd(tgt.T @ src)
    f, target_trafo = _calculate_rcsls_pt(U @ V, src, tgt, spectral=spectral, batchsize=batchsize, maxneg=maxneg,
                                          niter=niter, verbose=verbose, seed=seed, lr=lr_init, lr_stop=lr_stop)
    return target_trafo, f
