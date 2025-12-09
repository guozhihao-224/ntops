import torch

import ntops
from ntops.torch.utils import _cached_make


def diagflat(input, offset=0, *, out=None):
    n = input.numel()

    dim = n + abs(offset)
    if out is None:
        out = torch.zeros(dim, dim, dtype=input.dtype, device=input.device)
    else:
        out.zero_()

    if offset >= 0:
        out_view = out[0:n, offset:offset + n]
    else:
        out_view = out[-offset:-offset + n, 0:n]

    kernel = _cached_make(ntops.kernels.diagflat.premake, input.ndim, input.dtype)
    kernel(input, out_view)

    return out
