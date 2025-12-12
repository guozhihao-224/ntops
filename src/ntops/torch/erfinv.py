import torch

import ntops
from ntops.torch.utils import _cached_make


def erfinv(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.erfinv.premake, input.ndim)

    kernel(input, out)

    return out

