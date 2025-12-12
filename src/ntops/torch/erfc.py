import torch

import ntops
from ntops.torch.utils import _cached_make


def erfc(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.erfc.premake, input.ndim, 2)

    kernel(input, out)

    return out

