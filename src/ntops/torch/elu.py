import torch

import ntops
from ntops.torch.utils import _cached_make


def elu(input, alpha=1.0, *, inplace=False, out=None):
    if inplace:
        output = input
    else:
        if out is None:
            out = torch.empty_like(input)

        output = out

    kernel = _cached_make(ntops.kernels.elu.premake, input.ndim)

    kernel(input, alpha, output)

    return output
