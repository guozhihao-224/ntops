import torch
import ntops
from ntops.torch.utils import _cached_make


def diagflat(input, offset=0):
    n = input.numel()

    dim = n + abs(offset)

    output = torch.zeros((dim, dim), dtype=input.dtype, device=input.device)

    kernel = _cached_make(ntops.kernels.diagflat.premake, input.ndim)

    kernel(input, output, offset)

    return output
