import torch

import ntops
from ntops.torch.utils import _cached_make


def matrix_power(input, n, *, out=None):
    """
    Compute the n-th power of a square matrix for n > 0.
    
    Args:
        input: A square matrix of shape (m, m)
        n: Positive integer power
        out: Optional output tensor
    
    Returns:
        The n-th power of the input matrix
    """
    if input.shape[0] != input.shape[1]:
        raise ValueError("matrix_power: input must be a square matrix")
    
    if n <= 0:
        raise ValueError("matrix_power: n must be a positive integer")
    
    # For n == 1, use kernel to copy input to output
    if n == 1:
        if out is None:
            out = torch.empty_like(input)
        kernel = _cached_make(ntops.kernels.matrix_power.premake, n=1, dtype=input.dtype)
        kernel(input, out, 1)
        return out
    
    # Use sequential multiplication to match PyTorch's implementation
    # PyTorch's matrix_power uses sequential multiplication, not fast exponentiation
    
    # Initialize result with input using kernel
    result = torch.empty_like(input)
    kernel = _cached_make(ntops.kernels.matrix_power.premake, n=1, dtype=input.dtype)
    kernel(input, result, 1)  # Copy input to result
    
    # Compute power by repeated multiplication
    # For n > 1, we need n-1 additional multiplications
    for i in range(n - 1):
        # On the last iteration, write directly to out if provided
        if i == n - 2 and out is not None:
            ntops.torch.mm(result, input, out=out)
            return out
        else:
            result = ntops.torch.mm(result, input)
    
    # If out was not used, copy result to out if provided
    if out is not None:
        kernel_copy = _cached_make(ntops.kernels.matrix_power.premake, n=1, dtype=input.dtype)
        kernel_copy(result, out, 1)
        return out
    
    return result

