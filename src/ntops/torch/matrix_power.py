import torch

import ntops


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
    
    # For n == 1, just return a copy of input
    if n == 1:
        if out is None:
            return input.clone()
        else:
            out.copy_(input)
            return out
    
    # Use sequential multiplication to match PyTorch's implementation
    # PyTorch's matrix_power uses sequential multiplication, not fast exponentiation
    result = input.clone()
    
    # Compute power by repeated multiplication
    # For n > 1, we need n-1 additional multiplications
    for _ in range(n - 1):
        temp = ntops.torch.mm(result, input)
        result = temp
    
    if out is not None:
        out.copy_(result)
        return out
    
    return result

