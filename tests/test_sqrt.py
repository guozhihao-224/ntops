import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_sqrt(shape, dtype, device, rtol, atol):
    # TODO: Test for `float16` later.
    if dtype is torch.float16:
        return
    # sqrt requires non-negative input, so we use abs(randn) or rand
    # Using abs(randn) to ensure we have some zero values and positive values
    input = torch.abs(torch.randn(shape, dtype=dtype, device=device))
    # Also add some small positive values to test edge cases
    input = input + 1e-6

    ninetoothed_output = ntops.torch.sqrt(input)
    reference_output = torch.sqrt(input)

    assert torch.allclose(ninetoothed_output,
                          reference_output,
                          rtol=rtol,
                          atol=atol)


@skip_if_cuda_not_available
def test_sqrt_edge_cases():
    """Test sqrt with edge cases like zero, very small values, and large values."""
    device = "cuda"
    dtype = torch.float32

    # Test with zero
    input_zero = torch.zeros(10, dtype=dtype, device=device)
    assert torch.allclose(ntops.torch.sqrt(input_zero),
                          torch.sqrt(input_zero),
                          rtol=0.001,
                          atol=0.001)

    # Test with very small positive values
    input_small = torch.tensor([1e-8, 1e-6, 1e-4], dtype=dtype, device=device)
    assert torch.allclose(ntops.torch.sqrt(input_small),
                          torch.sqrt(input_small),
                          rtol=0.001,
                          atol=0.001)

    # Test with large values
    input_large = torch.tensor([1e6, 1e8, 1e10], dtype=dtype, device=device)
    assert torch.allclose(ntops.torch.sqrt(input_large),
                          torch.sqrt(input_large),
                          rtol=0.001,
                          atol=0.001)
