import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize("alpha", (0.1, 1.0, 2.0))
@pytest.mark.parametrize("inplace", (False, True))
@pytest.mark.parametrize(*generate_arguments())
def test_elu(shape, alpha, inplace, dtype, device, rtol, atol):
    # # TODO: Test for `float16` later.
    # if dtype is torch.float16:
    #     return
    input = torch.randn(shape, dtype=dtype, device=device)

    # For inplace operations, we need to clone the input for reference
    if inplace:
        input_clone = input.clone()
        ninetoothed_output = ntops.torch.elu(input, alpha=alpha, inplace=inplace)
        reference_output = F.elu(input_clone, alpha=alpha, inplace=inplace)
        # For inplace, the output should be the same object as input
        assert ninetoothed_output is input
    else:
        ninetoothed_output = ntops.torch.elu(input, alpha=alpha, inplace=inplace)
        reference_output = F.elu(input, alpha=alpha, inplace=inplace)

    assert torch.allclose(ninetoothed_output,
                          reference_output,
                          rtol=rtol,
                          atol=atol)


@skip_if_cuda_not_available
def test_elu_edge_cases():
    """Test elu with edge cases like zero, positive, and negative values."""
    device = "cuda"
    dtype = torch.float32
    alpha = 1.0

    # Test with zero
    input_zero = torch.zeros(10, dtype=dtype, device=device)
    assert torch.allclose(ntops.torch.elu(input_zero, alpha=alpha),
                          F.elu(input_zero, alpha=alpha),
                          rtol=0.001,
                          atol=0.001)

    # Test with positive values
    input_positive = torch.tensor([0.1, 1.0, 10.0], dtype=dtype, device=device)
    assert torch.allclose(ntops.torch.elu(input_positive, alpha=alpha),
                          F.elu(input_positive, alpha=alpha),
                          rtol=0.001,
                          atol=0.001)

    # Test with negative values
    input_negative = torch.tensor([-0.1, -1.0, -10.0], dtype=dtype, device=device)
    assert torch.allclose(ntops.torch.elu(input_negative, alpha=alpha),
                          F.elu(input_negative, alpha=alpha),
                          rtol=0.001,
                          atol=0.001)

    # Test with mixed values
    input_mixed = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype, device=device)
    assert torch.allclose(ntops.torch.elu(input_mixed, alpha=alpha),
                          F.elu(input_mixed, alpha=alpha),
                          rtol=0.001,
                          atol=0.001)
