import torch
import pytest
import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("size", [
    3,
    64,
    128,
])
@pytest.mark.parametrize("offset", [0, 1, -1, 10, -10])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_diagflat_main_diagonal_values(size, offset, dtype):
    device = "cuda"

    input_tensor = torch.randn(size, dtype=dtype, device=device)

    ntops_output = ntops.torch.diagflat(input_tensor, offset=offset)
    reference_output = torch.diagflat(input_tensor, offset=offset)

    assert ntops_output.shape == reference_output.shape, \
        f"Shape mismatch: ntops {ntops_output.shape} vs ref {reference_output.shape}"

    if not torch.equal(ntops_output, reference_output):
        max_diff = (ntops_output - reference_output).abs().max()
        assert max_diff == 0, f"Value mismatch. Max diff: {max_diff}"

    diagonal_elements = torch.diagonal(ntops_output, offset=offset)
    assert torch.equal(diagonal_elements,
                       input_tensor), "Diagonal elements do not match input"


@skip_if_cuda_not_available
def test_diagflat_input_flattening():
    device = "cuda"
    dtype = torch.float32

    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                dtype=dtype,
                                device=device)

    ntops_output = ntops.torch.diagflat(input_tensor, 0)
    reference_output = torch.diagflat(input_tensor, 0)

    assert torch.equal(ntops_output, reference_output)
    assert ntops_output.shape == (6, 6)