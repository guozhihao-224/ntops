import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


# Tolerance map matching the specified precision requirements
_TOLERANCE_MAP = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-2},
    torch.float32: {"atol": 1e-5, "rtol": 1e-4},
    torch.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}


def generate_arguments():
    arguments = []
    
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        device = "cuda"
        
        import random
        size = random.randint(2, 64)  # Square matrix size
        
        # Get tolerance from map
        tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-3, "rtol": 1e-2})
        atol = tolerance["atol"]
        rtol = tolerance["rtol"]
        
        # Test different positive powers
        for n in [1, 2, 3]:
            arguments.append((size, n, dtype, device, rtol, atol))
    
    return "size, n, dtype, device, rtol, atol", arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_matrix_power(size, n, dtype, device, rtol, atol):
    input = torch.randn((size, size), dtype=dtype, device=device)
    
    ninetoothed_output = ntops.torch.matrix_power(input, n)
    reference_output = torch.linalg.matrix_power(input, n)
    
    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)

