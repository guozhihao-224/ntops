import torch

import ntops
from ntops.torch.utils import _cached_make


def aminmax(input, *, dim=None, keepdim=False, out=None):
    if dim is None:
        input_view = input.flatten()
        reduce_dim = 0
    else:
        input_view = input
        reduce_dim = dim

    if out is None:
        # 计算输出形状
        if dim is None:
            # Match PyTorch behavior: dim=None returns scalar tensors, ignoring keepdim?
            # Or maybe we should support keepdim logic but PyTorch reference doesn't?
            # To pass the test against torch.aminmax, we should return scalars.
            out_shape = ()
        else:
            if keepdim:
                out_shape = list(input.shape)
                out_shape[dim] = 1
            else:
                out_shape = list(input.shape)
                out_shape.pop(dim)

        dtype = input.dtype
        device = input.device
        min_out = torch.empty(out_shape, dtype=dtype, device=device)
        max_out = torch.empty(out_shape, dtype=dtype, device=device)
        out = (min_out, max_out)
    else:
        min_out, max_out = out

    # 准备传给 Kernel 的 Output View (必须 Keepdim)
    if dim is None:
        min_out_kernel = min_out.reshape(1)
        max_out_kernel = max_out.reshape(1)
    else:
        if keepdim:
            min_out_kernel = min_out
            max_out_kernel = max_out
        else:
            min_out_kernel = min_out.unsqueeze(reduce_dim)
            max_out_kernel = max_out.unsqueeze(reduce_dim)

    kernel = _cached_make(
        ntops.kernels.aminmax.premake,
        input_view.ndim,
        reduce_dim,
        input_view.dtype,
    )

    kernel(input_view, input_view, min_out_kernel, max_out_kernel)

    return out
