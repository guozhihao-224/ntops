import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, alpha, output):
    if input.dtype == ntl.float16:
        input_fp32 = ntl.cast(input, ntl.float32)
        alpha_fp32 = ntl.cast(alpha, ntl.float32)
        result_fp32 = ntl.where(input_fp32 > 0, input_fp32,
                               alpha_fp32 * (ntl.exp(input_fp32) - 1))
        output = ntl.cast(result_fp32, ntl.float16)
    elif input.dtype == ntl.bfloat16:
        input_fp32 = ntl.cast(input, ntl.float32)
        alpha_fp32 = ntl.cast(alpha, ntl.float32)
        result_fp32 = ntl.where(input_fp32 > 0, input_fp32,
                               alpha_fp32 * (ntl.exp(input_fp32) - 1))
        output = ntl.cast(result_fp32, ntl.bfloat16)
    else:
        output = ntl.where(input > 0, input,
                          alpha * (ntl.exp(input) - 1))  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(0, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
