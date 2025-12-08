import functools
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    if input.dtype == ntl.float16:
        sqrt_result = ntl.sqrt(ntl.cast(input, ntl.float32))
        output = ntl.cast(sqrt_result, ntl.float16)
    elif input.dtype == ntl.bfloat16:
        sqrt_result = ntl.sqrt(ntl.cast(input, ntl.float32))
        output = ntl.cast(sqrt_result, ntl.bfloat16)
    else:
        output = ntl.sqrt(input)


def premake(ndim, dtype=None, block_size=None):
    if block_size is None:
        block_size = 256

    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors