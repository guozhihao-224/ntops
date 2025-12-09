import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def application(input_for_min, input_for_max, min_output, max_output):
    dtype = input_for_min.dtype.dtype

    curr_min = ntl.cast(float("inf"), dtype)
    curr_max = ntl.cast(float("-inf"), dtype)

    # 遍历归约维度上的所有 Block
    for i in range(input_for_min.shape[0]):
        input_min_i = ntl.cast(input_for_min[i], dtype)
        input_max_i = ntl.cast(input_for_max[i], dtype)

        # Block 内部归约
        # input_for_min padding 为 inf，不影响 min
        # input_for_max padding 为 -inf，不影响 max
        block_min = ntl.min(input_min_i)
        block_max = ntl.max(input_max_i)

        curr_min = ntl.cast(ntl.minimum(curr_min, block_min), dtype)
        curr_max = ntl.cast(ntl.maximum(curr_max, block_max), dtype)

    # 将结果写入输出
    if min_output.shape[0] > 0:
        min_output[0] = curr_min
        max_output[0] = curr_max


def premake(ndim, dim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype, other=float("inf")),   # input_for_min
        Tensor(ndim, dtype=dtype, other=float("-inf")),  # input_for_max
        Tensor(ndim, dtype=dtype),  # min_output
        Tensor(ndim, dtype=dtype),  # max_output
    )

    return arrangement_, application, tensors
