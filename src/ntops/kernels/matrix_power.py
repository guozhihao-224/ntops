import enum
import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE_M = ninetoothed.block_size()
BLOCK_SIZE_N = ninetoothed.block_size()
BLOCK_SIZE_K = ninetoothed.block_size()


class InputPrecisionVariant(enum.IntEnum):
    TF32 = enum.auto()

    IEEE = enum.auto()


def arrangement(
    input,
    output,
    n,
    block_size_m=None,
    block_size_n=None,
):
    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N

    output_arranged = output.tile((block_size_m, block_size_n))
    input_arranged = input.tile((block_size_m, block_size_n))
    n_arranged = n

    return input_arranged, output_arranged, n_arranged


def application(input, output, n):
    if n == 0:
        # Return identity matrix
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                if i == j:
                    output[i, j] = 1
                else:
                    output[i, j] = 0
        return
    
    if n == 1:
        # Return copy of input
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                output[i, j] = input[i, j]
        return


def premake(
    n,
    dtype=None,
    block_size_m=None,
    block_size_n=None,
):
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=4,
        block_size_n=4,
    )

    tensors = (
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=dtype),
        Tensor(0, dtype=dtype, constexpr=True, value=n),
    )

    return arrangement_, application, tensors
