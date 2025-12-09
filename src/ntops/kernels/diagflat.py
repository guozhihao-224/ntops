import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(
    input,
    output,
    block_size=None,
):
    if block_size is None:
        block_size = ninetoothed.block_size()

    output_arranged = output.tile((block_size, -1))
    input_arranged = input.flatten().tile((block_size, ))

    return input_arranged, output_arranged


# output.shape = (block_size,)
def application(input, output):
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            # row index: output[i, j].offsets(-2)
            # col index: output[i, j].offsets(-1)
            # Logic: col = row + offset  =>  col - row == offset
            output[i, j] = ntl.where(
                output[i, j].offsets(-2) == output[i, j].offsets(-1),
                input[i],
                0,
            )


def premake(
    ndim,
    dtype=None,
    block_size=None,
):
    arrangement_ = functools.partial(
        arrangement,
        block_size=block_size,
    )

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(
            2,
            dtype=dtype,
            shape_options=(None, {
                "constexpr": True,
                "upper_bound": 128
            }),
        ),
    )

    return arrangement_, application, tensors