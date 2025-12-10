import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(
    input,
    output,
    offset,
    block_size=None,
):
    if block_size is None:
        block_size = ninetoothed.block_size()

    output_arranged = output.tile((block_size, -1))

    #  (N, ) -> (1, N) -> (N, N)
    input_arranged = input.flatten().unsqueeze(0).expand((output.shape[0], -1))
    input_arranged = input_arranged.tile((block_size, -1))

    return input_arranged, output_arranged, offset


# output.shape = (block_size,)
def application(input, output, offset):
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):

            col = output[i, j].offsets(1)
            row = output[i, j].offsets(0)

            idx = ntl.where(offset >= 0, row, col)

            output[i, j] = ntl.where(
                col - row == offset,
                input[i, idx],
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
        Tensor(  # input 
            ndim,
            dtype=dtype,
        ),
        Tensor(  # output 
            2,
            dtype=dtype,
        ),
        # Tensor(  # output
        #     2,
        #     dtype=dtype,
        #     shape_options=(None, {
        #         "constexpr": True,
        #         "upper_bound": 128
        #     }),
        # ),
        Tensor(0, dtype=ninetoothed.uint64),  # offset 
    )

    return arrangement_, application, tensors