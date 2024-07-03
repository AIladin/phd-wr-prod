import math


def kernel_2d_spec(
    shape: tuple[int, int],
    threadsperblock: tuple[int, int] = (16, 16),
):
    blockspergrid_x = math.ceil(shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    return blockspergrid, threadsperblock
