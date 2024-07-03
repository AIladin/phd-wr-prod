import numba
from numba import cuda

from . import device_fn


@cuda.jit(
    numba.void(
        numba.int8[:],
        numba.int8[:, :],
        numba.int8[:, :],
        numba.int8[:, :, :],
    )
)
def action_2d_kernel(arities, portraits, all_points, result):
    portrait_idx, point_idx = cuda.grid(2)

    if portrait_idx >= len(portraits) or point_idx >= len(all_points):
        return

    device_fn.action(
        arities,
        portraits[portrait_idx],
        all_points[point_idx],
        result[portrait_idx, point_idx],
    )


@cuda.jit(
    numba.void(
        numba.int8[:],
        numba.int8[:, :],
        numba.int8[:, :],
        numba.int8[:, :, :],
        numba.int8[:, :],
    )
)
def multiplication_2d_kernel(
    arities,
    portraits_x,
    portraits_y,
    result,
    all_points,
):
    x_idx, y_idx = cuda.grid(2)

    if x_idx >= len(portraits_x) or y_idx >= len(portraits_y):
        return

    device_fn.portait_mul(
        arities,
        portraits_x[x_idx],
        portraits_y[y_idx],
        result[x_idx, y_idx],
        all_points,
    )


@cuda.jit(
    numba.void(
        numba.int8[:],
        numba.int8[:, :],
        numba.int8[:, :],
        numba.int8[:, :],
        numba.int8[:, :],
        numba.int64[:],
    )
)
def order_kernel(
    arities,
    portraits_x,
    portraits_y,
    portraits_z,
    all_points,
    orders,
):
    portrait_idx = cuda.grid(1)

    if portrait_idx >= len(portraits_x):
        return

    orders[portrait_idx] = device_fn.order(
        arities,
        portraits_x[portrait_idx],
        portraits_y[portrait_idx],
        portraits_z[portrait_idx],
        all_points,
    )


@cuda.jit(
    numba.void(
        numba.int8[:],
        numba.int8[:, :],
        numba.int8[:, :],
        numba.int8[:, :],
    )
)
def inverse_kernel(
    arities,
    portraits,
    result,
    all_points,
):
    portrait_idx = cuda.grid(1)

    if portrait_idx >= len(portraits):
        return

    device_fn.portait_inverse(
        arities,
        portraits[portrait_idx],
        result[portrait_idx],
        all_points,
    )
