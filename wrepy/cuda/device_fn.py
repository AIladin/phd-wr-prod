import numba
from numba import cuda


@cuda.jit(
    numba.int64(numba.int8[:], numba.int64, numba.int64),
    device=True,
)
def get_child_idx(arities, parent_index, path):
    prev_n_nodes = 0
    n_nodes = 1
    product = 1
    level = 0

    for level in range(len(arities)):
        if parent_index < n_nodes:
            # stop on level of i
            break
        product *= arities[level]
        # nodes on prev level
        prev_n_nodes = n_nodes
        # nodes on current level
        n_nodes += product

    local_level_index = parent_index - prev_n_nodes

    child_idx = n_nodes + local_level_index * arities[level] + path

    return child_idx


@cuda.jit(
    numba.void(
        numba.int8[:],
        numba.int8[:],
        numba.int8[:],
        numba.int8[:],
    ),
    device=True,
)
def action(arities, portrait, x, y):
    """Writes the result of portrait action on x into y"""
    current_node_index = 0
    for i in range(len(x)):
        y[i] = (x[i] + portrait[current_node_index]) % arities[i]
        if i < len(x) - 1:
            current_node_index = get_child_idx(arities, current_node_index, x[i])


@cuda.jit(
    numba.void(
        numba.int8[:],
        numba.int8[:],
        numba.int8[:],
        numba.int8[:],
        numba.int8[:, :],
    ),
    device=True,
)
def portait_mul(
    arities,
    portrait_x,
    portrait_y,
    result_portrait,
    all_points,
):
    """Writes the result of multiplication between x and y

    Nodes x/y/result are equaly shaped 1d arrays.
    """

    for point in all_points:
        node_x = 0
        node_y = 0

        result_node = 0

        for level in range(len(arities)):
            x_result = (point[level] + portrait_x[node_x]) % arities[level]
            y_result = (x_result + portrait_y[node_y]) % arities[level]

            result_portrait[result_node] = (y_result - point[level]) % arities[level]

            node_x = get_child_idx(arities, node_x, point[level])
            node_y = get_child_idx(arities, node_y, x_result)

            result_node = get_child_idx(arities, result_node, point[level])


@cuda.jit(numba.bool_(numba.int8[:]), device=True)
def check_zero(portrait):
    """Check is portrait is all zeros"""
    for p in portrait:
        if p != 0:
            return False
    return True


@cuda.jit(numba.void(numba.int8[:], numba.int8[:]), device=True)
def copy_portrait(source, target):
    for i in range(len(source)):
        target[i] = source[i]


@cuda.jit(
    numba.int64(
        numba.int8[:],
        numba.int8[:],
        numba.int8[:],
        numba.int8[:],
        numba.int8[:, :],
    ),
    device=True,
)
def order(
    arities,
    portrait_x,
    portrait_y,
    portrait_z,
    all_points,
):
    """Returns the order of a portrait x.

    Requires 2 tmp portraits which will be mutated during computation:
    - portrait_y
    - portrait_z
    """

    order = 1

    copy_portrait(portrait_x, portrait_y)

    while not check_zero(portrait_y):
        portait_mul(
            arities,
            portrait_y,
            portrait_x,
            portrait_z,
            all_points,
        )

        copy_portrait(portrait_z, portrait_y)
        order += 1

    return order


@cuda.jit(
    numba.void(
        numba.int8[:],
        numba.int8[:],
        numba.int8[:],
        numba.int8[:, :],
    ),
    device=True,
)
def portait_inverse(
    arities,
    portrait_x,
    result_portrait,
    all_points,
):
    for point in all_points:
        node_x = 0

        result_node = 0

        for level in range(len(arities)):
            x_result = (point[level] + portrait_x[node_x]) % arities[level]

            result_portrait[result_node] = (point[level] - x_result) % arities[level]

            node_x = get_child_idx(arities, node_x, point[level])

            result_node = get_child_idx(arities, result_node, x_result)


@cuda.jit(
    numba.int64(
        numba.int8[:],
        numba.int8[:],
        numba.int8[:, :],
    ),
    device=True,
)
def get_n_fixed(
    arities,
    portrait_x,
    all_points,
):
    n_points = 0

    for point in all_points:
        node_x = 0
        result_node = 0
        valid_point = True

        for level in range(len(arities)):
            x_result = (point[level] + portrait_x[node_x]) % arities[level]

            if x_result != point[level]:
                valid_point = False
                break

            node_x = get_child_idx(arities, node_x, point[level])
            result_node = get_child_idx(arities, result_node, point[level])

        if valid_point:
            n_points += 1

    return n_points
