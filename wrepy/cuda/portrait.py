import itertools
from typing import Sequence

import numba
import numpy as np
from numba import cuda

from wrepy.main import Permutation, PermutationGroup

from . import device_fn, kernels
from .utils import kernel_2d_spec


def get_n_nodes(arities: Sequence):
    n_nodes = 1
    product = 1

    for level in range(len(arities) - 1):
        product *= arities[level]
        n_nodes += product

    return n_nodes


def get_n_elements(arities: Sequence):
    n_elements = arities[0]
    nodes_per_level = 1

    for level in range(1, len(arities)):
        nodes_per_level *= arities[level - 1]
        n_elements *= arities[level] ** nodes_per_level

    return n_elements


def get_zn_decart_space(*arities: int) -> np.ndarray:
    return np.fromiter(
        itertools.chain.from_iterable(
            itertools.product(*(range(arity) for arity in arities))
        ),
        dtype=np.int8,
    ).reshape((-1, len(arities)))


def portrait_array_from_arities(*arities: int) -> np.ndarray:
    levels_nodes = [1]

    for level in range(len(arities) - 1):
        levels_nodes.append(levels_nodes[-1] * arities[level])

    decart_dims = []

    for level_nodes, level_arity in zip(levels_nodes, arities):
        for i in range(level_nodes):
            decart_dims.append(range(level_arity))

    portraits = np.fromiter(
        itertools.chain.from_iterable(itertools.product(*decart_dims)),
        dtype=np.int8,
    ).reshape(get_n_elements(arities), get_n_nodes(arities))

    return portraits


def to_dict_permutation(
    arities: np.ndarray,
    portraits: np.ndarray,
    group: PermutationGroup,
    all_points: np.ndarray | None = None,
) -> list[Permutation]:
    if all_points is None:
        all_points = get_zn_decart_space(*arities)

    arities_cuda = cuda.to_device(arities)
    portraits_cuda = cuda.to_device(portraits)

    points_cuda = cuda.to_device(all_points)

    result_cuda = cuda.device_array(
        (portraits.shape[0], *all_points.shape), dtype="i1"
    )

    bpg, tpg = kernel_2d_spec((portraits.shape[0], all_points.shape[0]))
    kernels.action_2d_kernel[bpg, tpg](
        arities_cuda,
        portraits_cuda,
        points_cuda,
        result_cuda,
    )

    result = result_cuda.copy_to_host()

    permutations = [
        Permutation(
            {
                tuple(source_point): tuple(target_point)
                for source_point, target_point in zip(all_points, target_points)
            },
            group,
        )
        for target_points in result
    ]

    return permutations
