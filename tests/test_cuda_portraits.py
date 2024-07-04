import itertools

import numba
import numpy as np
from numba import cuda
from pytest import fixture

from wrepy.cuda import device_fn, kernels, portrait
from wrepy.cuda.utils import kernel_2d_spec
from wrepy.main import (CyclicGroupPermutationFactory, Permutation,
                        PermutationGroup)


def test_get_n_nodes():
    arities = np.array([3, 2, 3], dtype=np.uint8)
    assert portrait.get_n_nodes(arities) == 10

    arities = np.array([3], dtype=np.uint8)
    assert portrait.get_n_nodes(arities) == 1

    arities = np.array([2, 2], dtype=np.uint8)
    assert portrait.get_n_nodes(arities) == 3

    arities = np.array([3, 3], dtype=np.uint8)
    assert portrait.get_n_nodes(arities) == 4

    arities = np.array([3, 3, 3], dtype=np.uint8)
    assert portrait.get_n_nodes(arities) == 13


def test_n_elements():
    arities = np.array([3, 3, 3], dtype=np.uint8)
    assert portrait.get_n_elements(arities) == 3**13
    arities = np.array([2, 2], dtype=np.uint8)
    assert portrait.get_n_elements(arities) == 2**3
    arities = np.array([2, 2, 2], dtype=np.uint8)
    assert portrait.get_n_elements(arities) == 2**7
    arities = np.array([2, 3, 2], dtype=np.uint8)
    assert portrait.get_n_elements(arities) == 2 * 3**2 * 2**6
    arities = np.array([3, 2, 3], dtype=np.uint8)
    assert portrait.get_n_elements(arities) == 3 * 2**3 * 3**6


def test_child_idx():
    @cuda.jit()
    def kernel(arities, parent_node_idx, child_node_idx, result):
        thread_idx = cuda.grid(1)

        if thread_idx >= len(parent_node_idx):
            return

        result[thread_idx] = device_fn.get_child_idx(
            arities, parent_node_idx[thread_idx], child_node_idx[thread_idx]
        )

    arities = np.array([3, 2, 3], dtype=np.int8)

    parent_node_idx = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3])
    path_idx = np.array([0, 1, 2, 0, 1, 0, 1, 0, 1])

    arities_cuda = cuda.to_device(arities)
    parent_cuda = cuda.to_device(parent_node_idx)
    path_cuda = cuda.to_device(path_idx)
    result = cuda.device_array_like(parent_cuda)

    kernel.forall(len(parent_cuda))(
        arities_cuda,
        parent_cuda,
        path_cuda,
        result,
    )

    child_idx = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    assert np.all(result.copy_to_host() == child_idx)


def helper_np_find(source: np.ndarray, x: np.ndarray) -> int | None:
    for i, row in enumerate(source):
        if np.all(row == x):
            return i

    return None


@fixture
def z3z3():
    z3 = PermutationGroup(set(range(3)), CyclicGroupPermutationFactory)
    z3z3 = z3.wreath_product(z3)

    arities = np.array([3, 3], dtype=np.int8)

    domain = portrait.get_zn_decart_space(3, 3)
    portraits = portrait.portrait_array_from_arities(*arities)

    permutations = portrait.to_dict_permutation(arities, portraits, z3z3, domain)

    return z3z3, arities, domain, portraits, permutations


@fixture
def z2z2z2():
    z2 = PermutationGroup(set(range(2)), CyclicGroupPermutationFactory)
    z2z2z2 = (z2.wreath_product(z2)).wreath_product(z2)

    arities = np.array([2, 2, 2], dtype=np.int8)

    domain = portrait.get_zn_decart_space(2, 2, 2)
    portraits = portrait.portrait_array_from_arities(*arities)

    permutations = portrait.to_dict_permutation(arities, portraits, z2z2z2, domain)

    return z2z2z2, arities, domain, portraits, permutations


def multiplication(group, arities, domain, portraits, permutations):
    arities_cuda = cuda.to_device(arities)
    domain_cuda = cuda.to_device(domain)
    portraits_cuda = cuda.to_device(portraits)
    result_cuda = cuda.device_array((portraits.shape[0], *portraits.shape), dtype="i1")

    bpg, tpg = kernel_2d_spec((portraits.shape[0], portraits.shape[0]))

    kernels.multiplication_2d_kernel[bpg, tpg](
        arities_cuda,
        portraits_cuda,
        portraits_cuda,
        result_cuda,
        domain_cuda,
    )

    result = result_cuda.copy_to_host()

    result_indexes = np.asarray(
        [[helper_np_find(portraits, element) for element in row] for row in result]
    )

    assert np.all(result_indexes != None)

    for i, j in itertools.product(
        range(result.shape[0]),
        range(result.shape[1]),
    ):
        p1 = permutations[i]
        p2 = permutations[j]

        dict_res = p1 * p2

        cuda_res = permutations[result_indexes[i][j]]

        assert dict_res == cuda_res, (
            f"{p1}\n"
            f"{p2},\n"
            f"{portraits[i]}\n"
            f"{portraits[j]}\n"
            f"{portraits[result_indexes[i][j]]}"
        )


def test_multiplication_z3z3(z3z3):
    multiplication(*z3z3)


def test_multiplication_z2z2z2(z2z2z2):
    multiplication(*z2z2z2)


def order(group, arities, domain, portraits, permutations):
    arities_cuda = cuda.to_device(arities)
    domain_cuda = cuda.to_device(domain)
    portraits_cuda = cuda.to_device(portraits)

    portraits_y = cuda.device_array_like(portraits_cuda)
    portraits_z = cuda.device_array_like(portraits_cuda)

    orders_cuda = cuda.device_array(len(portraits), dtype="i8")

    kernels.order_kernel.forall(len(portraits))(
        arities_cuda,
        portraits_cuda,
        portraits_y,
        portraits_z,
        domain_cuda,
        orders_cuda,
    )

    orders = orders_cuda.copy_to_host()

    for order, dict_element in zip(orders, permutations):
        assert order == dict_element.order


def test_order_z3z3(z3z3):
    order(*z3z3)


def test_order_z2z2z2(z2z2z2):
    order(*z2z2z2)


def inverse(group, arities, domain, portraits, permutations):
    arities_cuda = cuda.to_device(arities)
    domain_cuda = cuda.to_device(domain)
    portraits_cuda = cuda.to_device(portraits)

    result_cuda = cuda.device_array_like(portraits_cuda)

    kernels.inverse_kernel.forall(len(portraits))(
        arities_cuda,
        portraits_cuda,
        result_cuda,
        domain_cuda,
    )

    result = result_cuda.copy_to_host()
    result_indexes = np.asarray(
        [helper_np_find(portraits, element) for element in result]
    )

    for i, permutation in enumerate(permutations):
        inverse_p = permutations[result_indexes[i]]

        assert permutation.inverse() == inverse_p


def test_inverse_z3z3(z3z3):
    inverse(*z3z3)


def test_inverse_z2z2z2(z2z2z2):
    inverse(*z2z2z2)
