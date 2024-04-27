import numpy as np
import pandas as pd
from numba import cuda, uint8
from tqdm.auto import tqdm

from array_based import ArrayWrPermutation
from main import CyclicGroupPermutationFactory, Permutation, PermutationGroup
from portraits import Portrait


def as_pd(arr_p: ArrayWrPermutation, name: str) -> pd.DataFrame:
    records = [f"{x}: {y}" for x, y in arr_p.to_frozendict().items()]

    return pd.DataFrame({name: records})


if __name__ == "__main__":
    z3 = PermutationGroup(set(range(3)), CyclicGroupPermutationFactory)
    z3z3z3 = z3.wreath_product(z3).wreath_product(z3)
    alpha = list(z3.elements)
    alpha = [alpha[-1], alpha[0], alpha[1]]

    group = np.load("z3z3z3.npy")
    order_3_mask = np.load("z3z3z3_order_3.npy")
    # order_3_elements = [ArrayWrPermutation(arr) for arr in group[order_3_mask]]
    # order_3_inverse = [e.inverse() for e in order_3_elements]

    C3d = Portrait.from_lists(
        [
            0,
            [
                [2, 1, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [2, 1, 0],
            ],
        ],
        alpha,
    )

    C3d.print_tree()

    D3d = Portrait.from_lists(
        [
            1,
            [
                [0, 0, 0],
            ],
            [
                [1, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        ],
        alpha,
    )
    D3d.print_tree()

    d3d_perm = D3d.as_permutation()
    c3d_perm = C3d.as_permutation()

    d3d_idx = 492072
    c3d_idx = 1473992

    d3d_array = ArrayWrPermutation(group[d3d_idx])
    c3d_array = ArrayWrPermutation(group[c3d_idx])

    assert d3d_array.to_frozendict() == d3d_perm
    assert c3d_array.to_frozendict() == c3d_perm

    print(c3d_idx, d3d_idx)

    c = c3d_array

    d = d3d_array
    # d = d * c * c
    # print(d.order(ArrayWrPermutation.from_dict_permutation(z3z3z3.identity_element)))

    x1 = c.inverse()
    x2 = x1 * d
    x3 = x2 * c
    x4 = d * x3
    x5 = x3 * c

    print(x4 == x5)

    
    # print(pd.concat([as_pd(x4, "x4"), as_pd(x5, "x5"),], axis=1).to_markdown())
