import dbm
from concurrent.futures import ProcessPoolExecutor
from random import shuffle

import numpy as np
# import pandas as pd
# from joblib import Parallel, delayed
from numba import cuda, uint8
from scipy.sparse import csc_array, dok_array, save_npz
from tqdm.auto import tqdm

from array_based import ArrayWrPermutation
from main import (CyclicGroupPermutationFactory, GeneratorSetFactory,
                  Permutation, PermutationGroup)
from ops_debbuger import as_pd
from portraits import Portrait


def check_conjugation(a, b):
    inv_a = a.inverse()
    # inv_b = b.inverse()

    inv_a_b_a = inv_a * b * a

    return b * inv_a_b_a == inv_a_b_a * b


@cuda.jit(device=True)
def mul(left, right, rez):
    for x in range(3):
        for y in range(3):
            for z in range(3):
                l_x, l_y, l_z = left[x, y, z]
                for i in range(3):
                    rez[x, y, z, i] = right[l_x, l_y, l_z, i]


@cuda.jit(device=True)
def eq(left, right):
    for x in range(3):
        for y in range(3):
            for z in range(3):
                for i in range(3):
                    if left[x, y, z, i] != right[x, y, z, i]:
                        return False

    return True


@cuda.jit(device=True)
def is_trivial(a):
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if (a[x, y, z, 0] != x) or (a[x, y, z, 1] != y) or (a[x, y, z, 2] != z):
                    return False

    return True


@cuda.jit(device=True)
def fixed_point_different_cycles(c, cd):
    for x in range(3):
        for y in range(3):
            for z in range(3):
                # fixed point check
                if (
                    (c[x, y, z, 0] == x)
                    and (c[x, y, z, 1] == y)
                    and (c[x, y, z, 2] == z)
                ):
                    cd_x, cd_y, cd_z = cd[x, y, z]

                    if (
                        (c[cd_x, cd_y, cd_z, 0] == cd_x)
                        and (c[cd_x, cd_y, cd_z, 1] == cd_y)
                        and (c[cd_x, cd_y, cd_z, 2] == cd_z)
                    ):
                        return False
    return True


@cuda.jit()
def conjugation_kernell(
    elements, inverse, no_fixed_point_mask, fixed_3_points_mask, tmp_x, tmp_y, tmp_z, idx_a, res
):
    idx_b = cuda.grid(1)

    if not fixed_3_points_mask[idx_a]:
        res[idx_b] = False
        return

    if idx_a == idx_b:
        res[idx_b] = False
        return

    if not no_fixed_point_mask[idx_b]:
        res[idx_b] = False
        return

    ## --- conj check
    # (inv_a b a) -> tmp_y

    # inv_a * b -> tmp_x
    mul(inverse[idx_a], elements[idx_b], tmp_x[idx_b])

    # tmp_x * a -> tmp_y
    mul(tmp_x[idx_b], elements[idx_a], tmp_y[idx_b])

    # ---

    # b * tmp_y -> tmp_x

    mul(elements[idx_b], tmp_y[idx_b], tmp_x[idx_b])

    # tmp_y * b -> tmp_z
    mul(tmp_y[idx_b], elements[idx_b], tmp_z[idx_b])

    # ---

    # tmp_x == tmp_z
    if eq(tmp_x[idx_b], tmp_z[idx_b]):
        res[idx_b] = True
    else:
        res[idx_b] = False
        return

    # --- non-trivial check

    # a^2 -> tmp_x
    mul(elements[idx_a], elements[idx_a], tmp_x[idx_b])

    # ba^2 -> tmp_y
    mul(elements[idx_b], tmp_x[idx_b], tmp_y[idx_b])

    # (ba^2)^2 -> tmp_z
    mul(tmp_y[idx_b], tmp_y[idx_b], tmp_z[idx_b])

    # (ba^2)^3 -> tmp_x
    mul(tmp_y[idx_b], tmp_z[idx_b], tmp_x[idx_b])

    # tmp_x is trivial
    if is_trivial(tmp_x[idx_b]):
        res[idx_b] = False
        return
    else:
        res[idx_b] = True

    # --- different cycles check

    # ab -> tmp_x
    mul(elements[idx_a], elements[idx_b], tmp_x[idx_b])
    if fixed_point_different_cycles(elements[idx_a], tmp_x[idx_b]):
        res[idx_b] = True
    else:
        res[idx_b] = False


if __name__ == "__main__":
    z3 = PermutationGroup(set(range(3)), CyclicGroupPermutationFactory)
    z3z3z3 = z3.wreath_product(z3).wreath_product(z3)
    alpha = list(z3.elements)
    alpha = [alpha[-1], alpha[0], alpha[1]]

    group = np.load("z3z3z3.npy")
    inverse_group = np.load("z3z3z3_inverse.npy")
    order_3_mask = np.load("z3z3z3_order_3.npy")
    # inverse_group = np.stack([ArrayWrPermutation(arr).inverse().array for arr in tqdm(group, "inverse prep")])

    # np.save("z3z3z3_inverse.npy", inverse_group)

    order_3_elements = [ArrayWrPermutation(arr) for arr in group[order_3_mask]]
    order_3_inverse = [e.inverse() for e in order_3_elements]
    no_fixed_points_mask = [a.n_fixed_points() == 0 for a in order_3_elements]
    fixed_3_points_mask = [a.n_fixed_points() == 3 for a in order_3_elements]
    # print("fixed_3_point_mask", sum(fixed3_points_mask))

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

    # C3d.print_tree()

    D3d = Portrait.from_lists(
        [
            1,
            [
                [0, 0, 0],
            ],
            [
                [1, 1, 0],
                [0, 0, 0],
                [2, 2, 0],
            ],
        ],
        alpha,
    )
    D3d.print_tree()

    d3d_perm = D3d.as_permutation()

    c3d_perm = C3d.as_permutation()

    for i, obj in enumerate(order_3_elements):
        if obj.to_frozendict() == d3d_perm:
            print("d3d", i)
        if obj.to_frozendict() == c3d_perm:
            print("c3d", i)

    # raise RuntimeError

    # d3d_idx = 492072
    # c3d_idx = 1473992
    # c3d_arr = ArrayWrPermutation(group[c3d_idx])
    # c3d_ord3_idx = order_3_elements.index(c3d_arr)
    # print(c3d_ord3_idx)

    group_cuda = cuda.to_device(group[order_3_mask])
    inverse_group_cuda = cuda.to_device(inverse_group[order_3_mask])
    no_fixed_points_mask = cuda.to_device(no_fixed_points_mask)
    fixed_3_points_mask  = cuda.to_device(fixed_3_points_mask)

    # arr_inverse = cuda.to_device(np.stack([e.array for e in order_3_inverse]))

    tmp_x = cuda.device_array_like(group_cuda)
    tmp_y = cuda.device_array_like(group_cuda)
    tmp_z = cuda.device_array_like(group_cuda)

    batch_res = cuda.to_device(
        np.zeros(len(group_cuda), dtype=bool),
        copy=False,
    )
    print("order_3 elements", len(order_3_elements))
    print("no fixed_points check", sum(no_fixed_points_mask))

    # res_indexes = []

    print("----Prepared arrays----")
    res = 0

    res_matrix = dok_array((len(order_3_elements), len(order_3_elements)), dtype=bool)


    for idx_a in tqdm(range(len(group_cuda))):
        threads_per_block = 256
        blocks_per_grid = (
            len(group_cuda) + (threads_per_block - 1)
        ) // threads_per_block

        conjugation_kernell[
            blocks_per_grid,
            threads_per_block,
        ](
            group_cuda,
            inverse_group_cuda,
            no_fixed_points_mask,
            fixed_3_points_mask,
            tmp_x,
            tmp_y,
            tmp_z,
            idx_a,
            batch_res,
        )

        res_host = batch_res.copy_to_host()

        indices_b = np.nonzero(res_host)[0]
        # res += len(indices_b)

        res_matrix[idx_a, indices_b] = True
    # print(res)
    save_npz("sparse_res.npz", csc_array(res_matrix))
    # print(res_matrix)

    # a_arr = order_3_elements[idx_a]
    # print(len(indices_b))

    # a_fixed_points = set(order_3_elements[idx_a].fixed_points())
    # print(c_fixed_points)

    # new_b_ids = []

    # for idx_b in indices_b:
    #     b_candidate = order_3_elements[idx_b]
    #     valid_b = True
    #     for point in a_fixed_points:
    #         image = tuple(b_candidate.action(point))
    #         if image in a_fixed_points - {point}:
    #             valid_b = False
    #             break
    #     if valid_b:
    #         new_b_ids.append(idx_b)

    # print("Different cycles check", len(new_b_ids))

    # trivial = ArrayWrPermutation.from_dict_permutation(z3z3z3.identity_element)

    # idx_b = new_b_ids
    # new_b_ids = []
    # for idx_b in indices_b:
    #     b_candidate = order_3_elements[idx_b]

    #     x = b_candidate * (a_arr * a_arr)
    #     x = x * x * x

    #     if x != trivial:
    #         new_b_ids.append(idx_b)

    # print(len(new_b_ids))

    # break

    # print("non trivial check", len(new_b_ids))

    # def check_group_order(idx) -> int:
    #     d = order_3_elements[idx]
    #     # assert check_conjugation(c3d_arr, d)
    #     test_g = PermutationGroup(
    #         z3z3z3.underlying_set,
    #         GeneratorSetFactory,
    #         generator_set=([c3d_arr.to_frozendict(), d.to_frozendict()]),
    #     )

    #     i = 0

    #     for _ in test_g.elements:
    #         i += 1

    #     # if i == 81:
    #     #     print(idx)

    #     return idx, i

    # group_order_check = []

    # shuffle(new_b_ids)

    # for idx_b, order in tqdm(
    #     Parallel(n_jobs=-1)([delayed(check_group_order)(idx_b) for idx_b in new_b_ids])
    # ):
    #     if order == 81:
    #         group_order_check.append(idx_b)
    #     # print(order)

    # print("order 81 check", len(group_order_check))

    # if len(indices_b) > 0:
    # print(indices_b)
    # print(len(indices_b))
    # print(len(order_3_elements))

    # c = c3d_arr
    # d = arr_b

    # records = []

    # with ProcessPoolExecutor(max_workers=2) as executor:
    # records = list(
    #     tqdm(map(check_group_order, indices_b), total=len(indices_b)),
    # )

    # for idx in tqdm(indices_b):
    #     records.append({"idx_b": idx, "group_order": i})

    #     if i == 81:
    #         print(idx)

    # pd.DataFrame.from_records(records).to_csv(
    #     "generated_group_orders.csv",
    #     index=None,
    # )

    # cols.append(as_pd(d, "d"))
    # cols.append(as_pd(c.inverse() * d  * c, "c^{-1}dc"))
    # cols.append(as_pd((c.inverse() * c.inverse()) * d  * (c * c), "c^{-2}dc^{2}"))

    # res = pd.concat(cols, axis=1)
    # res.to_csv("b[0]_stuff.csv")
    # print(len(indices_b))

    # np.save("res.npy", np.asarray(res_indexes))
    # x_arr = ArrayWrPermutation(x)
    # y_arr = ArrayWrPermutation(y)

    # test = ArrayWrPermutation(x) * ArrayWrPermutation(y)
