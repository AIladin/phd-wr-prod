import numpy as np
from numba import cuda, uint8
from tqdm.auto import tqdm

from array_based import ArrayWrPermutation
from main import CyclicGroupPermutationFactory, Permutation, PermutationGroup
from portraits import Portrait


def check_conjugation(a, b):
    inv_a = a.inverse()
    # inv_b = b.inverse()

    inv_a_b_a = inv_a * b * a

    return b * inv_a_b_a == inv_a_b_a * a


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


@cuda.jit()
def conjugation_kernell(elements, inverse, tmp_x, tmp_y, tmp_z, idx_a, res):
    idx_b = cuda.grid(1)

    if idx_a == idx_b:
        res[idx_b] = False
        return

    # (inv_a b a) -> tmp_y

    # inv_a * b -> tmp_x
    mul(inverse[idx_a], elements[idx_b], tmp_x[idx_b])

    # tmp_x * a -> tmp_y
    mul(tmp_x[idx_b], elements[idx_a], tmp_y[idx_b])

    # ---

    # b * tmp_y -> tmp_x

    mul(elements[idx_b], tmp_y[idx_b], tmp_x[idx_b])

    # tmp_y * a -> tmp_z
    mul(tmp_y[idx_b], elements[idx_a], tmp_z[idx_b])

    # ---

    # tmp_x == tmp_z
    if eq(tmp_x[idx_b], tmp_z[idx_b]):
        res[idx_b] = True
    else:
        res[idx_b] = False


if __name__ == "__main__":
    z3 = PermutationGroup(set(range(3)), CyclicGroupPermutationFactory)
    alpha = list(z3.elements)
    alpha = [alpha[-1], alpha[0], alpha[1]]

    group = np.load("z3z3z3.npy")
    order_3_mask = np.load("z3z3z3_order_3.npy")
    order_3_elements = [ArrayWrPermutation(arr) for arr in group[order_3_mask]]
    order_3_inverse = [e.inverse() for e in order_3_elements]

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

    # D3d = Portrait.from_lists(
    #     [
    #         1,
    #         [
    #             [0, 0, 0],
    #         ],
    #         [
    #             [1, 1, 0],
    #             [0, 0, 0],
    #             [0, 0, 0],
    #         ],
    #     ],
    #     alpha,
    # )
    # D3d.print_tree()

    # d3d_perm = D3d.as_permutation()
    # c3d_perm = C3d.as_permutation()




    # d3d_idx = 492072
    c3d_idx = 1473992

    arr_elements = cuda.to_device(order_3_mask)
    # arr_inverse = cuda.to_device(np.stack([e.array for e in order_3_inverse]))

    # tmp_x = cuda.device_array_like(arr_elements)
    # tmp_y = cuda.device_array_like(arr_elements)
    # tmp_z = cuda.device_array_like(arr_elements)

    # batch_res = cuda.to_device(
    #     np.zeros(len(arr_elements), dtype=bool),
    #     copy=False,
    # )

    # # res_indexes = []

    # print("----Prepared arrays----")

    # # for idx_b in tqdm(range(len(arr_elements))):
    # threads_per_block = 256
    # blocks_per_grid = (len(arr_elements) + (threads_per_block - 1)) // threads_per_block

    # conjugation_kernell[
    #     blocks_per_grid,
    #     threads_per_block,
    # ](
    #     arr_elements,
    #     arr_inverse,
    #     tmp_x,
    #     tmp_y,
    #     tmp_z,
    #     2,
    #     batch_res,
    # )

    # res_host = batch_res.copy_to_host()

    # indices_b = np.nonzero(res_host)[0]

    # print(len(indices_b))

    # np.save("res.npy", np.asarray(res_indexes))
    # x_arr = ArrayWrPermutation(x)
    # y_arr = ArrayWrPermutation(y)

    # test = ArrayWrPermutation(x) * ArrayWrPermutation(y)
