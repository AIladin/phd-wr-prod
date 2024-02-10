import numpy as np
from numba import cuda, uint8
from tqdm.auto import tqdm

from array_based import ArrayWrPermutation


def check_conjugation(a, b):
    inv_a = a.inverse()
    inv_b = b.inverse()

    return (inv_a * b * a * b) == (inv_b * a * b * a)


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
def conjugation_kernell(elements, inverse, tmp_x, tmp_y, tmp_z, idx_b, res):
    idx_a = cuda.grid(1)

    if idx_a == idx_b:
        res[idx_a] = False
        return

    # inv_a * b * a * b

    # inv_a * b -> tmp_x
    mul(inverse[idx_a], elements[idx_b], tmp_x[idx_a])

    # tmp_x * a -> tmp_y
    mul(tmp_x[idx_a], elements[idx_a], tmp_y[idx_a])

    # tmp_y * b -> tmp_x
    mul(tmp_y[idx_a], elements[idx_b], tmp_x[idx_a])

    # inv_b * a * b * a

    # inv_b * a -> tmp_z
    mul(inverse[idx_b], elements[idx_a], tmp_z[idx_a])

    # tmp_z * b -> tmp_y
    mul(tmp_z[idx_a], elements[idx_b], tmp_y[idx_a])

    # tmp_y * a -> tmp_z
    mul(tmp_y[idx_a], elements[idx_a], tmp_z[idx_a])

    # tmp_x == tmp_z
    if eq(tmp_x[idx_a], tmp_z[idx_a]):
        res[idx_a] = True
    else:
        res[idx_a] = False


if __name__ == "__main__":
    group = np.load("z3z3z3.npy")
    order_3_mask = np.load("z3z3z3_order_3.npy")
    order_3_elements = [ArrayWrPermutation(arr) for arr in group[order_3_mask]]
    order_3_inverse = [e.inverse() for e in order_3_elements]

    arr_elements = cuda.to_device(group[order_3_mask])
    arr_inverse = cuda.to_device(np.stack([e.array for e in order_3_inverse]))
    res = np.zeros((len(arr_elements), len(arr_elements)), dtype=bool)

    tmp_x = cuda.device_array_like(arr_elements)
    tmp_y = cuda.device_array_like(arr_elements)
    tmp_z = cuda.device_array_like(arr_elements)

    batch_res = cuda.to_device(
        np.zeros(len(arr_elements), dtype=bool),
        copy=False,
    )

    print("----Prepared arrays----")

    for idx_b in tqdm(range(len(arr_elements))):
        threads_per_block = 256
        blocks_per_grid = (
            len(arr_elements) + (threads_per_block - 1)
        ) // threads_per_block

        conjugation_kernell[
            blocks_per_grid,
            threads_per_block,
        ](
            arr_elements,
            arr_inverse,
            tmp_x,
            tmp_y,
            tmp_z,
            idx_b,
            batch_res,
        )

        res_host = batch_res.copy_to_host()

        if any(res_host):
            for idx_a in res_host:
                a = order_3_elements[idx_a]
                b = order_3_elements[idx_b]
                print(check_conjugation(a, b))
                print(a.to_frozendict())
                print(b.to_frozendict())

        res[:, idx_b] = res_host

    np.save("res.npy", res)
    # x_arr = ArrayWrPermutation(x)
    # y_arr = ArrayWrPermutation(y)

    # test = ArrayWrPermutation(x) * ArrayWrPermutation(y)
