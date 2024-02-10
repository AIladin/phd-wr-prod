import itertools
import math

import numpy as np
from frozendict import frozendict
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from main import CyclicGroupPermutationFactory, Permutation, PermutationGroup


class ArrayWrPermutation:
    def __init__(self, array: np.ndarray):
        assert array.shape[-1] == array.ndim - 1, array.shape
        assert (np.array(array.shape[:-1]) == 3).all(), array.shape
        self.array = array

    @classmethod
    def from_dict_permutation(
        cls, dict_permutation: Permutation
    ) -> "ArrayWrPermutation":
        key = dict_permutation.rule.key(0)
        if isinstance(key, tuple):
            key_dim = len(key)
        else:
            key_dim = 1

        domain = int(len(dict_permutation.group.underlying_set) ** (1 / key_dim))

        array = np.empty(
            (
                *((domain,) * key_dim),
                key_dim,
            ),
            dtype=np.uint8,
        )

        for k, v in dict_permutation.rule.items():
            array[k] = v

        return cls(array)

    def to_frozendict(self) -> frozendict:
        mapping = {}

        for index in itertools.product(
            range(self.domain),
            repeat=self.wr_order,
        ):
            value = tuple(self.array[index])

            if len(index) == 1:
                index = index[0]

            if len(value) == 1:
                value = value[0]

            mapping[index] = value

        return frozendict(mapping)

    @property
    def domain(self):
        return self.array.shape[0]

    @property
    def wr_order(self):
        return self.array.shape[-1]

    def __repr__(self):
        return repr(self.array)

    def __mul__(self, thr: "ArrayWrPermutation") -> "ArrayWrPermutation":
        if self.array.shape != thr.array.shape:
            raise NotImplementedError

        idx = self.array.reshape((-1, self.wr_order))

        new_arr = thr.array[(*tuple(idx.T), slice(None))]
        new_arr = new_arr.reshape((*((self.domain,) * self.wr_order), self.wr_order))

        return ArrayWrPermutation(new_arr)

    def inverse(self) -> "ArrayWrPermutation":
        inverse_arr = np.empty_like(self.array)

        indices = np.indices(self.array.shape[:-1])
        indices = np.moveaxis(indices, 0, -1).reshape(-1, self.wr_order)

        inverse_arr[tuple(self.array.reshape(-1, self.wr_order).T)] = indices

        return ArrayWrPermutation(inverse_arr)

    def action(self, val: np.ndarray):
        assert len(val) == self.array.ndim - 1
        return self.array[val]

    def __eq__(self, other) -> bool:
        if not isinstance(other, ArrayWrPermutation):
            raise NotImplementedError

        return np.array_equal(self.array, other.array)

    def order(self, id_element: "ArrayWrPermutation") -> int:
        order = 1
        p = ArrayWrPermutation(self.array.copy())
        while p != id_element:
            p *= self
            order += 1
        return order


class ArrPermutationGroup:
    def __init__(self, arr: np.ndarray):
        self.arr = arr

    @property
    def domain(self) -> int:
        return self.arr.shape[1]

    @classmethod
    def from_permutation_group(cls, group: PermutationGroup):
        elements = [
            ArrayWrPermutation.from_dict_permutation(element).array
            for element in tqdm(group.elements)
        ]

        return cls(np.stack(elements))

    @property
    def elements(self):
        for arr in self.arr:
            yield ArrayWrPermutation(arr)


def check_order_3(element: ArrayWrPermutation, id_element: ArrayWrPermutation):
    return element.order(id_element) == 3


if __name__ == "__main__":
    z3 = PermutationGroup(set(range(3)), CyclicGroupPermutationFactory)
    z3z3z3 = z3.wreath_product(z3).wreath_product(z3)

    # arr_z3 = ArrPermutationGroup.from_permutation_group(z3z3z3)
    # np.save("z3z3z3.npy", arr_z3.arr)

    group = ArrPermutationGroup(np.load("z3z3z3.npy"))
    identity = ArrayWrPermutation.from_dict_permutation(z3z3z3.identity_element)

    # order_3_idx = np.asarray(Parallel(n_jobs=10)(
    #     delayed(check_order_3)(e, identity)
    #     for e in tqdm(group.elements, total=len(group.arr))
    # ))
    # np.save("z3z3z3_order_3.npy", order_3_idx)

    order_3_mask = np.load("z3z3z3_order_3.npy")

    order_3_elements = [ArrayWrPermutation(arr) for arr in group.arr[order_3_mask]]

    def check_conjugation(a, b):
        inv_a = a.inverse()
        inv_b = b.inverse()

        # a_b = a * b

        return (inv_a * b * a * b) == (inv_b * a * b * a)

    for i, j, res in Parallel(11, return_as="generator")(
        delayed(check_conjugation)(i, a, j, b)
        for (i, a), (j, b) in tqdm(
            itertools.combinations(enumerate(order_3_elements), r=2),
            total=math.comb(len(order_3_elements), 2),
        )
    ):
        if res:
            print(i, j)
