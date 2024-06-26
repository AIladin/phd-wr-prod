import itertools
from collections import deque

from frozendict import frozendict
from tqdm.auto import tqdm

from wrepy.main import (CyclicGroupPermutationFactory, GeneratorSetFactory,
                  Permutation, PermutationGroup)
# from array_based import ArrayWrPermutation


def check_conjugation(a, b):
    inv_a = a.inverse()
    # inv_b = b.inverse()

    inv_a_b_a = inv_a * b * a

    return b * inv_a_b_a == inv_a_b_a * b


class Portrait:
    def __init__(self, alpha: list[Permutation], node_value: int):
        self.alpha = alpha
        self.value = node_value
        self.children: list["Portrait"] = []

    def __repr__(self):
        return f"Portrait({self.value}, children={self.children})"

    def __eq__(self: "Portrait", other):
        if self.alpha != other.alpha:
            raise NotImplementedError
        return self.value == other.value and all(
            [c1 == c2 for c1, c2 in zip(self.children, other.children)]
        )

    def __getitem__(self, v: int) -> "Portrait":
        return self.children[v]

    def __call__(self, value: tuple[int, ...]) -> tuple[int, ...]:
        root = self

        res = []

        for i in value:
            res.append(i ** root.alpha[root.value])
            if root.children:
                root = root[i]
            else:
                break

        return tuple(res)

    @classmethod
    def from_lists(cls, lists: list, alpha: list[Permutation]) -> "Portrait":
        root = Portrait(alpha, lists[0])

        last_flat_level = [root]

        for level in lists[1:]:
            new_flat_level = []

            for parent, children in zip(last_flat_level, level):
                for child_value in children:
                    child = Portrait(alpha, child_value)

                    parent.children.append(child)
                    new_flat_level.append(child)

            last_flat_level = new_flat_level

        return root

    @classmethod
    def from_permutation(cls, alpha: list[Permutation], permutation: Permutation):
        n_children = len(alpha)
        depth = len(permutation.rule)
        raise NotImplementedError

    def depth(self):
        res = 0
        root = self
        while root.children:
            res += 1
            root = root[0]

        return res

    def as_permutation(self) -> frozendict:
        N = len(self.children)
        d = self.depth()

        domain = itertools.product(range(N), repeat=d + 1)

        return frozendict({x: self(x) for x in domain})

    def print_tree(self):
        res = []
        q = deque([self])
        while q:
            row = []
            for _ in range(len(q)):
                node = q.popleft()
                if not node:
                    row.append("#")
                    continue
                row.append(node.value)
                q.extend(node.children)
            res.append(row)
        rows = len(res)
        base = len(self.alpha[0].group.underlying_set) ** (rows)
        for r in range(rows):
            for v in res[r]:
                print(" " * (base), end="")
                print(v, end="")
                print(" " * (base - 1), end="")
            print("|")
            base //= len(self.alpha[0].group.underlying_set)


if __name__ == "__main__":
    N = 3
    z3 = PermutationGroup(set(range(N)), CyclicGroupPermutationFactory)
    z3z3 = z3.wreath_product(z3)
    z3z3z3 = z3z3.wreath_product(z3)

    alpha = list(z3.elements)
    alpha = [alpha[-1], alpha[0], alpha[1]]

    print("Let alpha be defined as the following:")
    for i, e in enumerate(alpha):
        print(f"{i} <-> {e}")

    print("-----------")

    C2d = [
        0,
        [
            [2, 1, 0],
        ],
    ]

    # print("Permutation for portrait C2d")
    # c2d_perm = portrait_to_permutation(C2d, N, alpha)
    # print(c2d_perm)
    # c2d_perm = frozendict(c2d_perm)

    # for element in z3z3.elements:
    #     if c2d_perm == element.rule:
    #         print("Found c2d in z3z3")
    #         break
    # print("--------")

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
    c3d_perm = Permutation(C3d.as_permutation(), z3z3z3)

    D_new = Portrait.from_lists(
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
    d_new_perm = Permutation(D_new.as_permutation(), z3z3z3)
    D_new.print_tree()
    print(d_new_perm.order)

    # p = Portrait.from_lists(
    #     [
    #         0,
    #         [
    #             [1, 2, 1],
    #         ],
    #         [
    #             [1, 1, 1],
    #             [1, 1, 1],
    #             [1, 1, 1],
    #         ],
    #     ],
    #     alpha,
    # )
    # p.print_tree()
    # for k, v in p.as_permutation().items():
    #     print(f"{k}: {v}")

    # p_perm = p.as_permutation()

    # for k,v in e.rule.items():
    #     print(f"{k}:{v}")

    # for i, element in enumerate(tqdm(z3z3z3.elements)):
    #     if d_new_perm == element:
    #         print(f"Found d_new in z3z3z3: {i}")
    #         break
    # for i, element in enumerate(tqdm(z3z3z3.elements)):
    #     if c3d_perm == element.rule:
    #         print("Found d3d in z3z3z3")
    #         break

    # print(element.order)
    # print(i)

    # print(check_conjugation(c3d_perm, d_new_perm))

    test_g = PermutationGroup(
        z3z3z3.underlying_set,
        GeneratorSetFactory,
        generator_set=([c3d_perm.rule, d_new_perm.rule]),
    )
    print(len(test_g.elements))

    # print(ArrayWrPermutation.from_dict_permutation(c3d_perm).n_fixed_points())
    # print(ArrayWrPermutation.from_dict_permutation(d_new_perm).n_fixed_points())

    # print("--------")
