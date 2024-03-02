import itertools

from frozendict import frozendict
from tqdm.auto import tqdm

from main import CyclicGroupPermutationFactory, Permutation, PermutationGroup


def portrait_to_permutation(
    portrait: list,
    N: int,
    alpha: list[Permutation],
):
    mapping = {}

    for element in itertools.product(range(N), repeat=len(portrait)):
        new_element = []

        # print(element)
        for depth, portrait_level in enumerate(portrait):
            # print(portrait_level)
            if depth == 0:
                permutation = alpha[portrait_level[depth]]
            else:
                permutation_index = portrait_level
                for e in new_element:
                    permutation_index = permutation_index[e]
                permutation = alpha[permutation_index]

            # print(permutation)

            new_element.append(element[depth] ** permutation)
            # print(new_element)
            # print("---")

        mapping[element] = tuple(new_element)
        # print("\n")

    return mapping


if __name__ == "__main__":
    N = 3
    z3 = PermutationGroup(set(range(N)), CyclicGroupPermutationFactory)
    z3z3 = z3.wreath_product(z3)
    z3z3z3 = z3.wreath_product(z3z3)

    alpha = list(z3.elements)
    alpha = [alpha[-1], alpha[0], alpha[1]]

    print("Let alpha be defined as the following:")
    for i, e in enumerate(alpha):
        print(f"{i} <-> {e}")

    print("-----------")

    C2d = [
        [0],
        [2, 1, 0],
    ]

    print("Permutation for portrait C2d")
    c2d_perm = portrait_to_permutation(C2d, N, alpha)
    print(c2d_perm)
    c2d_perm = frozendict(c2d_perm)

    for element in z3z3.elements:
        if c2d_perm == element.rule:
            print("Found c2d in z3z3")
            break

    print("--------")

    C3d = [
        [0],
        [2, 1, 0],
        [[0, 0, 0], [0, 0, 0], [2, 1, 0]],
    ]

    print("Permutation for portrait C3d")
    c3d_perm = portrait_to_permutation(C3d, N, alpha)
    print(c3d_perm)
    # c3d_perm = frozendict(c3d_perm)

    # for element in tqdm(z3z3z3.elements):
    #     if c3d_perm == element.rule:
    #         print("Found c3d in z3z3z3")
    #         break

    # print("--------")
