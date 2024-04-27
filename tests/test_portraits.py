import itertools
import sys

sys.path.append("src")

from frozendict import frozendict

from portraits import CyclicGroupPermutationFactory, PermutationGroup, Portrait

z3z3z3_10_mapping = frozendict(
    {
        # | Manually on call |             Script         |
        (0, 0, 0): (0, 1, 1),  # (0, 0, 0): (0, 2, 1) FIXME
        (0, 0, 1): (0, 1, 2),  # (0, 0, 1): (0, 2, 2) FIXME
        (0, 0, 2): (0, 1, 0),  # (0, 0, 2): (0, 2, 0) FIXME
        (0, 1, 0): (0, 2, 1),  # (0, 1, 0): (0, 0, 1) FIXME
        (0, 1, 1): (0, 2, 2),  # (0, 1, 1): (0, 0, 2) FIXME
        (0, 1, 2): (0, 2, 0),  # (0, 1, 2): (0, 0, 0) FIXME
        (0, 2, 0): (0, 0, 1),  # (0, 2, 0): (0, 1, 1) FIXME
        (0, 2, 1): (0, 0, 2),  # (0, 2, 1): (0, 1, 2) FIXME
        (0, 2, 2): (0, 0, 0),  # (0, 2, 2): (0, 1, 0) FIXME
        (1, 0, 0): (1, 2, 1),  # (1, 0, 0): (1, 2, 1) OK
        (1, 0, 1): (1, 2, 2),  # (1, 0, 1): (1, 2, 2) OK
        (1, 0, 2): (1, 2, 0),  # (1, 0, 2): (1, 2, 0) OK
        (1, 1, 0): (1, 0, 1),  # (1, 1, 0): (1, 0, 1) OK
        (1, 1, 1): (1, 0, 2),  # (1, 1, 1): (1, 0, 2) OK
        (1, 1, 2): (1, 0, 0),  # (1, 1, 2): (1, 0, 0) OK
        (1, 2, 0): (1, 1, 1),  # (1, 2, 0): (1, 1, 1) OK
        (1, 2, 1): (1, 1, 2),  # (1, 2, 1): (1, 1, 2) OK
        (1, 2, 2): (1, 1, 0),  # (1, 2, 2): (1, 1, 0) OK
        (2, 0, 0): (2, 1, 1),  # (2, 0, 0): (2, 1, 1) OK
        (2, 0, 1): (2, 1, 2),  # (2, 0, 1): (2, 1, 2) OK
        (2, 0, 2): (2, 1, 0),  # (2, 0, 2): (2, 1, 0) OK
        (2, 1, 0): (2, 2, 1),  # (2, 1, 0): (2, 2, 1) OK
        (2, 1, 1): (2, 2, 2),  # (2, 1, 1): (2, 2, 2) OK
        (2, 1, 2): (2, 2, 0),  # (2, 1, 2): (2, 2, 0) OK
        (2, 2, 0): (2, 0, 1),  # (2, 2, 0): (2, 0, 1) OK
        (2, 2, 1): (2, 0, 2),  # (2, 2, 1): (2, 0, 2) OK
        (2, 2, 2): (2, 0, 0),  # (2, 2, 2): (2, 0, 0) OK
    }
)

N = 3
z3 = PermutationGroup(set(range(N)), CyclicGroupPermutationFactory)
alpha = list(z3.elements)
alpha = [alpha[-1], alpha[0], alpha[1]]

z3z3z3_10_portrait = Portrait.from_lists(
    [
        0,
        [
            [1, 2, 1],
        ],
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
    ],
    alpha,
)


def test_z3z3z3_manual_portrait_to_permutation():
    assert z3z3z3_10_portrait.as_permutation() == z3z3z3_10_mapping
