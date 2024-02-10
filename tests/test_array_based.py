import itertools
import sys

sys.path.append("src")


from array_based import ArrayWrPermutation
from main import CyclicGroupPermutationFactory, PermutationGroup


def test_arr_conversion_z3():
    z3 = PermutationGroup(set(range(3)), CyclicGroupPermutationFactory)

    for e in z3.elements:
        arr = ArrayWrPermutation.from_dict_permutation(e)
        assert e.rule == arr.to_frozendict()


def test_arr_conversion_z3z3():
    z3 = PermutationGroup(set(range(3)), CyclicGroupPermutationFactory)
    z3z3 = z3.wreath_product(z3)

    for e in z3z3.elements:
        arr = ArrayWrPermutation.from_dict_permutation(e)
        assert e.rule == arr.to_frozendict(), arr


def test_inv_z3():
    z3 = PermutationGroup(set(range(3)), CyclicGroupPermutationFactory)

    for e in z3.elements:
        arr = ArrayWrPermutation.from_dict_permutation(e)
        assert e.inverse().rule == arr.inverse().to_frozendict()


def test_inv_z3z3():
    z3 = PermutationGroup(set(range(3)), CyclicGroupPermutationFactory)
    z3z3 = z3.wreath_product(z3)

    for e in z3z3.elements:
        arr = ArrayWrPermutation.from_dict_permutation(e)
        assert e.inverse().rule == arr.inverse().to_frozendict()


def test_mul_z3():
    z3 = PermutationGroup(set(range(3)), CyclicGroupPermutationFactory)

    for e1, e2 in itertools.product(z3.elements, z3.elements):
        arr1 = ArrayWrPermutation.from_dict_permutation(e1)
        arr2 = ArrayWrPermutation.from_dict_permutation(e2)

        assert (e1 * e2).rule == (arr1 * arr2).to_frozendict()


def test_mul_z3z3():
    z3 = PermutationGroup(set(range(3)), CyclicGroupPermutationFactory)
    z3z3 = z3.wreath_product(z3)

    for e1, e2 in itertools.product(z3z3.elements, z3z3.elements):
        arr1 = ArrayWrPermutation.from_dict_permutation(e1)
        arr2 = ArrayWrPermutation.from_dict_permutation(e2)

        assert (e1 * e2).rule == (arr1 * arr2).to_frozendict()
