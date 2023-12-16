import itertools
from typing import Iterable

from frozendict import frozendict
from rich import print
from rich.table import Table


def get_all_mappings(this: set, other: set) -> set[frozendict]:
    return {
        frozendict({x: y for x, y in zip(this, image)})
        for image in itertools.product(other, repeat=len(this))
    }


def safe_unpack(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


class Permutation:
    def __init__(self, rule: dict):
        self.rule = frozendict(rule)

    def __call__(self, val):
        return self.rule[val]

    def __rpow__(self, val):
        return self(val)

    def __mul__(self, thr: "Permutation") -> "Permutation":
        return Permutation({k: thr(v) for k, v in self.rule.items()})

    def __repr__(self):
        return repr(self.rule)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Permutation):
            raise NotImplementedError
        return self.rule == other.rule

    def __hash__(self):
        return hash(self.rule)


class PermutationGroup:
    def __init__(self, underlying_set: set, permutations: set[Permutation]):
        self.underlying_set = underlying_set
        self.permutations = permutations

    @classmethod
    def symmetric_group(cls, underlying_set: set):
        ordered_repr = list(underlying_set)

        permutations = {
            Permutation({k: v for k, v in zip(ordered_repr, permutation)})
            for permutation in itertools.permutations(underlying_set)
        }

        return cls(underlying_set, permutations)

    @classmethod
    def from_generator(cls, generator: Permutation) -> "PermutationGroup":
        underlying_set = set(generator.rule.keys())

        permutations = {generator}
        next_element = generator * generator

        while next_element != generator:
            permutations.add(next_element)
            next_element *= generator

        return cls(underlying_set, permutations)

    def multiplication_table(self, use_rich=False):
        table = [
            [p1] + [p1 * p2 for p2 in self.permutations] for p1 in self.permutations
        ]
        if not use_rich:
            return table

        rich_table = Table()
        rich_table.add_column("")

        for p in self.permutations:
            rich_table.add_column(str(p))

        for row in table:
            rich_table.add_row(*map(str, row))

        return rich_table

    def wreath_product(self, other: "PermutationGroup") -> "PermutationGroup":
        new_underling_set = {
            (x, y) for x in self.underlying_set for y in other.underlying_set
        }

        new_permutations: set[Permutation] = set()

        for h in get_all_mappings(self.underlying_set, other.permutations):
            print(h)
            raise RuntimeError
            for g in self.permutations:
                new_permutation = Permutation(
                    {
                        (*safe_unpack(x), y): (*safe_unpack(x ** g), y ** h[x])
                        for (x, y) in new_underling_set
                    }
                )
                new_permutations.add(new_permutation)

        new_underling_set = {(*safe_unpack(x), y) for (x, y) in new_underling_set}

        return PermutationGroup(new_underling_set, new_permutations)

    def exponentiation(self, other: "PermutationGroup") -> "PermutationGroup":
        new_underling_set = get_all_mappings(self.underlying_set, other.underlying_set)
        new_permutations: set[Permutation] = set()

        for h in get_all_mappings(self.underlying_set, other.permutations):
            for g in other.permutations:
                new_permutation = Permutation(
                    {
                        mapping: frozendict(
                            {t: (mapping[t**g]) ** h[t] for t in mapping}
                        )
                        for mapping in new_underling_set
                    }
                )
                new_permutations.add(new_permutation)

        return PermutationGroup(new_underling_set, new_permutations)


if __name__ == "__main__":
    z3 = PermutationGroup.from_generator(Permutation({0: 1, 1: 2, 2: 0}))

    print("Z3")
    print(z3.underlying_set, z3.permutations)

    # print("Z3 wr Z3")
    # z3z3 = z3.wreath_product(z3)
    # print(z3z3.underlying_set, len(z3z3.permutations))

    # print("Z3 wr Z3 wr Z3")
    # z3z3z3 = z3z3.wreath_product(z3)
    # print(z3z3z3.underlying_set, len(z3z3z3.permutations))
    # print("Z3 ^ Z3")
    exponentiation = z3.exponentiation(z3)
    print(exponentiation.underlying_set, len(exponentiation.permutations))

    # wr = s3.wreath_product(s3)
    # print(wr.underlying_set, len(wr.Permutation
