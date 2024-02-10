import itertools
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Collection, Iterator, Type
from warnings import warn

from frozendict import frozendict
from rich import print
from tqdm.auto import tqdm

from utils import AllMappings, SizedIterable, safe_unpack


class Permutation:
    def __init__(self, rule: dict, group: "PermutationGroup"):
        self.group = group
        self.rule = frozendict(rule)

    def __rpow__(self, val):
        return self.rule[val]

    def __mul__(self, thr: "Permutation") -> "Permutation":
        if self.group != thr.group:
            raise NotImplementedError
        return Permutation({k: v**thr for k, v in self.rule.items()}, self.group)

    def __repr__(self):
        return repr(self.rule)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Permutation) or (
            self.group.underlying_set != other.group.underlying_set
        ):
            raise NotImplementedError
        return self.rule == other.rule

    def inverse(self) -> "Permutation":
        return Permutation({v: k for k, v in self.rule.items()}, self.group)

    def is_conjugate(self, other: "Permutation") -> bool:
        return self.inverse() * other * self == other

    def __hash__(self):
        return hash(self.rule)

    @cached_property
    def order(self):
        order = 1
        p = self
        while p != self.group.identity_element:
            p *= self
            order += 1
        return order


class PermutationFactory(ABC):
    def __init__(self, group: "PermutationGroup"):
        self.group = group

    @abstractmethod
    def __iter__(self) -> Iterator[Permutation]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class CyclicGroupPermutationFactory(PermutationFactory):
    def __iter__(self) -> Iterator[Permutation]:
        sorted_underlying_set = list(sorted(self.group.underlying_set))
        generator = Permutation(
            {
                x: y
                for x, y in zip(
                    sorted_underlying_set,
                    sorted_underlying_set[1:] + [sorted_underlying_set[0]],
                )
            },
            self.group,
        )

        yield generator

        p = generator * generator

        while p != generator:
            yield p
            p *= generator

    def __len__(self):
        return len(self.group.underlying_set)


class PermutationalWreathProductFactory(PermutationFactory):
    def __init__(
        self,
        group: "PermutationGroup",
        group_g: "PermutationGroup",
        group_h: "PermutationGroup",
    ):
        super().__init__(group)
        self.group_g = group_g
        self.group_h = group_h

        self.all_mappings = AllMappings(
            self.group_g.underlying_set, self.group_h.element_factory
        )

    def __iter__(self) -> Iterator[Permutation]:
        new_underling_set = {
            (x, y)
            for x in self.group_g.underlying_set
            for y in self.group_h.underlying_set
        }

        for h in self.all_mappings:
            for g in self.group_g.element_factory:
                new_permutation = Permutation(
                    {
                        (*safe_unpack(x), y): (*safe_unpack(x**g), y ** h[x])
                        for (x, y) in new_underling_set
                    },
                    self.group,
                )
                yield new_permutation

    def __len__(self) -> int:
        return len(self.all_mappings) * self.group_g.order


class ExponentiationFactory(PermutationFactory):
    def __init__(
        self,
        group: "PermutationGroup",
        group_h: "PermutationGroup",
        group_g: "PermutationGroup",
    ):
        super().__init__(group)
        self.group_g = group_g
        self.group_h = group_h

        self.all_mappings = AllMappings(
            self.group_g.underlying_set, self.group_h.element_factory
        )

    def __iter__(self) -> Iterator[Permutation]:
        new_underling_set = set(
            AllMappings(
                self.group_g.underlying_set,
                self.group_h.underlying_set,
            )
        )

        for h in self.all_mappings:
            for g in self.group_g.element_factory:
                new_permutation = Permutation(
                    {
                        mapping: frozendict(
                            {t: (mapping[t**g]) ** h[t] for t in mapping}
                        )
                        for mapping in new_underling_set
                    },
                    self.group,
                )
                yield new_permutation

    def __len__(self) -> int:
        return len(self.all_mappings) * self.group_g.order


class GeneratorSetFactory(PermutationFactory):
    def __init__(
        self,
        group: "PermutationGroup",
        generator_set: Collection[dict],
    ):
        warn(
            (
                "Current implementation of generator set factory "
                "uses a lot of memory & doen't stop for infinite group."
            )
        )
        super().__init__(group)
        self.generator_set = {Permutation(mapping, group) for mapping in generator_set}

    def __iter__(self) -> Iterator[Permutation]:
        elements = self.generator_set

        while True:
            new_elements = {
                x * y for x, y in itertools.product(elements, repeat=2)
            } | elements
            if len(new_elements) == len(elements):
                break

            elements = new_elements

        yield from new_elements

    def __len__(self):
        raise NotImplementedError


class PermutationGroup:
    def __init__(
        self,
        underlying_set: set,
        element_factory_type: Type[PermutationFactory],
        **element_factory_kwargs,
    ):
        self.underlying_set = underlying_set
        self.element_factory = element_factory_type(
            group=self, **element_factory_kwargs
        )

        self.identity_element = Permutation(
            {x: x for x in self.underlying_set},
            self,
        )

    @property
    def elements(self) -> SizedIterable[Permutation]:
        return self.element_factory

    @property
    def order(self) -> int:
        return len(self.element_factory)

    def wreath_product(
        self: "PermutationGroup", other: "PermutationGroup"
    ) -> "PermutationGroup":
        new_underling_set = {
            (*safe_unpack(x), y)
            for x in self.underlying_set
            for y in other.underlying_set
        }

        return PermutationGroup(
            new_underling_set,
            PermutationalWreathProductFactory,
            group_g=self,
            group_h=other,
        )

    def exponentiation(
        self: "PermutationGroup", other: "PermutationGroup"
    ) -> "PermutationGroup":
        new_underling_set = set(
            AllMappings(
                other.underlying_set,
                self.underlying_set,
            )
        )

        return PermutationGroup(
            new_underling_set,
            ExponentiationFactory,
            group_h=self,
            group_g=other,
        )

    def __repr__(self):
        return f"PermutationGroup(order={self.order})"

    def generating_set(self) -> set[Permutation]:
        elements = list(self.elements)
        source_elements = set(elements)
        min_generating_set = source_elements

        for mask in tqdm(
            itertools.product((0, 1), repeat=len(elements)), total=2 ** len(elements)
        ):
            if not any(mask):
                continue

            if sum(mask) >= len(min_generating_set):
                continue

            generaing_set = {
                p for p, indicator in zip(source_elements, mask) if indicator
            }

            if len(generaing_set) < len(min_generating_set):
                new_group = PermutationGroup(
                    self.underlying_set,
                    GeneratorSetFactory,
                    generator_set={p.rule for p in generaing_set},
                )

                generated_elements = set(new_group.elements)

                if generated_elements == source_elements:
                    min_generating_set = generaing_set
                    print(min_generating_set, len(min_generating_set))

        return min_generating_set


if __name__ == "__main__":
    # print("Z3")
    z3 = PermutationGroup(set(range(3)), CyclicGroupPermutationFactory)
    exponentiation = z3.exponentiation(z3)

    # print(Permutation({0: 1, 1: 2, 2: 0}, z3).order)
    # print(exponentiation.generating_set())

    # order_3_elements: set[Permutation] = set()

    z3z3z3 = z3.wreath_product(z3).wreath_product(z3)
    print(z3z3z3)
    # for e in tqdm(z3z3z3.elements):
    #     if e.order == 3:
    #         order_3_elements.add(e)

    # print(len(order_3_elements))

    print("Z3 wr Z3")
    z3z3 = z3.wreath_product(z3)
    print(z3z3)
    for e in tqdm(z3.elements):
        print(e)
        break

    for e in tqdm(z3z3.elements):
        print(e)
        break

    for e in tqdm(z3z3z3.elements):
        print(e)
        break

    for e in exponentiation.elements:
        print(e)
        break

    # print("Z3 wr Z3 wr Z3")
    # print(z3z3z3)

    # print("Z3 ^ Z3")
    # exponentiation.generating_set()

    # print(exponentiation)
    # for e in tqdm(exponentiation.elements):
    #     pass
