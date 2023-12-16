import itertools
from typing import Iterator, Protocol, TypeVar

from frozendict import frozendict

T = TypeVar("T", covariant=True)


class SizedIterable(Protocol[T]):
    def __iter__(self) -> Iterator[T]:
        pass

    def __len__(self) -> int:
        pass


def safe_unpack(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


class AllMappings:
    def __init__(self, domain: SizedIterable, image: SizedIterable):
        self.domain = domain
        self.image = image

    def __iter__(self) -> Iterator[frozendict]:
        for image in itertools.product(self.image, repeat=len(self.domain)):
            yield frozendict({x: y for x, y in zip(self.domain, image)})

    def __len__(self):
        return len(self.image) ** len(self.domain)
