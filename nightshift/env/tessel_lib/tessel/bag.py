"""tessel.bag: a multiset."""


class Bag:
    __slots__ = ("_c",)

    def __init__(self):
        self._c = {}

    def drop(self, item, n=1):
        """Add n copies of item. Returns self (chainable)."""
        self._c[item] = self._c.get(item, 0) + n
        return self

    def lift(self, item):
        """Remove ONE copy of item. Raises KeyError if absent. Returns self."""
        if self._c.get(item, 0) == 0:
            raise KeyError(item)
        self._c[item] -= 1
        if self._c[item] == 0:
            del self._c[item]
        return self

    def weight(self, item):
        """How many copies of item (0 if absent)."""
        return self._c.get(item, 0)

    def heaviest(self, k):
        """Top-k (item, count) pairs: count descending, ties by item ascending."""
        return sorted(self._c.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    def __len__(self):
        return sum(self._c.values())
