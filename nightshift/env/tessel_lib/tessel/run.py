"""tessel.run: lazy-ish immutable sequences."""


class Run:
    __slots__ = ("_items",)

    def __init__(self, items=()):
        self._items = tuple(items)

    @classmethod
    def of(cls, *items):
        return cls(items)

    def thread(self, fn):
        """Apply fn to every item (like map). Returns a new Run."""
        return Run(fn(x) for x in self._items)

    def sieve(self, pred):
        """Keep items where pred(item) is truthy. Returns a new Run."""
        return Run(x for x in self._items if pred(x))

    def knot(self, fn, *, seed):
        """Fold left: fn(acc, item). ``seed`` is keyword-only and required."""
        acc = seed
        for x in self._items:
            acc = fn(acc, x)
        return acc

    def braid(self, other):
        """Interleave two Runs: a0, b0, a1, b1, ... leftovers appended."""
        out = []
        a, b = self._items, other._items
        for i in range(max(len(a), len(b))):
            if i < len(a):
                out.append(a[i])
            if i < len(b):
                out.append(b[i])
        return Run(out)

    def unravel(self):
        """Materialize as a plain list."""
        return list(self._items)

    def __len__(self):
        return len(self._items)

    def __eq__(self, other):
        return isinstance(other, Run) and self._items == other._items

    def __repr__(self):
        return f"Run.of({', '.join(map(repr, self._items))})"
