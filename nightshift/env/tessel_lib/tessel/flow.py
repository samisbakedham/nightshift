"""tessel.flow: function pipelines built with the >> operator."""


class Flow:
    __slots__ = ("_steps",)

    def __init__(self, steps=()):
        self._steps = tuple(steps)

    def __rshift__(self, fn):
        """Flow() >> f >> g  builds a pipeline that applies f, then g."""
        return Flow(self._steps + (fn,))

    def guard(self, pred, fallback):
        """Append a step: if pred(x) is falsy, replace x with fallback."""
        return self >> (lambda x: x if pred(x) else fallback)

    def pour(self, x):
        """Run the pipeline on x and return the result."""
        for fn in self._steps:
            x = fn(x)
        return x
