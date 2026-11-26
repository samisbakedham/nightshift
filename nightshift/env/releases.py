"""The release schedule for tessel: one module per simulated day.

Each release has
  * ``docs``      - the reference docs the model can read (RAG arm / teacher only)
  * ``templates`` - parameterized task generators with a reference solution

Every template's ``gen(rng)`` returns:
  prompt  : what the function must do (no API hints beyond the module name)
  inputs  : list of Python expressions, each a tuple of args for ``solve``
  ref     : reference solution source (defines ``solve``), used ONLY to compute
            expected outputs and to validate the template. Never shown to models.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Template:
    id: str
    gen: Callable[[random.Random], dict]
    heldout: bool = False   # never practiced, never gated, eval only ("unseen family")


@dataclass
class Release:
    day: int
    module: str          # e.g. "tessel.run"
    preamble: str        # imports the TEST harness needs to build inputs/expected
    docs: str
    templates: list[Template] = field(default_factory=list)


def _ints(rng, n, lo=-9, hi=20):
    return [rng.randint(lo, hi) for _ in range(n)]


# --------------------------------------------------------------------------- day 1
RUN_DOCS = """\
## tessel.run  (released day 1)
`from tessel.run import Run`
- `Run.of(*items) -> Run` / `Run(iterable) -> Run`: build an immutable sequence.
- `r.thread(fn) -> Run`: apply fn to every item (tessel's map).
- `r.sieve(pred) -> Run`: keep items where pred is truthy (tessel's filter).
- `r.knot(fn, *, seed) -> value`: left fold, fn(acc, item); returns the final accumulator itself (not a Run). `seed` is KEYWORD-ONLY and required.
- `r.braid(other) -> Run`: interleave a0,b0,a1,b1,... leftovers appended.
- `r.unravel() -> list`: materialize as a list.
- Runs have no `.map`, `.filter`, `.reduce`, indexing or iteration helpers."""


def _run_sum_sq_even(rng):
    k = rng.randint(2, 5)
    return dict(
        prompt=f"Write `solve(r)` that takes a tessel `Run` of ints and returns the sum of "
               f"(x * {k}) for the even items only, as an int.",
        inputs=[f"(Run.of({', '.join(map(str, _ints(rng, rng.randint(3, 7))))}),)" for _ in range(3)],
        ref=f"from tessel.run import Run\n"
            f"def solve(r):\n    return r.sieve(lambda x: x % 2 == 0).thread(lambda x: x * {k}).knot(lambda a, b: a + b, seed=0)\n",
    )


def _run_braid_shift(rng):
    d = rng.randint(1, 9)
    return dict(
        prompt=f"Write `solve(a, b)` that takes two tessel `Run`s of ints, adds {d} to every item of b, "
               f"interleaves a with the shifted b (a first), and returns the result as a `Run`.",
        inputs=[f"(Run.of({', '.join(map(str, _ints(rng, rng.randint(1, 4))))}), "
                f"Run.of({', '.join(map(str, _ints(rng, rng.randint(1, 4))))}))" for _ in range(3)],
        ref=f"from tessel.run import Run\n"
            f"def solve(a, b):\n    return a.braid(b.thread(lambda x: x + {d}))\n",
    )


def _run_max_fold(rng):
    t = rng.randint(0, 6)
    return dict(
        prompt=f"Write `solve(r)` that takes a tessel `Run` of ints and returns a plain list of the "
               f"items strictly greater than {t}, each squared.",
        inputs=[f"(Run.of({', '.join(map(str, _ints(rng, rng.randint(2, 6))))}),)" for _ in range(3)],
        ref=f"from tessel.run import Run\n"
            f"def solve(r):\n    return r.sieve(lambda x: x > {t}).thread(lambda x: x * x).unravel()\n",
    )


# --------------------------------------------------------------------------- day 2
GRID_DOCS = """\
## tessel.grid  (released day 2)
`from tessel.grid import Grid`
- `Grid.weave(rows, cols, fill=0) -> Grid`: new grid.
- `g.span() -> (rows, cols)`.
- `g.at(r, c) -> value`: the cell's value. Negative indices raise IndexError.
- `g.stamp(r, c, value) -> Grid`: returns a NEW grid; grids are immutable (no item assignment).
- `g.rotate() -> Grid`: 90 degrees clockwise.
- `g.rows() -> list[tuple]`."""


def _grid_diag(rng):
    n = rng.randint(2, 5)
    v = rng.randint(1, 9)
    return dict(
        prompt=f"Write `solve(n)` that returns a tessel `Grid` of size n x n filled with 0, "
               f"with {v} on the main diagonal.",
        inputs=[f"({rng.randint(1, 5)},)" for _ in range(2)] + [f"({n},)"],
        ref=f"from tessel.grid import Grid\n"
            f"def solve(n):\n    g = Grid.weave(n, n, fill=0)\n    for i in range(n):\n        g = g.stamp(i, i, {v})\n    return g\n",
    )


def _grid_corner_after_rotate(rng):
    return dict(
        prompt="Write `solve(g)` that takes a tessel `Grid`, rotates it clockwise once, and returns "
               "the value in the top-left cell of the rotated grid.",
        inputs=[f"(Grid({[[rng.randint(0, 9) for _ in range(c)] for _ in range(r)]!r}),)"
                for r, c in [(rng.randint(1, 4), rng.randint(1, 4)) for _ in range(3)]],
        ref="from tessel.grid import Grid\n"
            "def solve(g):\n    return g.rotate().at(0, 0)\n",
    )


def _grid_row_sums(rng):
    m = rng.randint(2, 4)
    return dict(
        prompt=f"Write `solve(g)` that takes a tessel `Grid` of ints and returns a list with each row's "
               f"sum multiplied by {m}.",
        inputs=[f"(Grid({[[rng.randint(-5, 9) for _ in range(c)] for _ in range(r)]!r}),)"
                for r, c in [(rng.randint(1, 4), rng.randint(1, 4)) for _ in range(3)]],
        ref=f"from tessel.grid import Grid\n"
            f"def solve(g):\n    return [sum(row) * {m} for row in g.rows()]\n",
    )


# --------------------------------------------------------------------------- day 3
SCROLL_DOCS = """\
## tessel.scroll  (released day 3)
`from tessel.scroll import Scroll`
- `Scroll(text) -> Scroll`; the raw string is `s.text` (a str).
- `s.verses() -> list[str]`: non-empty lines, stripped.
- `s.tally() -> dict[str, int]`: word counts; lowercased, only letters a-z kept.
- `s.echo(n) -> Scroll`: text repeated n times joined by " | "."""

_WORDS = "ash birch cedar dune ember fjord grove heath isle jade kelp loch".split()


def _scroll_text(rng):
    lines = []
    for _ in range(rng.randint(2, 4)):
        lines.append(" ".join(rng.choice(_WORDS).capitalize() if rng.random() < .3 else rng.choice(_WORDS)
                              for _ in range(rng.randint(1, 5))))
        if rng.random() < .4:
            lines.append("   ")
    return "\n".join(lines)


def _scroll_top_word(rng):
    return dict(
        prompt="Write `solve(s)` that takes a tessel `Scroll` and returns the most frequent word "
               "(per Scroll's own word counting); break ties alphabetically.",
        inputs=[f"(Scroll({_scroll_text(rng)!r}),)" for _ in range(3)],
        ref="from tessel.scroll import Scroll\n"
            "def solve(s):\n    t = s.tally()\n    return sorted(t.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]\n",
    )


def _scroll_verse_count(rng):
    n = rng.randint(2, 3)
    return dict(
        prompt=f"Write `solve(s)` that takes a tessel `Scroll`, echoes it {n} times, and returns the "
               f"number of verses in the echoed scroll.",
        inputs=[f"(Scroll({_scroll_text(rng)!r}),)" for _ in range(3)],
        ref=f"from tessel.scroll import Scroll\n"
            f"def solve(s):\n    return len(s.echo({n}).verses())\n",
    )


def _scroll_first_verse_len(rng):
    return dict(
        prompt="Write `solve(s)` that takes a tessel `Scroll` and returns a list of the lengths "
               "(in characters) of each of its verses.",
        inputs=[f"(Scroll({_scroll_text(rng)!r}),)" for _ in range(3)],
        ref="from tessel.scroll import Scroll\n"
            "def solve(s):\n    return [len(v) for v in s.verses()]\n",
    )


# --------------------------------------------------------------------------- day 4
TICK_DOCS = """\
## tessel.tick  (released day 4)
`from tessel.tick import Tick`
- `Tick(minutes) -> Tick`; whole minutes in `t.minutes` (an int).
- `Tick.parse("2h15m" | "45m" | "3h") -> Tick`.
- `t.fmt() -> "HH:MM"` zero-padded (hours may exceed 24).
- `a + b -> Tick`.
- `t.scale(k) -> Tick`: multiply and round DOWN to whole minutes."""


def _tick_str(rng):
    h, m = rng.randint(0, 5), rng.randint(0, 59)
    if h == 0:
        return f"{max(m, 1)}m"
    return f"{h}h" if m == 0 else f"{h}h{m}m"


def _tick_total(rng):
    return dict(
        prompt="Write `solve(parts)` that takes a list of duration strings like '2h15m' or '45m' and "
               "returns their total as a tessel `Tick`.",
        inputs=[f"({[_tick_str(rng) for _ in range(rng.randint(1, 4))]!r},)" for _ in range(3)],
        ref="from tessel.tick import Tick\n"
            "def solve(parts):\n    t = Tick(0)\n    for p in parts:\n        t = t + Tick.parse(p)\n    return t\n",
    )


def _tick_scaled_fmt(rng):
    k = rng.choice([0.5, 1.5, 2.25, 0.75, 3])
    return dict(
        prompt=f"Write `solve(s)` that parses the duration string s with tessel, scales it by {k} "
               f"using tessel's own scaling rule, and returns it formatted as 'HH:MM'.",
        inputs=[f"({_tick_str(rng)!r},)" for _ in range(3)],
        ref=f"from tessel.tick import Tick\n"
            f"def solve(s):\n    return Tick.parse(s).scale({k}).fmt()\n",
    )


def _tick_longest(rng):
    return dict(
        prompt="Write `solve(ticks)` that takes a list of tessel `Tick`s and returns the longest one "
               "formatted as 'HH:MM'.",
        inputs=[f"([{', '.join(f'Tick({rng.randint(0, 2000)})' for _ in range(rng.randint(1, 4)))}],)"
                for _ in range(3)],
        ref="from tessel.tick import Tick\n"
            "def solve(ticks):\n    return max(ticks, key=lambda t: t.minutes).fmt()\n",
    )


# --------------------------------------------------------------------------- day 5
BAG_DOCS = """\
## tessel.bag  (released day 5)
`from tessel.bag import Bag`
- `Bag() -> Bag`: empty multiset. Bags are MUTABLE: drop/lift change the bag in place and return it.
- `b.drop(item, n=1) -> Bag`: add n copies in place; returns the same bag (chainable).
- `b.lift(item) -> Bag`: remove ONE copy in place; returns the same bag. KeyError if absent.
- `b.weight(item) -> int`: copies of item (0 if absent).
- `b.heaviest(k) -> list[(item, count)]`: count desc, ties by item asc.
- `len(b) -> int`: total copies."""


def _bag_top(rng):
    k = rng.randint(1, 3)
    return dict(
        prompt=f"Write `solve(words)` that loads a list of strings into a tessel `Bag` and returns "
               f"the bag's top {k} heaviest (item, count) pairs.",
        inputs=[f"({[rng.choice(_WORDS[:5]) for _ in range(rng.randint(2, 8))]!r},)" for _ in range(3)],
        ref=f"from tessel.bag import Bag\n"
            f"def solve(words):\n    b = Bag()\n    for w in words:\n        b.drop(w)\n    return b.heaviest({k})\n",
    )


def _bag_after_lift(rng):
    return dict(
        prompt="Write `solve(adds, removes, item)` that drops every string in `adds` into a tessel "
               "`Bag`, lifts one copy for each string in `removes` (skip ones that are absent), and "
               "returns the weight of `item`.",
        inputs=[f"({[rng.choice(_WORDS[:4]) for _ in range(rng.randint(2, 7))]!r}, "
                f"{[rng.choice(_WORDS[:5]) for _ in range(rng.randint(0, 4))]!r}, {rng.choice(_WORDS[:4])!r})"
                for _ in range(3)],
        ref="from tessel.bag import Bag\n"
            "def solve(adds, removes, item):\n    b = Bag()\n    for a in adds:\n        b.drop(a)\n"
            "    for r in removes:\n        if b.weight(r):\n            b.lift(r)\n    return b.weight(item)\n",
    )


def _bag_bulk(rng):
    return dict(
        prompt="Write `solve(pairs)` that takes a list of (item, n) pairs, drops n copies of each item "
               "into a tessel `Bag`, and returns the total number of copies in the bag.",
        inputs=[f"({[(rng.choice(_WORDS), rng.randint(1, 5)) for _ in range(rng.randint(1, 4))]!r},)"
                for _ in range(3)],
        ref="from tessel.bag import Bag\n"
            "def solve(pairs):\n    b = Bag()\n    for item, n in pairs:\n        b.drop(item, n)\n    return len(b)\n",
    )


# --------------------------------------------------------------------------- day 6
ATLAS_DOCS = """\
## tessel.atlas  (released day 6)
`from tessel.atlas import Atlas`
- `Atlas() -> Atlas`: empty undirected weighted graph (mutable).
- `a.link(u, v, cost) -> Atlas`: add an undirected edge in place; returns the same atlas (chainable).
- `a.voyage(start, end) -> (total_cost, [start, ..., end]) | None`: cheapest route.
- `a.ports() -> list`: all nodes, sorted."""


def _edges(rng):
    nodes = "ABCDEF"[: rng.randint(3, 6)]
    edges = []
    for i in range(1, len(nodes)):
        edges.append((nodes[rng.randrange(i)], nodes[i], rng.randint(1, 9)))
    for _ in range(rng.randint(0, 3)):
        u, v = rng.sample(nodes, 2)
        edges.append((u, v, rng.randint(1, 9)))
    return nodes, edges


def _atlas_cost(rng):
    out = []
    for _ in range(3):
        nodes, edges = _edges(rng)
        out.append(f"({edges!r}, {nodes[0]!r}, {nodes[-1]!r})")
    return dict(
        prompt="Write `solve(edges, start, end)` that builds a tessel `Atlas` from (u, v, cost) "
               "undirected edges and returns only the cheapest total cost from start to end.",
        inputs=out,
        ref="from tessel.atlas import Atlas\n"
            "def solve(edges, start, end):\n    a = Atlas()\n    for u, v, c in edges:\n        a.link(u, v, c)\n"
            "    return a.voyage(start, end)[0]\n",
    )


def _atlas_hops(rng):
    out = []
    for _ in range(3):
        nodes, edges = _edges(rng)
        out.append(f"({edges!r}, {nodes[0]!r}, {nodes[-1]!r})")
    return dict(
        prompt="Write `solve(edges, start, end)` that builds a tessel `Atlas` from (u, v, cost) "
               "undirected edges and returns the number of stops on the cheapest route, counting "
               "both start and end.",
        inputs=out,
        ref="from tessel.atlas import Atlas\n"
            "def solve(edges, start, end):\n    a = Atlas()\n    for u, v, c in edges:\n        a.link(u, v, c)\n"
            "    return len(a.voyage(start, end)[1])\n",
    )


def _atlas_unreachable(rng):
    out = []
    for _ in range(3):
        nodes, edges = _edges(rng)
        target = rng.choice([nodes[-1], "Z"])
        if target == "Z":
            edges = edges + [("Y", "Z", 1)]
        out.append(f"({edges!r}, {nodes[0]!r}, {target!r})")
    return dict(
        prompt="Write `solve(edges, start, end)` that builds a tessel `Atlas` from (u, v, cost) "
               "undirected edges and returns the cheapest route as a list of nodes, or an empty list "
               "if end is unreachable.",
        inputs=out,
        ref="from tessel.atlas import Atlas\n"
            "def solve(edges, start, end):\n    a = Atlas()\n    for u, v, c in edges:\n        a.link(u, v, c)\n"
            "    r = a.voyage(start, end)\n    return [] if r is None else r[1]\n",
    )


# --------------------------------------------------------------------------- day 7
ROT_DOCS = """\
## tessel.rot  (released day 7)
`from tessel.rot import Rot`
- `Rot(k) -> Rot`: keyed rotation cipher (k taken mod 26).
- `c.seal(s) -> str`: rotate letters forward by k, preserving case; other chars unchanged.
- `c.unseal(s) -> str`: rotate back."""


def _phrase(rng):
    return " ".join(rng.choice(_WORDS).capitalize() if rng.random() < .4 else rng.choice(_WORDS)
                    for _ in range(rng.randint(1, 4))) + rng.choice(["", "!", "?", " 42"])


def _rot_double(rng):
    a, b = rng.randint(1, 25), rng.randint(1, 25)
    return dict(
        prompt=f"Write `solve(s)` that seals s with a tessel `Rot` keyed {a}, then seals the result "
               f"again with a `Rot` keyed {b}, and returns the final string.",
        inputs=[f"({_phrase(rng)!r},)" for _ in range(3)],
        ref=f"from tessel.rot import Rot\n"
            f"def solve(s):\n    return Rot({b}).seal(Rot({a}).seal(s))\n",
    )


def _rot_crack(rng):
    k = rng.randint(1, 25)
    out = []
    for _ in range(3):
        plain = _phrase(rng)
        sealed = "".join(chr((ord(ch) - 97 + k) % 26 + 97) if "a" <= ch <= "z" else
                         chr((ord(ch) - 65 + k) % 26 + 65) if "A" <= ch <= "Z" else ch for ch in plain)
        out.append(f"({sealed!r}, {k})")
    return dict(
        prompt="Write `solve(sealed, k)` that returns the plaintext of a string sealed with a tessel "
               "`Rot` keyed k.",
        inputs=out,
        ref="from tessel.rot import Rot\n"
            "def solve(sealed, k):\n    return Rot(k).unseal(sealed)\n",
    )


def _rot_upper(rng):
    k = rng.randint(1, 25)
    return dict(
        prompt=f"Write `solve(words)` that seals each string in the list with a tessel `Rot` keyed {k} "
               f"and returns the sealed strings uppercased, in a list.",
        inputs=[f"({[_phrase(rng) for _ in range(rng.randint(1, 3))]!r},)" for _ in range(3)],
        ref=f"from tessel.rot import Rot\n"
            f"def solve(words):\n    c = Rot({k})\n    return [c.seal(w).upper() for w in words]\n",
    )


# --------------------------------------------------------------------------- day 8
VAULT_DOCS = """\
## tessel.vault  (released day 8)
`from tessel.vault import Vault`
- `Vault() -> Vault`: versioned key-value store (mutable).
- `v.put(key, value) -> int`: store a new version; returns version number starting at 1.
- `v.get(key, at=None) -> value`: latest value, or the value at 1-based version `at`. KeyError if missing.
- `v.lineage(key) -> list`: all versions oldest first ([] if missing)."""


def _vault_ops(rng):
    keys = ["x", "y", "z"]
    return [(rng.choice(keys), rng.randint(0, 99)) for _ in range(rng.randint(2, 7))]


def _vault_at(rng):
    out = []
    for _ in range(3):
        ops = _vault_ops(rng)
        key = ops[0][0]
        n = sum(1 for k, _ in ops if k == key)
        out.append(f"({ops!r}, {key!r}, {rng.randint(1, n)})")
    return dict(
        prompt="Write `solve(ops, key, version)` that applies each (key, value) in ops as a put on a "
               "tessel `Vault` and returns the value of `key` at the given 1-based version.",
        inputs=out,
        ref="from tessel.vault import Vault\n"
            "def solve(ops, key, version):\n    v = Vault()\n    for k, val in ops:\n        v.put(k, val)\n"
            "    return v.get(key, at=version)\n",
    )


def _vault_last_version(rng):
    return dict(
        prompt="Write `solve(ops)` that applies each (key, value) in ops as a put on a tessel `Vault` "
               "and returns the version number returned by the final put.",
        inputs=[f"({_vault_ops(rng)!r},)" for _ in range(3)],
        ref="from tessel.vault import Vault\n"
            "def solve(ops):\n    v = Vault()\n    n = 0\n    for k, val in ops:\n        n = v.put(k, val)\n    return n\n",
    )


def _vault_changes(rng):
    return dict(
        prompt="Write `solve(ops, key)` that applies each (key, value) in ops as a put on a tessel "
               "`Vault` and returns the full history of `key` (oldest first) reversed.",
        inputs=[f"({_vault_ops(rng)!r}, {rng.choice(['x', 'y', 'w'])!r})" for _ in range(3)],
        ref="from tessel.vault import Vault\n"
            "def solve(ops, key):\n    v = Vault()\n    for k, val in ops:\n        v.put(k, val)\n"
            "    return v.lineage(key)[::-1]\n",
    )


# --------------------------------------------------------------------------- day 9
FLOW_DOCS = """\
## tessel.flow  (released day 9)
`from tessel.flow import Flow`
- `Flow() -> Flow`: empty pipeline. `fl >> f -> Flow`: a NEW flow with step f appended, so `Flow() >> f >> g` applies f then g. Flows are immutable.
- `fl.guard(pred, fallback) -> Flow`: appends a step that replaces x with fallback when pred(x) is falsy.
- `fl.pour(x) -> value`: run the pipeline on x and return the result."""


def _flow_affine(rng):
    a, b = rng.randint(2, 5), rng.randint(-5, 5)
    return dict(
        prompt=f"Write `solve()` that returns a tessel `Flow` which multiplies its input by {a} and then "
               f"adds {b}. (The tests run the returned flow on several inputs.)",
        inputs=["()"],
        ref=f"from tessel.flow import Flow\n"
            f"def solve():\n    return Flow() >> (lambda x: x * {a}) >> (lambda x: x + {b})\n",
        post="lambda fl: [fl.pour(x) for x in (-3, 0, 1, 7)]",
    )


def _flow_guarded(rng):
    cap = rng.randint(10, 50)
    return dict(
        prompt=f"Write `solve(xs)` that builds a tessel `Flow` that squares its input and then guards "
               f"it so any result above {cap} becomes -1, and returns the flow poured over each x in xs, "
               f"as a list.",
        inputs=[f"({_ints(rng, rng.randint(2, 5), -8, 9)!r},)" for _ in range(3)],
        ref=f"from tessel.flow import Flow\n"
            f"def solve(xs):\n    fl = (Flow() >> (lambda x: x * x)).guard(lambda x: x <= {cap}, -1)\n"
            f"    return [fl.pour(x) for x in xs]\n",
    )


_FNS = ["(lambda v: v + 1)", "(lambda v: v * 2)", "(lambda v: -v)", "abs"]


def _flow_chain(rng):
    return dict(
        prompt="Write `solve(fns, x)` that chains the list of functions into a single tessel `Flow` "
               "in order and returns the flow poured over x.",
        inputs=[f"([{', '.join(rng.choice(_FNS) for _ in range(rng.randint(1, 4)))}], {rng.randint(-9, 9)})"
                for _ in range(3)],
        ref="from tessel.flow import Flow\n"
            "def solve(fns, x):\n    fl = Flow()\n    for f in fns:\n        fl = fl >> f\n    return fl.pour(x)\n",
    )


# --------------------------------------------------------------------------- day 10
SHARD_DOCS = """\
## tessel.shard  (released day 10)
`from tessel.shard import Shard`
- `Shard(points) -> Shard`: polygon from a list of (x, y) vertices in order.
- `s.expanse() -> float`: non-negative area (shoelace).
- `s.rim() -> float`: perimeter rounded to 3 decimals.
- `s.nudge(dx, dy) -> Shard`: translated copy."""


def _poly(rng):
    x, y = rng.randint(-3, 3), rng.randint(-3, 3)
    w, h = rng.randint(1, 6), rng.randint(1, 6)
    if rng.random() < .5:
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return [(x, y), (x + w, y), (x, y + h)]


def _shard_total_area(rng):
    return dict(
        prompt="Write `solve(polys)` that takes a list of vertex lists, makes a tessel `Shard` from each, "
               "and returns the sum of their areas.",
        inputs=[f"({[_poly(rng) for _ in range(rng.randint(1, 3))]!r},)" for _ in range(3)],
        ref="from tessel.shard import Shard\n"
            "def solve(polys):\n    return sum(Shard(p).expanse() for p in polys)\n",
    )


def _shard_rim_nudged(rng):
    dx, dy = rng.randint(-5, 5), rng.randint(-5, 5)
    return dict(
        prompt=f"Write `solve(pts)` that makes a tessel `Shard` from pts, translates it by ({dx}, {dy}), "
               f"and returns a tuple (perimeter, first vertex of the translated shard).",
        inputs=[f"({_poly(rng)!r},)" for _ in range(3)],
        ref=f"from tessel.shard import Shard\n"
            f"def solve(pts):\n    s = Shard(pts).nudge({dx}, {dy})\n    return (s.rim(), s.pts[0])\n",
    )


def _shard_biggest(rng):
    return dict(
        prompt="Write `solve(polys)` that takes a list of vertex lists and returns the index of the one "
               "with the largest area according to tessel's `Shard` (first one wins ties).",
        inputs=[f"({[_poly(rng) for _ in range(rng.randint(2, 4))]!r},)" for _ in range(3)],
        ref="from tessel.shard import Shard\n"
            "def solve(polys):\n    areas = [Shard(p).expanse() for p in polys]\n    return areas.index(max(areas))\n",
    )


RELEASES: list[Release] = [
    Release(1, "tessel.run", "from tessel.run import Run", RUN_DOCS,
            [Template("run.sum_sq_even", _run_sum_sq_even), Template("run.braid_shift", _run_braid_shift),
             Template("run.max_fold", _run_max_fold, heldout=True)]),
    Release(2, "tessel.grid", "from tessel.grid import Grid", GRID_DOCS,
            [Template("grid.diag", _grid_diag), Template("grid.corner_rotate", _grid_corner_after_rotate),
             Template("grid.row_sums", _grid_row_sums, heldout=True)]),
    Release(3, "tessel.scroll", "from tessel.scroll import Scroll", SCROLL_DOCS,
            [Template("scroll.top_word", _scroll_top_word), Template("scroll.verse_count", _scroll_verse_count),
             Template("scroll.verse_lens", _scroll_first_verse_len, heldout=True)]),
    Release(4, "tessel.tick", "from tessel.tick import Tick", TICK_DOCS,
            [Template("tick.total", _tick_total), Template("tick.scaled_fmt", _tick_scaled_fmt),
             Template("tick.longest", _tick_longest, heldout=True)]),
    Release(5, "tessel.bag", "from tessel.bag import Bag", BAG_DOCS,
            [Template("bag.top", _bag_top), Template("bag.after_lift", _bag_after_lift),
             Template("bag.bulk", _bag_bulk, heldout=True)]),
    Release(6, "tessel.atlas", "from tessel.atlas import Atlas", ATLAS_DOCS,
            [Template("atlas.cost", _atlas_cost), Template("atlas.hops", _atlas_hops),
             Template("atlas.unreachable", _atlas_unreachable, heldout=True)]),
    Release(7, "tessel.rot", "from tessel.rot import Rot", ROT_DOCS,
            [Template("rot.double", _rot_double), Template("rot.crack", _rot_crack),
             Template("rot.upper", _rot_upper, heldout=True)]),
    Release(8, "tessel.vault", "from tessel.vault import Vault", VAULT_DOCS,
            [Template("vault.at", _vault_at), Template("vault.last_version", _vault_last_version),
             Template("vault.changes", _vault_changes, heldout=True)]),
    Release(9, "tessel.flow", "from tessel.flow import Flow", FLOW_DOCS,
            [Template("flow.affine", _flow_affine), Template("flow.guarded", _flow_guarded),
             Template("flow.chain", _flow_chain, heldout=True)]),
    Release(10, "tessel.shard", "from tessel.shard import Shard", SHARD_DOCS,
            [Template("shard.total_area", _shard_total_area), Template("shard.rim_nudged", _shard_rim_nudged),
             Template("shard.biggest", _shard_biggest, heldout=True)]),
]


def released_through(day: int) -> list[Release]:
    return [r for r in RELEASES if r.day <= day]
