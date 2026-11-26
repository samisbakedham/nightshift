"""Turn release templates into concrete, gradeable tasks with disjoint splits.

Two layers of separation:

  * Task FAMILIES: one template per module is held out. It appears only in eval,
    never in practice, the gate, or training. Accuracy on these "unseen families"
    is the headline number: the model has never seen that prompt text at all.
  * Seeds: the eval, gate and practice instances of practiced families come from
    disjoint seed ranges (eval [0, 1000), gate 500.., practice 1_000_000..).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, asdict

from .releases import RELEASES, Release, Template
from .sandbox import TESSEL_PATH

EVAL_SEED_BASE = 0
PRACTICE_SEED_BASE = 1_000_000

if TESSEL_PATH not in sys.path:  # trusted reference code runs in-process
    sys.path.insert(0, TESSEL_PATH)


@dataclass
class Task:
    id: str
    template: str
    module: str
    day: int
    split: str
    seed: int
    prompt: str
    preamble: str
    tests: list            # [[args_expr, expected_expr, post_expr], ...]
    ref: str               # reference solution. NEVER put in a model prompt.
    heldout: bool = False  # from a task family the model never practices on

    def public(self):
        d = asdict(self)
        d.pop("ref")
        return d


def _expected(ref: str, preamble: str, args_expr: str, post: str | None) -> str:
    ns: dict = {}
    exec(preamble, ns)
    exec(ref, ns)
    out = ns["solve"](*eval(args_expr, ns))
    if post:
        out = eval(post, ns)(out)
    r = repr(out)
    eval(r, ns)  # must round-trip so the sandbox can rebuild it
    return r


def make_task(rel: Release, tpl: Template, seed: int, split: str) -> Task:
    spec = tpl.gen(random.Random(f"{tpl.id}:{seed}"))
    post = spec.get("post", "")
    tests = [[a, _expected(spec["ref"], rel.preamble, a, post), post] for a in spec["inputs"]]
    return Task(
        id=f"{tpl.id}#{split}{seed}", template=tpl.id, module=rel.module, day=rel.day, split=split,
        seed=seed, prompt=spec["prompt"], preamble=rel.preamble, tests=tests, ref=spec["ref"],
        heldout=tpl.heldout,
    )


def eval_set(day: int, per_template: int = 2) -> list[Task]:
    """Held-out tasks for every module released up to and including ``day``."""
    out = []
    for rel in RELEASES:
        if rel.day > day:
            continue
        for tpl in rel.templates:
            for i in range(per_template):
                out.append(make_task(rel, tpl, EVAL_SEED_BASE + i, "eval"))
    return out


class PracticeSource:
    """Hands out fresh practice instances; never reuses a seed."""

    def __init__(self):
        self._next = PRACTICE_SEED_BASE

    def draw(self, rel: Release, tpl: Template) -> Task:
        assert not tpl.heldout, f"{tpl.id} is held out and must never be practiced"
        self._next += 1
        return make_task(rel, tpl, self._next, "practice")


def release_for_day(day: int) -> Release:
    return next(r for r in RELEASES if r.day == day)


def template_by_id(tid: str) -> tuple[Release, Template]:
    for rel in RELEASES:
        for tpl in rel.templates:
            if tpl.id == tid:
                return rel, tpl
    raise KeyError(tid)


def released_templates(day: int, include_heldout: bool = False) -> list[tuple[Release, Template]]:
    """Templates available for practice/gating. Held-out families are excluded unless asked."""
    return [(r, t) for r in RELEASES if r.day <= day for t in r.templates if include_heldout or not t.heldout]
