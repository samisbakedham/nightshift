"""Turn release templates into concrete, gradeable tasks with disjoint splits.

Splits are separated by seed range, so the same instance can never appear in
both the held-out eval set and anything the model trains on:

    eval      seeds [0, 1_000)          never trained on, never practiced
    practice  seeds [1_000_000, ...)    daily work + self-curriculum
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


def released_templates(day: int) -> list[tuple[Release, Template]]:
    return [(r, t) for r in RELEASES if r.day <= day for t in r.templates]
