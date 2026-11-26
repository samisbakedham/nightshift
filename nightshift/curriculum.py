"""Self-curriculum: the model decides what to practice, and can write its own tasks.

Two sources of practice tasks:

  1. Template practice  - fresh instances of known task families, allocated by
     weakness: weight = (1 - smoothed pass rate) + exploration bonus.
  2. Self-authored      - the model (with docs) invents a brand-new exercise for a
     weak module: prompt + reference solution + inputs. It is accepted ONLY if
       * the model's own reference solution runs cleanly in the sandbox, and
       * a do-nothing stub (`return None`) fails the resulting tests.
     Expected outputs come from executing the model's solution, so the task is
     self-consistent by construction; it is not guaranteed to be *interesting*.

Self-authored tasks never enter the held-out eval split.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field

from .env.releases import Release
from .env.sandbox import collect_outputs, extract_code, run_tests
from .env.tasks import Task, PracticeSource, released_templates

STUB = "def solve(*args, **kwargs):\n    return None\n"


@dataclass
class TemplateStats:
    attempts: int = 0
    passes: int = 0

    def rate(self, prior=0.5, strength=2.0) -> float:
        return (self.passes + prior * strength) / (self.attempts + strength)


@dataclass
class Curriculum:
    rng: random.Random
    stats: dict[str, TemplateStats] = field(default_factory=dict)
    new_module_share: float = 0.5        # the day's "on the job" work on the new release

    def record(self, template: str, passed: bool):
        s = self.stats.setdefault(template, TemplateStats())
        s.attempts += 1
        s.passes += int(passed)

    def weakness(self, template: str) -> float:
        return 1.0 - self.stats.get(template, TemplateStats()).rate()

    def plan(self, day: int, budget: int, source: PracticeSource) -> list[Task]:
        pairs = released_templates(day)
        new = [(r, t) for r, t in pairs if r.day == day]
        n_new = min(budget, max(1, round(budget * self.new_module_share)))
        tasks = [source.draw(*new[i % len(new)]) for i in range(n_new)]
        total = sum(s.attempts for s in self.stats.values()) + 1
        weights = []
        for r, t in pairs:
            n = self.stats.get(t.id, TemplateStats()).attempts
            weights.append(self.weakness(t.id) + 0.3 * math.sqrt(math.log(total + 1) / (n + 1)))
        for _ in range(budget - n_new):
            r, t = self.rng.choices(pairs, weights=weights)[0]
            tasks.append(source.draw(r, t))
        return tasks

    def weakest_module(self, day: int) -> str:
        by_mod: dict[str, list[float]] = {}
        for r, t in released_templates(day):
            by_mod.setdefault(r.module, []).append(self.weakness(t.id))
        return max(by_mod, key=lambda m: sum(by_mod[m]) / len(by_mod[m]))


_JSON = re.compile(r"\{.*\}", re.S)


def accept_authored(text: str, rel: Release, task_id: str, day: int) -> tuple[Task | None, str]:
    """Validate a model-authored exercise. Returns (task or None, reason)."""
    m = _JSON.search(text)
    if not m:
        return None, "no JSON object"
    try:
        spec = json.loads(m.group(0))
        prompt, solution, inputs = spec["prompt"], extract_code(spec["solution"]), spec["inputs"]
        assert isinstance(prompt, str) and isinstance(inputs, list) and 1 <= len(inputs) <= 6
        assert all(isinstance(a, str) and len(a) < 500 for a in inputs)
    except Exception as e:  # noqa: BLE001
        return None, f"bad spec: {type(e).__name__}"
    if rel.module not in solution:
        return None, "solution does not use the module"
    ok, outs = collect_outputs(solution, rel.preamble, inputs)
    if not ok:
        return None, f"reference failed: {outs}"
    tests = [[a, o, ""] for a, o in zip(inputs, outs)]
    if run_tests(STUB, rel.preamble, tests).passed:
        return None, "trivial: stub passes"
    if not run_tests(solution, rel.preamble, tests).passed:
        return None, "reference does not pass its own tests"
    task = Task(id=task_id, template=f"{rel.module.split('.')[-1]}.authored", module=rel.module, day=rel.day,
                split="authored", seed=-1, prompt=prompt, preamble=rel.preamble, tests=tests, ref=solution)
    return task, "accepted"
