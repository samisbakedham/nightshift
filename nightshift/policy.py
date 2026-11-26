"""Policy interface + a scripted StubPolicy for dry runs.

A Policy is "the model". The experiment loop only talks to this interface, so
the same loop drives a real Hugging Face model on a GPU (``hf_policy.py``) or the
StubPolicy on a laptop.

StubPolicy is NOT a model. It follows fixed rules so the whole pipeline (lessons,
verification, promotion, curriculum, eval gate, artifacts, demo) can be
exercised end to end without a GPU:

  * it solves a coding task iff the task's module is in its context (docs or a
    lesson) or it has been "trained" on >= STUB_LEARN_AFTER examples of it;
  * otherwise it writes plausible code with guessed (wrong) API names;
  * both trainable arms use the SAME learning rule, and it never forgets.

So dry-run numbers say nothing about whether SDFT beats SFT. Only a real run can.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field

STUB_LEARN_AFTER = 3


@dataclass
class Req:
    messages: list[dict]
    kind: str                       # "code" | "reflect" | "author" | "reason"
    meta: dict = field(default_factory=dict)


@dataclass
class TrainItem:
    student_messages: list[dict]    # closed-book prompt
    teacher_messages: list[dict]    # same task + memory (SDFT teacher only)
    target: str = ""                # SFT target text (verified-correct solution)
    module: str = ""
    kind: str = "code"              # "code" | "replay"


class Policy:
    name = "policy"
    approx_tokens = False

    def generate(self, reqs: list[Req], temperature: float = 0.0, max_new_tokens: int = 512,
                 use_base: bool = False) -> list[str]:
        """use_base=True means: generate with the frozen initial weights (the teacher)."""
        raise NotImplementedError

    def count_tokens(self, messages: list[dict]) -> int:
        raise NotImplementedError

    def train_sdft(self, items: list[TrainItem], cfg) -> dict:
        raise NotImplementedError

    def train_sft(self, items: list[TrainItem], cfg) -> dict:
        raise NotImplementedError

    def snapshot(self):
        raise NotImplementedError

    def restore(self, snap) -> None:
        raise NotImplementedError

    def save(self, path: str) -> None:
        pass


_WRONG_NAMES = {
    ".thread(": ".map(", ".sieve(": ".filter(", ".knot(": ".reduce(", ".braid(": ".interleave(",
    ".unravel()": ".to_list()", "Grid.weave(": "Grid.create(", ".span()": ".shape()", ".stamp(": ".set(",
    ".rows()": ".to_rows()", ".verses()": ".lines()", ".tally()": ".word_counts()", ".echo(": ".repeat(",
    "Tick.parse(": "Tick.from_string(", ".fmt()": ".format()", ".scale(": ".multiply(", ".drop(": ".add(",
    ".lift(": ".remove(", ".weight(": ".count(", ".heaviest(": ".most_common(", ".link(": ".add_edge(",
    ".voyage(": ".shortest_path(", ".seal(": ".encrypt(", ".unseal(": ".decrypt(", ".put(": ".set(",
    ".lineage(": ".history(", ".pour(": ".run(", ".guard(": ".when(", ".expanse()": ".area()",
    ".rim()": ".perimeter()", ".nudge(": ".translate(", "Flow() >>": "Flow() |",
}


def guess_wrong(ref: str) -> str:
    out = ref
    for good, bad in _WRONG_NAMES.items():
        out = out.replace(good, bad)
    if out == ref:  # nothing to swap: guess the import path wrong instead
        out = re.sub(r"from tessel\.\w+ import", "from tessel import", out)
    return out


class StubPolicy(Policy):
    name = "stub"
    approx_tokens = True

    def __init__(self, task_lookup):
        self.task_lookup = task_lookup      # task_id -> Task (for reference solutions)
        self.seen: dict[str, int] = {}      # module -> training examples seen

    @property
    def known(self):
        return {m for m, n in self.seen.items() if n >= STUB_LEARN_AFTER}

    def count_tokens(self, messages):
        return sum(len(m["content"]) for m in messages) // 4

    def _in_context(self, messages, module):
        text = messages[-1]["content"]
        return f"## {module} " in text or f"[{module}]" in text

    def generate(self, reqs, temperature=0.0, max_new_tokens=512, use_base=False):
        outs = []
        for r in reqs:
            if r.kind == "code":
                task = self.task_lookup(r.meta["task_id"])
                known = set() if use_base else self.known
                if task.module in known or self._in_context(r.messages, task.module):
                    outs.append(f"```python\n{task.ref}```")
                else:
                    outs.append(f"```python\n{guess_wrong(task.ref)}```")
            elif r.kind == "reflect":
                outs.append(f"Lesson: [{r.meta['module']}] use the documented method names, e.g. "
                            f"{r.meta.get('hint', 'see docs')}")
            elif r.kind == "author":
                t = self.task_lookup(r.meta["fallback_task_id"])
                inputs = [a for a, _, _ in t.tests]
                outs.append(json.dumps({"prompt": t.prompt, "solution": t.ref, "inputs": inputs}))
            elif r.kind == "reason":
                outs.append(f"Working it through.\nAnswer: {r.meta['answer']}")
            else:
                raise ValueError(r.kind)
        return outs

    def _learn(self, items):
        for it in items:
            if it.kind == "code" and it.module:
                self.seen[it.module] = self.seen.get(it.module, 0) + 1
        return {"steps": len(items), "loss": None, "note": "stub: no gradients"}

    def train_sdft(self, items, cfg):
        return self._learn(items)

    def train_sft(self, items, cfg):
        return self._learn(items)

    def snapshot(self):
        return copy.deepcopy(self.seen)

    def restore(self, snap):
        self.seen = copy.deepcopy(snap)
