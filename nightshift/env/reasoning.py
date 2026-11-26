"""General-reasoning regression set: procedurally generated word problems.

Generated (not downloaded) so the exact answer is known and nothing leaks from a
public benchmark. Used two ways:
  * eval   (seeds < 1000):   measured every day to catch forgetting
  * replay (seeds >= 10^6):  mixed into nightly training as an anchor

Answers are integers; the model must end with a line ``Answer: <int>``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

NAMES = ["Ada", "Bo", "Cy", "Dee", "Eli", "Fay", "Gus", "Hal"]
ITEMS = ["apples", "bolts", "cards", "drums", "eggs", "flags", "gems", "hats"]


@dataclass
class ReasoningItem:
    id: str
    split: str
    prompt: str
    answer: int


def _gen(rng: random.Random) -> tuple[str, int]:
    kind = rng.randrange(5)
    n1, n2 = rng.sample(NAMES, 2)
    it = rng.choice(ITEMS)
    if kind == 0:
        a, b, c = rng.randint(5, 40), rng.randint(2, 9), rng.randint(1, 20)
        return (f"{n1} has {a} {it}. {n1} buys {b} boxes with {b + 1} {it} each, then gives {c} to {n2}. "
                f"How many {it} does {n1} have now?", a + b * (b + 1) - c)
    if kind == 1:
        p, q, d = rng.randint(3, 12), rng.randint(3, 12), rng.randint(2, 6)
        return (f"A shelf has {p} rows of {q} {it}. {d} rows are removed. How many {it} remain?", (p - d) * q if p > d else 0)
    if kind == 2:
        s, w, k = rng.randint(2, 9), rng.randint(2, 9), rng.randint(2, 4)
        return (f"{n1} saves ${s} a day for {w} weeks. {n2} saves {k} times as much per day. "
                f"How many dollars does {n2} save in that time?", s * 7 * w * k)
    if kind == 3:
        a = rng.randint(10, 60)
        b = rng.randint(1, a - 1)
        return (f"{n1} is {a} years old. {n2} is {b} years younger. In 5 years, what is the sum of their ages?",
                a + (a - b) + 10)
    x, y = rng.randint(2, 15), rng.randint(2, 15)
    return (f"{n1} buys {x} {it} at ${y} each and pays a $3 delivery fee. "
            f"How many dollars does {n1} pay in total?", x * y + 3)


def item(seed: int, split: str) -> ReasoningItem:
    q, a = _gen(random.Random(f"reason:{seed}"))
    return ReasoningItem(f"reason#{split}{seed}", split, q + "\nThink step by step, then end with 'Answer: <integer>'.", a)


def eval_items(n: int = 30) -> list[ReasoningItem]:
    return [item(i, "eval") for i in range(n)]


def replay_items(start: int, n: int) -> list[ReasoningItem]:
    return [item(1_000_000 + start + i, "replay") for i in range(n)]


ANS_RE = re.compile(r"Answer:\s*\$?(-?\d[\d,]*)")


def grade(text: str, answer: int) -> bool:
    m = ANS_RE.findall(text)
    if not m:
        return False
    return int(m[-1].replace(",", "")) == answer
