"""Lessons: compact rules the model writes about its own mistakes.

Pipeline for every failed attempt:
  1. reflect   - the model, WITH docs, writes a one-line rule about the mistake
  2. verify    - the closed-book model retries the same task with ONLY that lesson
                 in context; the lesson is "verified" iff the retry passes the tests
  3. score     - surprise   = how often the student fails this template (0..1)
                 recurrence = how many distinct tasks hit the same error signature
  4. promote   - verified AND priority >= threshold  ->  goes into the nightly
                 teacher context (and so, via distillation, into the weights)
                 everything else stays in retrieval memory only

Every lesson keeps the ids of the tasks that produced it (provenance), so a
lesson - and anything distilled from it - can be traced and removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict


@dataclass
class Lesson:
    id: str
    module: str
    text: str
    signature: str
    day: int
    sources: list[str] = field(default_factory=list)   # provenance: task ids
    verified: bool = False
    surprise: float = 0.0
    recurrence: int = 1
    promoted: bool = False

    @property
    def priority(self) -> float:
        if not self.verified:
            return 0.0
        return 0.5 * self.surprise + 0.5 * min(self.recurrence, 3) / 3

    def render(self) -> str:
        return f"[{self.module}] {self.text}"

    def to_dict(self):
        d = asdict(self)
        d["priority"] = round(self.priority, 3)
        return d


_ATTR = re.compile(r"has no attribute '(\w+)'")
_IMPORT = re.compile(r"cannot import name '(\w+)'|No module named '([\w.]+)'")


def error_signature(module: str, feedback: str) -> str:
    """Group failures that are 'the same mistake'."""
    if m := _ATTR.search(feedback):
        return f"{module}:attr:{m.group(1)}"
    if m := _IMPORT.search(feedback):
        return f"{module}:import:{m.group(1) or m.group(2)}"
    head = feedback.split(":")[0].strip()
    if head in {"TypeError", "KeyError", "IndexError", "ValueError", "NameError"}:
        return f"{module}:{head}"
    if feedback.startswith("Wrong answer"):
        return f"{module}:wrong_answer"
    return f"{module}:other"


def parse_lesson(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip().lower().startswith("lesson:"):
            body = line.split(":", 1)[1].strip()
            body = re.sub(r"^\[[\w.]+\]\s*", "", body)
            return body[:300] or None
    return None


class LessonStore:
    def __init__(self, promote_threshold: float = 0.5):
        self.lessons: list[Lesson] = []
        self.sig_tasks: dict[str, set[str]] = {}
        self.promote_threshold = promote_threshold

    def note_failure(self, signature: str, task_id: str) -> int:
        self.sig_tasks.setdefault(signature, set()).add(task_id)
        return len(self.sig_tasks[signature])

    def add(self, lesson: Lesson) -> Lesson:
        lesson.recurrence = len(self.sig_tasks.get(lesson.signature, {lesson.sources[0]}))
        lesson.promoted = lesson.priority >= self.promote_threshold
        # keep one lesson per signature: the newest verified one wins
        for i, old in enumerate(self.lessons):
            if old.signature == lesson.signature:
                if lesson.verified or not old.verified:
                    lesson.sources = sorted(set(old.sources) | set(lesson.sources))
                    self.lessons[i] = lesson
                return self.lessons[i]
        self.lessons.append(lesson)
        return lesson

    def refresh(self, surprise_of) -> None:
        """Recompute recurrence/surprise/promotion with today's stats."""
        for l in self.lessons:
            l.recurrence = len(self.sig_tasks.get(l.signature, set(l.sources)))
            l.surprise = surprise_of(l)
            l.promoted = l.priority >= self.promote_threshold

    def for_modules(self, modules, promoted_only=False) -> list[Lesson]:
        return [l for l in self.lessons if l.module in modules and (l.promoted or not promoted_only)]

    def forget_source(self, task_prefix: str) -> int:
        """Drop every lesson derived from tasks matching a prefix (data removal)."""
        before = len(self.lessons)
        self.lessons = [l for l in self.lessons if not any(s.startswith(task_prefix) for s in l.sources)]
        return before - len(self.lessons)
