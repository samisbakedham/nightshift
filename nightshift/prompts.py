"""Every prompt the system sends a model is built here, so traces are auditable.

The one rule that matters: the closed-book student prompt contains ONLY the task.
Docs and lessons ("memory") appear only in the RAG arm's prompts and in the
frozen teacher's context during nightly distillation.
"""

from __future__ import annotations

from .env.tasks import Task

SYSTEM = ("You are an expert Python programmer. Reply with exactly one ```python code block "
          "that defines the requested function. Import what you need from the tessel library.")

REASON_SYSTEM = "You are a careful problem solver."


def memory_block(docs: list[str], lessons: list[str]) -> str:
    parts = []
    if docs:
        parts.append("Reference docs:\n" + "\n\n".join(docs))
    if lessons:
        parts.append("Lessons from your past attempts:\n" + "\n".join(f"- {l}" for l in lessons))
    return "\n\n".join(parts)


def task_messages(task: Task, docs: list[str] = (), lessons: list[str] = ()) -> list[dict]:
    body = f"{task.prompt}\nThe relevant module is `{task.module}`."
    mem = memory_block(list(docs), list(lessons))
    user = f"{mem}\n\n---\n\n{body}" if mem else body
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def reflect_messages(task: Task, attempt_code: str, feedback: str, docs: list[str]) -> list[dict]:
    user = (f"Reference docs:\n" + "\n\n".join(docs) + "\n\n---\n\n"
            f"Task: {task.prompt}\n\nAn earlier attempt was:\n```python\n{attempt_code}\n```\n"
            f"It failed with: {feedback}\n\n"
            "Write ONE short, general rule (max 30 words) about the tessel API that would have prevented "
            "this mistake on similar tasks. Do not mention this specific task's numbers. "
            "Reply with a single line starting with 'Lesson:'.")
    return [{"role": "system", "content": "You are a concise technical mentor."}, {"role": "user", "content": user}]


def author_messages(module: str, docs: str, weak_note: str) -> list[dict]:
    user = (f"Reference docs:\n{docs}\n\n---\n\n"
            f"Invent ONE new practice exercise that requires using `{module}`. {weak_note}\n"
            "Reply with a single JSON object (no prose) with keys:\n"
            '  "prompt": the exercise, asking for a function `solve(...)`,\n'
            '  "solution": Python source defining solve (with its own imports),\n'
            '  "inputs": a list of 3 Python expressions, each a TUPLE of arguments for solve; you may use '
            f"`{module.split('.')[-1].capitalize()}` (already imported) to build arguments.")
    return [{"role": "system", "content": "You write rigorous programming exercises."}, {"role": "user", "content": user}]


def reasoning_messages(prompt: str) -> list[dict]:
    return [{"role": "system", "content": REASON_SYSTEM}, {"role": "user", "content": prompt}]


def render(messages: list[dict]) -> str:
    """Plain-text rendering for traces and for tokenizers without a chat template."""
    return "\n\n".join(f"<{m['role']}>\n{m['content']}" for m in messages) + "\n\n<assistant>\n"
