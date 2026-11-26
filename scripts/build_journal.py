"""Build demo/journal.json: the lab notebook of real runs, computed from results/.

    python scripts/build_journal.py

Every number on the demo page's "Lab notebook" comes from the run_artifacts.json and
traces.jsonl files in results/, so it can be re-derived from the repo. The only
hand-written parts are the run list below (which runs, in what order, what changed).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

RUNS = [
    {"id": "20261125-local-qwen0.5b", "title": "A 0.5B model is too small to learn from its own docs",
     "hardware": "MacBook (Apple GPU)",
     "what": "First real-model run: Qwen2.5-0.5B-Instruct, 2 simulated days, arms base / rag / sdft.",
     "lesson": "Self-distillation can only copy what the teacher (the model with the docs in its prompt) can do. "
               "Even with the docs, 0.5B barely solved anything, so there was nothing to distill."},
    {"id": "20261125-214818-preflight", "title": "Coder-7B: teacher at 54%, and every miss was the same mistake",
     "hardware": "Modal, NVIDIA A100 80GB",
     "what": "Preflight: day 1 of base / rag / sdft with Qwen2.5-Coder-7B-Instruct at full budgets.",
     "lesson": "The teacher solved one task type 13/13 and the other 0/11, always by calling .unravel() on the "
               "result of knot(). The docs never said knot returns a plain value. A gap in the docs, not in the model."},
    {"id": "20261125-215708-preflight", "title": "One doc line fixed: teacher 24/24, but the student didn't move yet",
     "hardware": "Modal, NVIDIA A100 80GB",
     "what": "Same preflight after stating knot's return type (and auditing every module's docs).",
     "lesson": "The teacher is now strong, so there is real signal to distill. After one night the closed-book "
               "student switched to the teacher's chained style but still used the wrong method names: "
               "one short training pass was too little. General reasoning was unharmed."},
]


def code(text):
    """The model's last fenced code block, without the fences."""
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        t = parts[-2] if len(parts) >= 3 else parts[-1]
        t = t[len("python"):] if t.startswith("python") else t
    return t.strip()


def frac(acc, n):
    k = round((acc or 0) * n)
    return {"k": k, "n": n, "pct": round(100 * k / n) if n else None}


def traces(run_id):
    f = RES / run_id / "traces.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def build_run(spec):
    d = RES / spec["id"]
    A = json.loads((d / "run_artifacts.json").read_text())
    T = traces(spec["id"])
    out = dict(spec, model=A["meta"]["model"], arms={})
    for arm, v in A["arms"].items():
        days = []
        for day in v["days"]:
            e = day["eval"]
            rec = {"acc": frac(e["all_acc"], e["n_eval_tasks"]), "reasoning": e["reasoning_acc"]}
            if "night" in day:
                n, p = day["night"], day["practice"]
                rec["teacher"] = frac(n["teacher_pass_rate"], p["n_practice"])
                rec["train_items"] = n["train_items"]
                rec["loss"] = (n.get("train") or {}).get("loss_mean")
            days.append(rec)
        out["arms"][arm] = days
    out["base_reasoning"] = (A["meta"].get("base_gate") or {}).get("reasoning")

    # teacher pass rate by task type (day 1)
    by = defaultdict(lambda: [0, 0])
    fail = None
    for r in T:
        if r["phase"] == "teacher" and r["day"] == 1:
            tpl = r["task_id"].split("#")[0]
            by[tpl][0] += r["grade"]["passed"]
            by[tpl][1] += 1
            if not r["grade"]["passed"] and fail is None:
                fail = {"task": tpl, "code": code(r["output"])[-600:], "error": r["grade"]["feedback"][:200]}
    out["teacher_by_type"] = [{"type": k, "pass": v[0], "n": v[1]} for k, v in sorted(by.items())]
    out["teacher_fail_example"] = fail

    # same eval task: base (before any training) vs sdft (after night 1)
    ev = {(r["arm"], r["task_id"]): r for r in T if r["phase"] == "eval" and r["day"] == 1}
    for (arm, tid), r in sorted(ev.items(), key=lambda kv: ("sum_sq_even" not in kv[0][1], kv[0])):
        if arm == "sdft" and ("base", tid) in ev:
            out["before_after"] = {
                "task_id": tid,
                "prompt": next(m["content"] for m in r["messages"] if m["role"] == "user")[-400:],
                "before": code(ev[("base", tid)]["output"])[-500:],
                "before_error": ev[("base", tid)]["grade"]["feedback"][:160],
                "after": code(r["output"])[-500:],
                "after_error": r["grade"]["feedback"][:160],
            }
            break
    return out


def main():
    runs = [build_run(s) for s in RUNS if (RES / s["id"] / "run_artifacts.json").exists()]
    j = {"generated_from": "results/ (run_artifacts.json + traces.jsonl)", "runs": runs,
         "next": "Strengthen the nightly update (more on-policy passes, higher learning rate; same for sft and sdft), "
                 "check that one night moves the student off 0%, then run all 10 days."}
    (ROOT / "demo" / "journal.json").write_text(json.dumps(j, indent=1))
    print(f"wrote demo/journal.json ({len(runs)} runs)")


if __name__ == "__main__":
    main()
