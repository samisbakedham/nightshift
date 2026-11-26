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


TUNING = {
    "id": "night-tuning", "kind": "compare",
    "title": "A stronger night: 12/12 closed-book, including a task type it never trained on",
    "hardware": "Modal, NVIDIA A100 80GB",
    "what": "Same day 1, sdft only, three versions of the nightly update. Every pass re-samples the student's own "
            "attempts after the previous update; 2 attempts per optimizer step; LoRA rank 16.",
    "lesson": "One pass at lr 1e-4 barely nudged the weights. Five passes at 5e-4 moved the student from 0/12 to 12/12 "
              "with no docs in its prompt, including 4/4 on the held-out task type, and reasoning stayed at 93%. "
              "At 1e-3 it over-learned: it used knot() even where the answer had to be a list. 5e-4 is now the default "
              "for both sft and sdft.",
    "configs": [("lr 1e-4 · 1 pass", "20261125-215708-preflight"), ("lr 5e-4 · 5 passes", "sweep-lr5e-4"),
                ("lr 1e-3 · 5 passes", "sweep-lr1e-3")],
}


def build_compare(spec):
    import sys
    sys.path.insert(0, str(ROOT))
    from nightshift.env.releases import RELEASES
    heldout = {t.id for r in RELEASES for t in r.templates if t.heldout}
    rows = []
    for label, rid in spec["configs"]:
        A = json.loads((RES / rid / "run_artifacts.json").read_text())
        cfg, day = A["meta"]["config"], A["arms"]["sdft"]["days"][0]
        ev = [r for r in traces(rid) if r["phase"] == "eval" and r["arm"] == "sdft" and r["day"] == 1]
        un = [r for r in ev if r["task_id"].split("#")[0] in heldout]
        n_items = day["night"]["train_items"]
        sample = un[0] if un else None
        rows.append({
            "label": label, "run": rid, "lr": cfg["lr"], "passes": cfg["epochs"],
            "opt_steps": cfg["epochs"] * -(-n_items // cfg["grad_accum"]),
            "acc": {"k": sum(r["grade"]["passed"] for r in ev), "n": len(ev)},
            "unseen": {"k": sum(r["grade"]["passed"] for r in un), "n": len(un)},
            "reasoning": day["eval"]["reasoning_acc"],
            "unseen_code": code(sample["output"]) if sample else None,
            "unseen_result": sample["grade"]["feedback"][:120] if sample else None,
            "unseen_passed": bool(sample and sample["grade"]["passed"]),
        })
    unseen_prompt = None
    for r in traces(spec["configs"][1][1]):
        if r["phase"] == "eval" and r["task_id"].split("#")[0] in heldout:
            unseen_prompt = next(m["content"] for m in r["messages"] if m["role"] == "user")
            break
    return dict({k: v for k, v in spec.items() if k != "configs"}, rows=rows, unseen_prompt=unseen_prompt,
                model="Qwen/Qwen2.5-Coder-7B-Instruct")


def main():
    runs = [build_run(s) for s in RUNS if (RES / s["id"] / "run_artifacts.json").exists()]
    if all((RES / rid / "run_artifacts.json").exists() for _, rid in TUNING["configs"]):
        runs.append(build_compare(TUNING))
    j = {"generated_from": "results/ (run_artifacts.json + traces.jsonl)", "runs": runs,
         "next": "Run all 10 days with the tuned nightly update (all four arms) to see whether the student keeps "
                 "learning each new module without forgetting the old ones."}
    (ROOT / "demo" / "journal.json").write_text(json.dumps(j, indent=1))
    print(f"wrote demo/journal.json ({len(runs)} runs)")


if __name__ == "__main__":
    main()
