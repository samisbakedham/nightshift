"""The 10-day experiment: day work -> lessons -> self-curriculum -> nightly update -> eval.

Arms (all start from the same base weights, the same day-by-day releases and the
same held-out eval; the weakness-weighted half of practice adapts per arm):
  base   no memory, no training. The floor.
  rag    docs + lessons in the prompt every time. No training. The "memory only" ceiling.
  sft    closed-book student, nightly supervised fine-tuning on the teacher's
         verified solutions (off-policy imitation).
  sdft   closed-book student, nightly on-policy self-distillation from the frozen
         teacher that sees docs + promoted lessons, plus a reasoning replay anchor.

The teacher for BOTH sft and sdft is the same thing: the frozen initial weights
with docs + promoted lessons in context. The only difference between the two arms
is how they learn from it (imitating its samples vs matching its distribution on
their own samples). That isolates the method.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from . import prompts
from .curriculum import Curriculum, accept_authored
from .env import reasoning
from .env.releases import RELEASES, released_through
from .env.sandbox import extract_code, run_tests
from .env.tasks import (PracticeSource, Task, eval_set, make_task, release_for_day, released_templates)
from .lessons import Lesson, LessonStore, error_signature, parse_lesson
from .policy import Policy, Req, TrainItem

GATE_SEED_BASE = 500


@dataclass
class Config:
    engine: str = "stub"
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    arms: tuple = ("base", "rag", "sft", "sdft")
    days: int = 10
    practice_per_day: int = 24
    authored_per_day: int = 4
    eval_per_template: int = 2
    reasoning_eval_n: int = 30
    replay_per_day: int = 8
    gate_reasoning_n: int = 15
    gate_tolerance: float = 0.10
    # "log": score every update on the gate but never roll back (fair benchmark: sft forgetting stays visible)
    # "enforce": roll back regressed updates (deployment mode)   "off": skip gate scoring (fastest)
    gate: str = "log"
    # nightly update: 5 passes, each on FRESH on-policy rollouts; 2 items per Adam step.
    # Tuned on a 1-day Coder-7B test: lr 1e-4 x 1 pass -> 0/12, 5e-4 x 5 -> 12/12, 1e-3 x 5 -> 8/12 (overfit).
    epochs: int = 5
    grad_accum: int = 2
    lr: float = 5e-4
    lora_r: int = 16
    rollout_temperature: float = 1.0
    max_new_tokens: int = 384
    gen_batch: int = 32
    reason_max_new_tokens: int = 256
    kl_direction: str = "forward"
    promote_threshold: float = 0.5
    checkpoint_days: tuple = (1, 5, 10)
    out_dir: str = "runs/latest"
    seed: int = 0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


@dataclass
class ArmState:
    name: str
    lessons: LessonStore
    curriculum: Curriculum
    source: PracticeSource = field(default_factory=PracticeSource)
    authored: dict = field(default_factory=dict)


class Experiment:
    def __init__(self, cfg: Config, policy: Policy):
        self.cfg = cfg
        self.policy = policy
        self.tasks: dict[str, Task] = {}
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.trace_f = open(self.out / "traces.jsonl", "w")
        # showcase = an UNSEEN-family task per module (prompt text never seen in practice)
        self.showcase_ids = {make_task(r, next(t for t in r.templates if t.heldout), 0, "eval").id for r in RELEASES}
        self.eval_prompts = {t.prompt for t in eval_set(10, cfg.eval_per_template)}
        self.showcase_traces: list[dict] = []
        self.results: dict = {}
        self.units: dict[str, int] = {}     # generated sequences + training examples, per arm (for time estimates)
        self.seconds: dict[str, float] = {}
        self._arm = None
        self.base_gate = None
        if policy is not None:
            self._wrap(policy)

    def _wrap(self, policy):
        """Count work units (one generated sequence or one training example) per arm."""
        gen, sdft, sft = policy.generate, policy.train_sdft, policy.train_sft

        def count(n):
            if self._arm:
                self.units[self._arm] = self.units.get(self._arm, 0) + n

        def g(reqs, *a, **k):
            count(len(reqs))
            return gen(reqs, *a, **k)

        def t1(items, cfg):
            count(2 * len(items) * cfg.epochs)  # rollout + forward/backward
            return sdft(items, cfg)

        def t2(items, cfg):
            count(len(items) * cfg.epochs)
            return sft(items, cfg)
        policy.generate, policy.train_sdft, policy.train_sft = g, t1, t2

    # ------------------------------------------------------------------ helpers
    def lookup(self, tid: str) -> Task:
        return self.tasks[tid]

    def _reg(self, tasks):
        for t in tasks:
            self.tasks[t.id] = t
        return tasks

    def _docs(self, modules) -> list[str]:
        return [r.docs for r in RELEASES if r.module in modules]

    def _memory_for(self, st: ArmState, day: int, module: str | None = None, promoted_only=False):
        mods = [r.module for r in released_through(day)] if module is None else [module]
        return self._docs(mods), [l.render() for l in st.lessons.for_modules(mods, promoted_only)]

    def _solve(self, tasks: list[Task], msgs: list[list[dict]], use_base=False, temperature=0.0):
        reqs = [Req(m, "code", {"task_id": t.id}) for t, m in zip(tasks, msgs)]
        outs = self.policy.generate(reqs, temperature=temperature, max_new_tokens=self.cfg.max_new_tokens,
                                    use_base=use_base)
        grades = [run_tests(extract_code(o), t.preamble, t.tests) for t, o in zip(tasks, outs)]
        return outs, grades

    def _trace(self, **rec):
        self.trace_f.write(json.dumps(rec) + "\n")
        if rec.get("phase") == "eval" and rec.get("task_id") in self.showcase_ids:
            self.showcase_traces.append(rec)

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, st: ArmState, day: int) -> dict:
        tasks = self._reg(eval_set(day, self.cfg.eval_per_template))
        closed = [prompts.task_messages(t) for t in tasks]
        if st.name == "rag":
            docs, les = self._memory_for(st, day)
            msgs = [prompts.task_messages(t, docs, les) for t in tasks]
        else:
            msgs = closed
        outs, grades = self._solve(tasks, msgs)
        by_mod: dict[str, list[bool]] = {}
        seen, unseen = [], []
        for t, m, o, g in zip(tasks, msgs, outs, grades):
            by_mod.setdefault(t.module, []).append(g.passed)
            (unseen if t.heldout else seen).append(g.passed)
            self._trace(phase="eval", arm=st.name, day=day, task_id=t.id, module=t.module, messages=m,
                        output=o, grade=g.to_dict())
        # general reasoning (never gets memory)
        items = reasoning.eval_items(self.cfg.reasoning_eval_n)
        routs = self.policy.generate([Req(prompts.reasoning_messages(i.prompt), "reason", {"answer": i.answer})
                                      for i in items], max_new_tokens=self.cfg.reason_max_new_tokens)
        rok = [reasoning.grade(o, i.answer) for o, i in zip(routs, items)]
        for i, o, ok in zip(items[:3], routs[:3], rok[:3]):
            self._trace(phase="eval_reason", arm=st.name, day=day, task_id=i.id, prompt=i.prompt, output=o, ok=ok)

        acc = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
        today = release_for_day(day).module
        old = [p for m, v in by_mod.items() if m != today for p in v]
        tok_prompt = sum(self.policy.count_tokens(m) for m in msgs) / len(msgs)
        tok_closed = sum(self.policy.count_tokens(m) for m in closed) / len(closed)
        return {
            "day": day,
            "new_module_acc": acc(by_mod[today]),
            "retention_acc": acc(old),
            "day1_module_acc": acc(by_mod["tessel.run"]),
            "all_acc": acc([p for v in by_mod.values() for p in v]),
            "unseen_family_acc": acc(unseen),
            "seen_family_acc": acc(seen),
            "reasoning_acc": acc(rok),
            "per_module": {m: acc(v) for m, v in by_mod.items()},
            "avg_prompt_tokens": round(tok_prompt, 1),
            "memory_tokens": round(tok_prompt - tok_closed, 1),
            "n_eval_tasks": len(tasks),
        }

    def gate_score(self, day: int) -> dict:
        """Regression check on a SEPARATE gate split (never the reported eval set)."""
        old = [make_task(r, t, GATE_SEED_BASE, "gate") for r, t in released_templates(day - 1)]
        self._reg(old)
        old_acc = None
        if old:
            _, g = self._solve(old, [prompts.task_messages(t) for t in old])
            old_acc = sum(x.passed for x in g) / len(g)
        items = [reasoning.item(GATE_SEED_BASE + i, "gate") for i in range(self.cfg.gate_reasoning_n)]
        outs = self.policy.generate([Req(prompts.reasoning_messages(i.prompt), "reason", {"answer": i.answer})
                                     for i in items], max_new_tokens=self.cfg.reason_max_new_tokens)
        r_acc = sum(reasoning.grade(o, i.answer) for o, i in zip(outs, items)) / len(items)
        return {"old_modules": old_acc, "reasoning": r_acc}

    # ------------------------------------------------------------------ the day
    def practice(self, st: ArmState, day: int) -> dict:
        cfg = self.cfg
        rel = release_for_day(day)
        tasks = self._reg(st.curriculum.plan(day, cfg.practice_per_day, st.source))

        # self-authored exercises targeting the weakest module
        authored_log = []
        for k in range(cfg.authored_per_day):
            mod = st.curriculum.weakest_module(day)
            mrel = next(r for r in RELEASES if r.module == mod)
            fallback = self._reg([st.source.draw(mrel, random.Random(day * 100 + k).choice(
                [t for t in mrel.templates if not t.heldout]))])[0]
            msg = prompts.author_messages(mod, mrel.docs, "Target the parts of the API you find hardest.")
            [text] = self.policy.generate([Req(msg, "author", {"fallback_task_id": fallback.id})],
                                          temperature=0.7, max_new_tokens=cfg.max_new_tokens)
            task, why = accept_authored(text, mrel, f"{mod}.authored#d{day}k{k}:{st.name}", day)
            if task and task.prompt.strip() in self.eval_prompts:
                task, why = None, "rejected: identical to an eval prompt"
            authored_log.append({"module": mod, "accepted": task is not None, "reason": why})
            self._trace(phase="author", arm=st.name, day=day, module=mod, messages=msg, output=text, reason=why)
            if task:
                self._reg([task])
                tasks.append(task)

        if st.name == "rag":
            msgs = [prompts.task_messages(t, *self._memory_for(st, day)) for t in tasks]
        else:
            msgs = [prompts.task_messages(t) for t in tasks]
        outs, grades = self._solve(tasks, msgs)
        for t, m, o, g in zip(tasks, msgs, outs, grades):
            if t.split != "authored":
                st.curriculum.record(t.template, g.passed)
            self._trace(phase="practice", arm=st.name, day=day, task_id=t.id, module=t.module, messages=m,
                        output=o, grade=g.to_dict())

        # lessons from failures
        failed = [(t, o, g) for t, o, g in zip(tasks, outs, grades) if not g.passed]
        new_lessons = 0
        for t, o, g in failed:
            sig = error_signature(t.module, g.feedback)
            st.lessons.note_failure(sig, t.id)
        if failed:
            mrel = {r.module: r for r in RELEASES}
            reflect = [Req(prompts.reflect_messages(t, extract_code(o), g.feedback, [mrel[t.module].docs]), "reflect",
                           {"module": t.module, "hint": mrel[t.module].docs.splitlines()[2].lstrip("- ")})
                       for t, o, g in failed]
            texts = self.policy.generate(reflect, max_new_tokens=96)
            cands = []
            for (t, o, g), text in zip(failed, texts):
                body = parse_lesson(text)
                if body:
                    cands.append((t, g, Lesson(id=f"L{day}-{t.id}", module=t.module, text=body,
                                               signature=error_signature(t.module, g.feedback), day=day,
                                               sources=[t.id])))
            # verify: closed-book retry with ONLY this lesson in context
            if cands:
                _, vg = self._solve([t for t, _, _ in cands],
                                    [prompts.task_messages(t, [], [l.render()]) for t, _, l in cands])
                for (t, g, l), v in zip(cands, vg):
                    l.verified = v.passed
                    l.surprise = 1 - st.curriculum.stats[t.template].rate() if t.template in st.curriculum.stats else 1.0
                    st.lessons.add(l)
                    new_lessons += 1
        surprise = {}
        for l in st.lessons.lessons:
            tpls = [tid for tid in l.sources if tid in self.tasks]
            rates = [st.curriculum.weakness(self.tasks[tid].template) for tid in tpls
                     if self.tasks[tid].template in st.curriculum.stats]
            surprise[l.id] = sum(rates) / len(rates) if rates else l.surprise
        st.lessons.refresh(lambda l: surprise.get(l.id, l.surprise))

        return {"tasks": tasks, "outs": outs, "grades": grades,
                "practice_pass_rate": round(sum(g.passed for g in grades) / len(grades), 4),
                "n_practice": len(tasks), "authored": authored_log, "new_lessons": new_lessons,
                "lessons_total": len(st.lessons.lessons),
                "lessons_verified": sum(l.verified for l in st.lessons.lessons),
                "lessons_promoted": sum(l.promoted for l in st.lessons.lessons)}

    # ------------------------------------------------------------------ the night
    def night(self, st: ArmState, day: int, tasks: list[Task]) -> dict:
        cfg = self.cfg
        # teacher = frozen base + docs + PROMOTED lessons for the task's module
        t_msgs = [prompts.task_messages(t, *self._memory_for(st, day, t.module, promoted_only=True)) for t in tasks]
        t_outs, t_grades = self._solve(tasks, t_msgs, use_base=True)
        for t, m, o, g in zip(tasks, t_msgs, t_outs, t_grades):
            self._trace(phase="teacher", arm=st.name, day=day, task_id=t.id, module=t.module, messages=m,
                        output=o, grade=g.to_dict())
        keep = [i for i, g in enumerate(t_grades) if g.passed]
        if not keep:  # nothing verified to learn: training on replay alone only adds noise
            return {"teacher_pass_rate": 0.0, "train_items": 0, "train": {"steps": 0, "loss_mean": None},
                    "gate_mode": cfg.gate, "gate_before": None, "gate_after": None, "would_reject": False,
                    "accepted": True, "skipped": True,
                    "rejected_because": ["skipped: teacher solved no practice task, nothing to distill"]}
        items = [TrainItem(student_messages=prompts.task_messages(tasks[i]), teacher_messages=t_msgs[i],
                           target=extract_code(t_outs[i]).strip() and f"```python\n{extract_code(t_outs[i]).strip()}\n```",
                           module=tasks[i].module) for i in keep]
        # replay: general reasoning. sdft -> teacher is plain base (anchor). sft -> base's own correct answers.
        rep = reasoning.replay_items(day * 1000, cfg.replay_per_day)
        rmsgs = [prompts.reasoning_messages(r.prompt) for r in rep]
        if st.name == "sdft":
            items += [TrainItem(m, m, kind="replay") for m in rmsgs]
        else:
            routs = self.policy.generate([Req(m, "reason", {"answer": r.answer}) for m, r in zip(rmsgs, rep)],
                                         max_new_tokens=cfg.reason_max_new_tokens, use_base=True)
            items += [TrainItem(m, m, target=o, kind="replay") for m, r, o in zip(rmsgs, rep, routs)
                      if reasoning.grade(o, r.answer)]

        gate_on = cfg.gate != "off"
        before = self.gate_score(day) if gate_on else None
        snap = self.policy.snapshot()
        stats = (self.policy.train_sdft if st.name == "sdft" else self.policy.train_sft)(items, cfg)
        after = self.gate_score(day) if gate_on else None  # measured BEFORE any rollback
        reasons = []
        if gate_on:
            if after["reasoning"] < self.base_gate["reasoning"] - cfg.gate_tolerance:
                reasons.append("general reasoning regressed vs base")
            if before["old_modules"] is not None and after["old_modules"] < before["old_modules"] - cfg.gate_tolerance:
                reasons.append("earlier modules regressed vs yesterday")
        rolled_back = cfg.gate == "enforce" and bool(reasons)
        if rolled_back:
            self.policy.restore(snap)
        return {"teacher_pass_rate": round(len(keep) / max(1, len(tasks)), 4), "train_items": len(items),
                "train": stats, "gate_mode": cfg.gate, "gate_before": before, "gate_after": after,
                "would_reject": bool(reasons), "accepted": not rolled_back, "rejected_because": reasons}

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict:
        cfg = self.cfg
        init = self.policy.snapshot()
        if cfg.gate != "off":
            self.base_gate = self.gate_score(1)
            log(f"base gate: {self.base_gate}  (gate mode: {cfg.gate})")
        for arm in cfg.arms:
            self.policy.restore(init)
            self._arm = arm
            t_arm = time.time()
            st = ArmState(arm, LessonStore(cfg.promote_threshold), Curriculum(random.Random(cfg.seed)))
            days = []
            for day in range(1, cfg.days + 1):
                rec: dict = {"seconds": {}}
                t0 = time.time()
                if arm != "base":
                    p = self.practice(st, day)
                    rec["seconds"]["practice"] = round(time.time() - t0, 1); t0 = time.time()
                    rec["practice"] = {k: v for k, v in p.items() if k not in ("tasks", "outs", "grades")}
                    if arm in ("sft", "sdft"):
                        rec["night"] = self.night(st, day, p["tasks"])
                        if day in cfg.checkpoint_days:
                            self.policy.save(str(self.out / "checkpoints" / arm / f"day{day}"))
                        rec["seconds"]["night"] = round(time.time() - t0, 1); t0 = time.time()
                rec["eval"] = self.evaluate(st, day)
                rec["seconds"]["eval"] = round(time.time() - t0, 1)
                self.seconds[arm] = round(time.time() - t_arm, 1)
                days.append(rec)
                self.results[arm] = {"days": days, "lessons": [l.to_dict() for l in st.lessons.lessons]}
                self.flush()  # crash-safe: artifacts are always current to the last finished day
                e = rec["eval"]
                log(f"[{arm}] day {day}: new={e['new_module_acc']} retention={e['retention_acc']} "
                    f"all={e['all_acc']} unseen={e['unseen_family_acc']} reasoning={e['reasoning_acc']} mem_tokens={e['memory_tokens']}"
                    + (f" gate={_gate_word(rec['night'])}" if "night" in rec else ""))
        self.flush()
        self.trace_f.close()
        return self.results

    def flush(self):
        self.trace_f.flush()
        art = build_artifacts(self.cfg, self.policy, self.results, self.showcase_traces, self.tasks,
                              self.showcase_ids, self.base_gate)
        art["meta"]["work_units"] = dict(self.units)
        art["meta"]["arm_seconds"] = dict(self.seconds)
        art["meta"]["engine_info"] = getattr(self.policy, "engine_info", None)
        art["meta"]["gen_stats"] = dict(getattr(self.policy, "gen_stats", {}) or {})
        (self.out / "run_artifacts.json").write_text(json.dumps(art, indent=1))


def _gate_word(n: dict) -> str:
    if n.get("skipped"):
        return "skipped"
    if not n["accepted"]:
        return "ROLLBACK"
    return "WOULD-ROLLBACK(logged)" if n.get("would_reject") else "ok"


def build_artifacts(cfg, policy, results, showcase_traces, tasks, showcase_ids, base_gate) -> dict:
    placeholder = cfg.engine != "hf"
    return {
        "meta": {
            "placeholder": placeholder,
            "placeholder_reason": (None if not placeholder else
                                   f"engine={cfg.engine}: scripted/tiny model, NOT a real result. "
                                   "Run scripts/run_cloud.sh for real numbers."),
            "engine": cfg.engine, "model": policy.name, "gate_mode": cfg.gate, "approx_tokens": policy.approx_tokens,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": asdict(cfg), "base_gate": base_gate,
            "releases": [{"day": r.day, "module": r.module, "docs": r.docs} for r in RELEASES],
        },
        "arms": results,
        "showcase_tasks": [tasks[t].public() for t in sorted(showcase_ids) if t in tasks],
        "traces": showcase_traces,
    }
