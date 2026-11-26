"""Merge per-arm runs (same config, same seed) into one run folder.

    python scripts/merge_runs.py runs/full-coder7b \
        --arm base=runs/full-coder7b-v1 \
        --arm rag=runs/full-v2-rag \
        --arm sft=runs/full-v2-sft --arm sdft=runs/full-v2-sdft

Each `--arm NAME=SRC[:A-B][,SRC:C-D...]` takes that arm's days from one or more run folders (day ranges
are inclusive). Every arm restarts from the same frozen model and seed, so arms run in separate jobs are
directly comparable; the merged meta records where every day came from (`meta.merged_from`), and each
source's own base_gate / timing / gen stats are kept under `meta.sources`.
Writes run_artifacts.json and traces.jsonl (only the traces of the days taken from each source).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

CHECK = ("model", "days", "practice_per_day", "eval_per_template", "epochs", "grad_accum", "lr", "lora_r",
         "replay_per_day", "max_new_tokens", "seed", "gate")


def parse(spec: str):
    arm, srcs = spec.split("=", 1)
    out = []
    for part in srcs.split(","):
        path, _, rng = part.partition(":")
        lo, hi = (int(x) for x in rng.split("-")) if rng else (1, 10**9)
        out.append((Path(path), lo, hi))
    return arm, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--allow-diff", default="", help="comma-separated config keys allowed to differ between sources "
                    "(e.g. lr,epochs for arms that never train); recorded per arm in meta.arm_config")
    ap.add_argument("--meta-from", default=None, help="source folder whose meta/config heads the merged file "
                    "(default: the first source)")
    a = ap.parse_args()
    allow = {k for k in a.allow_diff.split(",") if k}
    arm_config = {}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cache: dict[Path, dict] = {}
    load = lambda p: cache.setdefault(p, json.loads((p / "run_artifacts.json").read_text()))
    merged, origin, traces, showcase, used_days = {}, {}, [], {}, {}
    ref = None
    for spec in a.arm:
        arm, parts = parse(spec)
        days = []
        for src, lo, hi in parts:
            A = load(src)
            cfg = A["meta"]["config"]
            ref = ref or cfg
            diff = {k: (ref.get(k), cfg.get(k)) for k in CHECK if ref.get(k) != cfg.get(k) and k not in allow}
            if diff:
                raise SystemExit(f"{src}: config differs from the first source: {diff}")
            got = A["arms"][arm]["days"][lo - 1:hi]
            arm_config[arm] = {k: cfg.get(k) for k in ("lr", "epochs", "grad_accum", "lr_schedule", "warmup_frac",
                                                        "lora_r", "gate")}
            if len(days) + 1 != lo and not (lo == 1 and not days):
                raise SystemExit(f"{arm}: day ranges must be contiguous (next is {len(days) + 1}, got {lo})")
            days += got
            rng = range(lo, lo + len(got))
            origin.setdefault(arm, []).append({"source": src.name, "days": [rng.start, rng.stop - 1]})
            used_days.setdefault(src, {}).setdefault(arm, set()).update(rng)
            lessons = A["arms"][arm].get("lessons", [])
            showcase.update({t["id"]: t for t in A.get("showcase_tasks", [])})
            traces += [t for t in A.get("traces", []) if t.get("arm") == arm and t.get("day") in rng]
        merged[arm] = {"days": days, "lessons": lessons}
        print(f"{arm:5s} {len(days):2d} days  <- " + " + ".join(f"{o['source']}[{o['days'][0]}-{o['days'][1]}]" for o in origin[arm]))

    first = load(Path(a.meta_from) if a.meta_from else Path(a.arm[0].split("=", 1)[1].split(",")[0].split(":")[0]))
    meta = dict(first["meta"])
    meta["config"] = dict(meta["config"], arms=list(merged), out_dir=str(out))
    meta["created"] = time.strftime("%Y-%m-%d %H:%M:%S") + " (merged)"
    meta["merged_from"] = origin
    meta["arm_config"] = arm_config
    meta["sources"] = {p.name: {k: A["meta"].get(k) for k in ("created", "base_gate", "arm_seconds", "work_units",
                                                                 "gen_stats", "engine_info")} for p, A in cache.items()}
    for k in ("arm_seconds", "work_units"):
        meta[k] = {arm: A["meta"].get(k, {}).get(arm) for p, A in cache.items() for arm in used_days.get(p, {})
                   if arm in A["meta"].get(k, {})}
    art = {"meta": meta, "arms": merged, "showcase_tasks": [showcase[k] for k in sorted(showcase)], "traces": traces}
    (out / "run_artifacts.json").write_text(json.dumps(art, indent=1))

    n = 0
    with open(out / "traces.jsonl", "w") as f:
        for src, arms in used_days.items():
            if not (src / "traces.jsonl").exists():
                continue
            for line in open(src / "traces.jsonl"):
                r = json.loads(line)
                if r.get("day") in arms.get(r.get("arm"), ()):
                    f.write(line)
                    n += 1
    print(f"wrote {out}/run_artifacts.json and traces.jsonl ({n} traces)")


if __name__ == "__main__":
    main()
