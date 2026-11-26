"""Go / no-go summary for a real-model preflight.

    python scripts/preflight_summary.py <preflight_dir> <stub_estimate_dir> [--days N]

<preflight_dir>      a 1-day hf run of base, rag and sdft at the full run's budgets
<stub_estimate_dir>  a stub run with the same config as the planned full run; the stub is
                     only used to COUNT work units (generations + training examples)

Time estimate = (preflight seconds per work unit, per arm) x (stub work units for the full run).
It is rough: real models fail more than the stub (more reflections), rag prompts grow
every day, and day-1 prompts are the shortest of the run. Treat it as +-50%.
Exit code 0 = gate passed, 1 = gate failed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_MAX, TEACHER_MIN = 0.25, 0.70


def load(d):
    return json.loads((Path(d) / "run_artifacts.json").read_text())


def frac(passes, n):
    return f"{passes / n:.0%} ({passes}/{n})" if n else "n/a"


def main():
    pre, stub = load(sys.argv[1]), load(sys.argv[2])
    arms = pre["arms"]

    def day1_eval(arm):
        e = arms[arm]["days"][0]["eval"]
        n = e["n_eval_tasks"]
        return round(e["all_acc"] * n), n, e

    bp, bn, _ = day1_eval("base")
    rp, rn, _ = day1_eval("rag")
    night = arms["sdft"]["days"][0]["night"]
    n_prac = arms["sdft"]["days"][0]["practice"]["n_practice"]
    tp = round(night["teacher_pass_rate"] * n_prac)

    # time estimate
    sec, units = pre["meta"]["arm_seconds"], pre["meta"]["work_units"]
    per_unit = {a: sec[a] / max(1, units.get(a, 0)) for a in sec}
    per_unit["sft"] = per_unit.get("sdft", max(per_unit.values()))  # sft not in preflight; assume sdft speed
    planned = stub["meta"]["work_units"]
    est = sum(planned[a] * per_unit.get(a, per_unit["sdft"]) for a in planned)
    days = stub["meta"]["config"]["days"]

    ok_base, ok_teacher = bp / max(1, bn) < BASE_MAX, tp / max(1, n_prac) >= TEACHER_MIN
    g = pre["meta"].get("gen_stats") or {}
    print()
    print("=" * 72)
    print(f"PREFLIGHT  model={pre['meta']['model']}  ({pre['meta'].get('engine_info')})")
    print(f"  base_acc           {frac(bp, bn):>14}   want < {BASE_MAX:.0%}  -> {'OK' if ok_base else 'FAIL: model can already guess the API'}")
    print(f"  rag_acc            {frac(rp, rn):>14}   (ceiling; should be well above base)")
    print(f"  teacher_pass_rate  {frac(tp, n_prac):>14}   want >= {TEACHER_MIN:.0%} -> {'OK' if ok_teacher else 'FAIL: teacher too weak, nothing to distill'}")
    print(f"  est. time, {days} days, arms {','.join(planned)}: ~{est / 3600:.1f} h (rough, +-50%)")
    if g.get("sequences"):
        print(f"  generation: {g['tokens'] / g['sequences']:.0f} tokens/seq on average, "
              f"{g['hit_cap'] / g['sequences']:.0%} hit max_new_tokens without EOS")
    print("=" * 72)
    if not ok_teacher:
        print("  Check which task types the teacher fails (phase=teacher in traces.jsonl), or try a bigger model:")
        print("    modal run scripts/modal_run.py --mode preflight --model Qwen/Qwen2.5-Coder-14B-Instruct")
        print("    MODEL=Qwen/Qwen2.5-Coder-14B-Instruct bash scripts/run_cloud.sh preflight")
    print("GO" if ok_base and ok_teacher else "NO-GO")
    sys.exit(0 if ok_base and ok_teacher else 1)


if __name__ == "__main__":
    main()
