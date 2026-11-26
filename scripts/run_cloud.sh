#!/usr/bin/env bash
# Real experiment on a single 80 GB GPU (A100/H100). Three modes, run them in order:
#
#   bash scripts/run_cloud.sh preflight   # ~15-30 min: tests + day 1 of base/rag/sdft -> GO / NO-GO + time estimate
#   bash scripts/run_cloud.sh quick       # 3 days, all 4 arms
#   bash scripts/run_cloud.sh full        # 10 days, all 4 arms
#
# Then save the results before you kill the pod:
#   bash scripts/publish_run.sh runs/<dir>
#
# Model: MODEL=Qwen/Qwen2.5-Coder-14B-Instruct bash scripts/run_cloud.sh preflight
# Extra flags go to the experiment, e.g. `bash scripts/run_cloud.sh quick --arms base,sdft --gate off`.
# Use tmux (or nohup) so an SSH drop doesn't kill the run.
#
# Model-written code is executed by the grader (in a subprocess with a timeout).
# Run this on a disposable cloud VM or container, not on a machine you care about.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-}"
case "$MODE" in
  preflight|quick|full) shift ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
RUN_DIR="runs/$(date +%Y%m%d-%H%M%S)-$MODE"
mkdir -p "$RUN_DIR"

if [ ! -d .venv ]; then python3 -m venv .venv; fi
. .venv/bin/activate
python -m pip --version >/dev/null 2>&1 || python -m ensurepip >/dev/null   # uv-made venvs ship without pip
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "WARNING: no NVIDIA GPU visible; a 7B run will be extremely slow."

run() {  # run <out_dir> <args...>
  local out="$1"; shift
  mkdir -p "$out"
  python -m nightshift.run --engine hf --model "$MODEL" --out-dir "$out" "$@" 2>&1 | tee "$out/run.log"
}

case "$MODE" in
  preflight)
    echo "== tests"
    python -m pytest -q
    echo "== stub run: counts the work a full 10-day run does (for the time estimate)"
    python -m nightshift.run --engine stub --out-dir "$RUN_DIR/stub-estimate" "$@" > "$RUN_DIR/stub-estimate.log"
    echo "== real model, day 1, arms base,rag,sdft, full practice budget, 4 eval tasks per type"
    run "$RUN_DIR/preflight" --days 1 --arms base,rag,sdft --eval-per-template 4 --checkpoint-days 0 "$@"
    python scripts/preflight_summary.py "$RUN_DIR/preflight" "$RUN_DIR/stub-estimate" | tee "$RUN_DIR/preflight-summary.txt"
    ;;
  quick)
    run "$RUN_DIR" --days 3 --checkpoint-days 3 "$@"
    ;;
  full)
    run "$RUN_DIR" "$@"
    ;;
esac

echo
echo "Done: $RUN_DIR"
[ "$MODE" = preflight ] || echo "Save it before terminating the pod:  bash scripts/publish_run.sh $RUN_DIR"
