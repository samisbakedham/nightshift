#!/usr/bin/env bash
# One-command real experiment on a single 80 GB GPU (A100/H100).
#
#   gh repo clone samisbakedham/nightshift && cd nightshift && bash scripts/run_cloud.sh
#   (the repo is private: run `gh auth login` on the GPU box first, or make the repo public)
#
# Extra flags are passed through, e.g.:
#   bash scripts/run_cloud.sh --days 3 --arms base,sdft        # quick partial run
#   bash scripts/run_cloud.sh --model Qwen/Qwen2.5-14B-Instruct
#
# Model-written code is executed by the grader (in a subprocess with a timeout).
# Run this on a disposable cloud VM or container, not on a machine you care about.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR="runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"

python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

nvidia-smi || echo "WARNING: no NVIDIA GPU visible; a 7B run will be extremely slow."

echo "== tests"
python -m pytest -q

echo "== dry run (stub, ~2 min) to prove the pipeline before spending GPU time"
python -m nightshift.run --engine stub --out-dir "$RUN_DIR/dryrun" --days 2 > "$RUN_DIR/dryrun.log"

echo "== real-model preflight (1 day, sdft only, tiny budgets): catches tokenizer/chat-template/GPU issues cheaply"
python -m nightshift.run --engine hf --model "${MODEL:-Qwen/Qwen2.5-7B-Instruct}" --out-dir "$RUN_DIR/preflight" \
  --days 1 --arms sdft --practice-per-day 4 --authored-per-day 1 --eval-per-template 1 \
  --reasoning-eval-n 5 --gate-reasoning-n 3 --replay-per-day 2 --checkpoint-days 1 2>&1 | tee "$RUN_DIR/preflight.log"
echo "   preflight ok. Check the night line above: teacher_pass_rate near 0 means the model can't use the docs."

echo "== real run -> $RUN_DIR (log: $RUN_DIR/run.log)"
python -m nightshift.run --engine hf --model "${MODEL:-Qwen/Qwen2.5-7B-Instruct}" --out-dir "$RUN_DIR" "$@" 2>&1 | tee "$RUN_DIR/run.log"

cp "$RUN_DIR/run_artifacts.json" demo/run_artifacts.json
echo
echo "Done. Real artifacts copied to demo/run_artifacts.json"
echo "Adapters: $RUN_DIR/checkpoints/{sft,sdft}/day{1,5,10}"
echo "Next: git add demo/run_artifacts.json && git commit -m 'real run' && git push   (then deploy demo/)"
