#!/usr/bin/env bash
# Save a finished (or partial) real run to GitHub before the cloud pod is terminated.
#
#   bash scripts/publish_run.sh runs/<dir>
#
# Copies <dir>/run_artifacts.json to demo/run_artifacts.json and results/<dir-name>/ (with the
# run log), commits only those files and pushes. Refuses placeholder (stub/tiny) artifacts.
# NOT saved: traces.jsonl (large) and LoRA adapters in <dir>/checkpoints. Copy those off the pod
# yourself if you want them (scp/rsync, or `huggingface-cli upload`).
set -euo pipefail
cd "$(dirname "$0")/.."

DIR="${1:?usage: bash scripts/publish_run.sh runs/<dir>}"
DIR="${DIR%/}"
ART="$DIR/run_artifacts.json"
[ -f "$ART" ] || { echo "no $ART"; exit 1; }

python3 - "$ART" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["meta"]
if m.get("placeholder"):
    sys.exit(f"refusing: {sys.argv[1]} is placeholder data ({m.get('placeholder_reason')})")
days = {a: len(v["days"]) for a, v in json.load(open(sys.argv[1]))["arms"].items()}
print(f"real run: model={m['model']} gate={m.get('gate_mode')} days per arm={days}")
PY

git config user.email >/dev/null || { echo "set a git identity first: git config --global user.name 'You'; git config --global user.email you@example.com"; exit 1; }
git ls-remote origin >/dev/null 2>&1 || { echo "can't reach origin; run 'gh auth login' (or check the remote) first"; exit 1; }

NAME="$(basename "$DIR")"
mkdir -p "results/$NAME"
cp "$ART" demo/run_artifacts.json
cp "$ART" "results/$NAME/run_artifacts.json"
for f in run.log preflight-summary.txt; do
  if [ -f "$DIR/$f" ]; then cp "$DIR/$f" "results/$NAME/"; fi
done

git add demo/run_artifacts.json "results/$NAME"
git commit -m "Real run results: $NAME"
git pull --rebase -q origin "$(git rev-parse --abbrev-ref HEAD)"
git push origin HEAD
echo "Pushed. traces.jsonl and $DIR/checkpoints are still only on this machine."
