"""Run nightshift on Modal (https://modal.com) instead of renting a GPU box.

    pip install modal && modal setup                                    # once
    modal run scripts/modal_run.py --mode preflight                     # day 1 of base/rag/sdft -> GO / NO-GO
    modal run --detach scripts/modal_run.py --mode quick                # 3 days, all 4 arms
    modal run --detach scripts/modal_run.py --mode full                 # 10 days, all 4 arms

Options: --model Qwen/Qwen2.5-Coder-14B-Instruct   --gpu {A100-80GB,L40S,H100}
         --extra "--arms base,sdft --gate off"     (passed through to the experiment)
         --name my-run                             (default: <timestamp>-<mode>)

`--detach` keeps the job running if your laptop sleeps or you hit Ctrl-C. If the local side is still
connected when the job ends, run_artifacts.json + run.log are downloaded to runs/<name>/; otherwise
the exact `modal volume get` command is printed (and works any time later).

Everything lives on the Modal Volume `nightshift-vol`, mounted at /vol:
  /vol/hf_cache          Hugging Face cache, so the model downloads once
  /vol/runs/<name>/      artifacts, run.log, traces.jsonl, LoRA checkpoints
The volume is committed every time a simulated day finishes (and every 10 minutes, and on exit),
so a run that stops early (timeout, credits, crash) keeps every finished day.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/nightshift"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_GPU = "A100-80GB"
GPUS = ("A100-80GB", "L40S", "H100")
VOL_RUNS = Path("/vol/runs")
TIMEOUT = 12 * 60 * 60  # 43200 s. Modal's default (300 s) would kill the run mid-experiment.

app = modal.App("nightshift")
vol = modal.Volume.from_name("nightshift-vol", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .env({"HF_HOME": "/vol/hf_cache", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(ROOT), REMOTE_ROOT,
                   ignore=[".venv", ".venv/**", "runs", "runs/**", ".git", ".git/**", "results", "results/**",
                           "**/__pycache__", "**/__pycache__/**", ".pytest_cache", ".pytest_cache/**"])
)


# ---------------------------------------------------------------------------------------- remote side
def _committer(watch_dir: Path, stop: threading.Event):
    """Commit the volume whenever a run_artifacts.json changes (= a day finished), and every 10 min."""
    seen: dict[str, float] = {}
    last = time.time()
    while not stop.wait(20):
        changed = False
        for f in watch_dir.rglob("run_artifacts.json"):
            m = f.stat().st_mtime
            if seen.get(str(f)) != m:
                seen[str(f)], changed = m, True
        if changed or time.time() - last > 600:
            try:
                vol.commit()
                last = time.time()
                if changed:
                    print(f"[modal] volume committed ({time.strftime('%H:%M:%S')})", flush=True)
            except Exception as e:  # noqa: BLE001  never let a commit hiccup kill the experiment
                print(f"[modal] volume commit failed: {e}", flush=True)


def _sh(cmd: list[str], log: Path | None = None) -> int:
    """Run a command, streaming output to stdout (Modal logs) and optionally a log file."""
    print("$", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=REMOTE_ROOT, env={**os.environ, "PYTHONPATH": REMOTE_ROOT}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    f = open(log, "a") if log else None
    for line in p.stdout:
        sys.stdout.write(line)
        if f:
            f.write(line)
            f.flush()
    p.wait()
    if f:
        f.close()
    return p.returncode


@app.function(image=image, gpu=DEFAULT_GPU, volumes={"/vol": vol}, timeout=TIMEOUT)
def run_remote(mode: str, model: str, name: str, extra: list[str]) -> dict:
    out = VOL_RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    t = threading.Thread(target=_committer, args=(out, stop), daemon=True)
    t.start()
    py = sys.executable
    hf = [py, "-m", "nightshift.run", "--engine", "hf", "--model", model]
    result = {"name": name, "mode": mode, "files": [], "summary": None, "exit_code": 0}
    try:
        try:
            _sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
        except FileNotFoundError:
            print("[modal] WARNING: no nvidia-smi; is a GPU attached?", flush=True)
        if mode == "preflight":
            if _sh([py, "-m", "pytest", "-q"]) != 0:
                raise RuntimeError("tests failed")
            _sh([py, "-m", "nightshift.run", "--engine", "stub", "--out-dir", str(out / "stub-estimate"), *extra],
                out / "stub-estimate.log")
            pre = out / "preflight"
            pre.mkdir(exist_ok=True)
            rc = _sh(hf + ["--out-dir", str(pre), "--days", "1", "--arms", "base,rag,sdft", "--eval-per-template", "4",
                           "--checkpoint-days", "0", *extra], pre / "run.log")
            if rc != 0:
                raise RuntimeError(f"preflight run failed (exit {rc}); see preflight/run.log")
            summary = out / "preflight-summary.txt"
            result["exit_code"] = _sh([py, "scripts/preflight_summary.py", str(pre), str(out / "stub-estimate")], summary)
            result["summary"] = summary.read_text()
            result["files"] = ["preflight/run_artifacts.json", "preflight/run.log", "preflight-summary.txt"]
        else:
            days = ["--days", "3", "--checkpoint-days", "3"] if mode == "quick" else []
            result["exit_code"] = _sh(hf + ["--out-dir", str(out), *days, *extra], out / "run.log")
            result["files"] = ["run_artifacts.json", "run.log"]
    finally:
        stop.set()
        vol.commit()
        print(f"[modal] final volume commit done: /vol/runs/{name}", flush=True)
    return result


# ---------------------------------------------------------------------------------------- local side
def _download(name: str, rel: str) -> bool:
    dest = ROOT / "runs" / name / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(dest, "wb") as f:
            for chunk in vol.read_file(f"runs/{name}/{rel}"):
                f.write(chunk)
        return True
    except Exception as e:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        print(f"  could not download {rel}: {e}")
        return False


@app.local_entrypoint()
def main(mode: str = "preflight", model: str = DEFAULT_MODEL, gpu: str = DEFAULT_GPU, extra: str = "", name: str = ""):
    if mode not in ("preflight", "quick", "full"):
        raise SystemExit("--mode must be preflight, quick or full")
    if gpu not in GPUS:
        print(f"warning: --gpu {gpu} is not one of {GPUS}; passing it to Modal anyway")
    name = name or f"{time.strftime('%Y%m%d-%H%M%S')}-{mode}"
    get_cmd = f"modal volume get nightshift-vol runs/{name} runs/"
    print(f"nightshift on Modal: mode={mode} model={model} gpu={gpu} run={name}")
    print(f"results go to the volume at runs/{name}; fetch them any time with:\n  {get_cmd}\n")

    fn = run_remote if gpu == DEFAULT_GPU else run_remote.with_options(gpu=gpu)
    res = fn.remote(mode, model, name, extra.split())

    print()
    if res.get("summary"):
        print(res["summary"])
    got = [rel for rel in res["files"] if _download(name, rel)]
    if got:
        print(f"downloaded to runs/{name}/: {', '.join(got)}")
    print(f"full run folder (incl. traces.jsonl, checkpoints):  {get_cmd}")
    if mode != "preflight" and "run_artifacts.json" in got:
        print(f"save to GitHub:  bash scripts/publish_run.sh runs/{name}")
    if res["exit_code"] != 0:
        print(f"(exit code {res['exit_code']}" + (": NO-GO)" if mode == "preflight" else ")"))
