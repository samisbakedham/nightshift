"""Start runs on a *deployed* Modal app, so no local client is attached to them at all.

    modal deploy scripts/modal_run.py                                   # once per code change
    python scripts/modal_spawn.py full-v2-sft  --mode full --extra "--arms sft"
    python scripts/modal_spawn.py full-v2-sdft --mode full --extra "--arms sdft"

`modal run --detach` ties the call to an ephemeral app; a spawned call on a deployed app runs until it
finishes or hits the function timeout (12 h), whatever happens to your laptop or terminal. Results land
on the volume at runs/<name>/ as usual; fetch with `modal volume get nightshift-vol runs/<name> runs/`.
Stop the deployed app afterwards with `modal app stop nightshift` (it costs nothing while idle).
"""

from __future__ import annotations

import argparse

import modal

p = argparse.ArgumentParser()
p.add_argument("name")
p.add_argument("--mode", default="full", choices=["preflight", "quick", "full"])
p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
p.add_argument("--gpu", default="A100-80GB")
p.add_argument("--extra", default="")
a = p.parse_args()

fn = modal.Function.from_name("nightshift", "run_remote")
if a.gpu != "A100-80GB":
    fn = fn.with_options(gpu=a.gpu)
call = fn.spawn(a.mode, a.model, a.name, a.extra.split())
print(f"spawned {a.name}: call {call.object_id}  ({a.mode}, {a.gpu}, extra={a.extra!r})")
