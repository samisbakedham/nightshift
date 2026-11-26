"""Run model-written code against hidden tests in a separate Python process.

This is containment, not a security boundary: a subprocess with a timeout, an
isolated interpreter (-I), a throwaway working directory, and CPU/memory rlimits
where the OS supports them. Run real-model experiments inside a disposable
container or cloud VM.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

TESSEL_PATH = str(Path(__file__).parent / "tessel_lib")

_HARNESS = r'''
import sys, json, traceback
sys.path.insert(0, {tessel_path!r})
results = []
try:
    ns = {{}}
    exec(compile(open("solution.py").read(), "solution.py", "exec"), ns)
    solve = ns["solve"]
except BaseException as e:
    print(json.dumps({{"load_error": f"{{type(e).__name__}}: {{e}}"}}))
    sys.exit(0)
{preamble}
for args_expr, expected_expr, post_expr in json.loads({tests_json!r}):
    try:
        args = eval(args_expr)
        out = solve(*args)
        if post_expr:
            out = eval(post_expr)(out)
        expected = eval(expected_expr)
        ok = out == expected
        results.append({{"ok": bool(ok), "got": repr(out)[:200], "want": expected_expr[:200]}})
    except BaseException as e:
        tb = traceback.extract_tb(e.__traceback__)
        results.append({{"ok": False, "error": f"{{type(e).__name__}}: {{e}}"[:300]}})
print(json.dumps({{"results": results}}))
'''


@dataclass
class GradeResult:
    passed: bool
    n_pass: int
    n_total: int
    feedback: str          # short, model-readable failure description
    raw: dict

    def to_dict(self):
        return dict(passed=self.passed, n_pass=self.n_pass, n_total=self.n_total, feedback=self.feedback)


CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    """Take the last fenced python block, or the raw text if there is none."""
    blocks = CODE_RE.findall(text)
    return blocks[-1] if blocks else text


def _limits():  # pragma: no cover - runs in child
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        if sys.platform != "darwin":
            resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    except Exception:
        pass


def run_tests(code: str, preamble: str, tests: list[list[str]], timeout: float = 10.0) -> GradeResult:
    """tests: list of [args_expr, expected_expr, post_expr_or_empty]."""
    harness = _HARNESS.format(tessel_path=TESSEL_PATH, preamble=preamble, tests_json=json.dumps(tests))
    with tempfile.TemporaryDirectory(prefix="nightshift_") as d:
        Path(d, "solution.py").write_text(code)
        Path(d, "harness.py").write_text(harness)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "harness.py"], cwd=d, capture_output=True, text=True,
                timeout=timeout, preexec_fn=_limits if os.name == "posix" else None,
                env={"PATH": os.environ.get("PATH", "")},
            )
        except subprocess.TimeoutExpired:
            return GradeResult(False, 0, len(tests), "Timed out.", {"timeout": True})
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return GradeResult(False, 0, len(tests), f"Crashed: {proc.stderr[-300:]}", {"stderr": proc.stderr[-2000:]})
    if "load_error" in data:
        return GradeResult(False, 0, len(tests), f"Code failed to load: {data['load_error']}", data)
    res = data["results"]
    n_pass = sum(r["ok"] for r in res)
    fb = "All tests passed."
    for r in res:
        if not r["ok"]:
            fb = r.get("error") or f"Wrong answer: got {r['got']}, expected {r['want']}"
            break
    return GradeResult(n_pass == len(res), n_pass, len(res), fb, data)


_COLLECT = r'''
import sys, json
sys.path.insert(0, {tessel_path!r})
{preamble}
ns = {{}}
exec(compile(open("solution.py").read(), "solution.py", "exec"), ns)
outs = []
for a in json.loads({inputs_json!r}):
    out = ns["solve"](*eval(a))
    r = repr(out)
    assert eval(r) == out, "output repr does not round-trip"
    outs.append(r)
print(json.dumps(outs))
'''


def collect_outputs(code: str, preamble: str, inputs: list[str], timeout: float = 10.0):
    """Run (untrusted) reference code on inputs in the sandbox; return (ok, [repr(output)])."""
    script = _COLLECT.format(tessel_path=TESSEL_PATH, preamble=preamble, inputs_json=json.dumps(inputs))
    with tempfile.TemporaryDirectory(prefix="nightshift_") as d:
        Path(d, "solution.py").write_text(code)
        Path(d, "collect.py").write_text(script)
        try:
            proc = subprocess.run([sys.executable, "-I", "collect.py"], cwd=d, capture_output=True, text=True,
                                  timeout=timeout, preexec_fn=_limits if os.name == "posix" else None,
                                  env={"PATH": os.environ.get("PATH", "")})
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if proc.returncode != 0:
        return False, (proc.stderr.strip().splitlines() or ["error"])[-1][:200]
    try:
        return True, json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return False, "unparseable output"
