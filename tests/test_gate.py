"""Gate modes: 'log' records regressions without rolling back; 'enforce' rolls back; 'off' skips scoring."""

import pytest

from nightshift.experiment import Config, Experiment
from nightshift.policy import StubPolicy
from nightshift.run import main


def _exp(tmp_path, gate):
    cfg = Config(engine="stub", arms=("sft",), days=2, practice_per_day=6, authored_per_day=0,
                 eval_per_template=1, reasoning_eval_n=3, gate_reasoning_n=3, replay_per_day=2,
                 checkpoint_days=(), gate=gate, out_dir=str(tmp_path))
    exp = Experiment(cfg, None)
    exp.policy = StubPolicy(exp.lookup)
    exp._wrap(exp.policy)
    restores = []
    real_restore = exp.policy.restore
    exp.policy.restore = lambda s: (restores.append(1), real_restore(s))
    calls = iter(range(1000))
    # call 0 = base gate, then (before, after) pairs per night: every "after" is a big reasoning regression
    exp.gate_score = lambda day: {"old_modules": None, "reasoning": 0.0 if (k := next(calls)) and k % 2 == 0 else 1.0}
    return exp, restores


@pytest.mark.parametrize("gate,rolled_back", [("log", False), ("enforce", True)])
def test_gate_modes(tmp_path, gate, rolled_back):
    exp, restores = _exp(tmp_path, gate)
    exp.run()
    nights = [d["night"] for d in exp.results["sft"]["days"] if not d["night"].get("skipped")]
    assert nights, "stub teacher should verify something"
    for n in nights:
        assert n["would_reject"] and n["gate_mode"] == gate
        assert n["accepted"] == (not rolled_back)
    assert (len(restores) > 1) == rolled_back  # 1 restore = the initial reset of the arm


def test_no_gate_flag(tmp_path):
    main(["--engine", "stub", "--arms", "sft", "--days", "1", "--practice-per-day", "4", "--authored-per-day", "0",
          "--eval-per-template", "1", "--reasoning-eval-n", "2", "--no-gate", "--checkpoint-days", "0",
          "--out-dir", str(tmp_path)])
    import json
    art = json.loads((tmp_path / "run_artifacts.json").read_text())
    assert art["meta"]["gate_mode"] == "off" and art["meta"]["work_units"]["sft"] > 0
    assert art["arms"]["sft"]["days"][0]["night"]["gate_after"] is None
