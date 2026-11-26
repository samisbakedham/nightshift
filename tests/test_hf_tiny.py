"""Real training code paths on a tiny random Llama (no download)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from types import SimpleNamespace  # noqa: E402

from nightshift.hf_policy import HFPolicy  # noqa: E402
from nightshift.losses import token_kl  # noqa: E402
from nightshift.policy import TrainItem  # noqa: E402

CFG = SimpleNamespace(epochs=1, grad_accum=1, rollout_temperature=1.0, max_new_tokens=8, kl_direction="forward")

STUDENT = [{"role": "user", "content": "task: add one"}]
TEACHER = [{"role": "user", "content": "DOCS: Run.thread maps. LESSON: use thread.\n\ntask: add one"}]


@pytest.fixture(scope="module")
def pol():
    p = HFPolicy(tiny=True, lr=1e-3, lora_r=8, seed=0)
    with torch.no_grad():  # sharpen the random model's outputs so student/teacher start clearly apart
        p.model.base_model.model.lm_head.weight.mul_(30)
    return p


def _kl_on(pol, y):
    with torch.no_grad():
        with pol.model.disable_adapter():
            t = pol._logits(pol.encode_prompt(TEACHER), y)
        s = pol._logits(pol.encode_prompt(STUDENT), y)
    return token_kl(s, t).item()


def test_sdft_reduces_kl_to_teacher_and_teacher_stays_fixed(pol):
    y = list(b"Run.thread(f)") + [pol.tok.eos_token_id]
    init = pol.snapshot()
    with torch.no_grad(), pol.model.disable_adapter():
        teacher_before = pol._logits(pol.encode_prompt(TEACHER), y).clone()
    kl0 = _kl_on(pol, y)
    # train on a FIXED completion so the measurement is comparable
    pol._generate_ids = lambda prompts, *a, **k: [y for _ in prompts]
    stats = pol.train_sdft([TrainItem(STUDENT, TEACHER)] * 20, CFG)
    kl1 = _kl_on(pol, y)
    assert stats["steps"] == 20
    assert kl0 > 0.05 and kl1 < kl0 * 0.2, (kl0, kl1)
    with torch.no_grad(), pol.model.disable_adapter():
        teacher_after = pol._logits(pol.encode_prompt(TEACHER), y)
    assert torch.allclose(teacher_before, teacher_after)  # adapter-off teacher never moves
    # snapshot / restore = exact rollback
    pol.restore(init)
    assert _kl_on(pol, y) == pytest.approx(kl0, rel=1e-5)
    del pol._generate_ids


def test_sft_reduces_ce(pol):
    item = TrainItem(STUDENT, STUDENT, target="Run.thread")
    stats = pol.train_sft([item] * 20, CFG)
    assert stats["loss_last"] < stats["loss_first"] * 0.5, stats


def test_generate_runs(pol):
    from nightshift.policy import Req
    out = pol.generate([Req(STUDENT, "code"), Req(TEACHER, "code")], max_new_tokens=5)
    assert len(out) == 2
