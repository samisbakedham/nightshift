"""Per-night LR schedule and grad-norm logging (tiny CPU model)."""

import dataclasses

import pytest

from nightshift.experiment import Config
from nightshift.hf_policy import HFPolicy
from nightshift.policy import TrainItem


def _items(n=5):
    return [TrainItem([{"role": "user", "content": f"q{i}"}], [{"role": "user", "content": f"docs q{i}"}],
                      target=f"a{i}") for i in range(n)]


@pytest.mark.parametrize("method", ["train_sft", "train_sdft"])
def test_cosine_warmup_then_decay(method):
    cfg = dataclasses.replace(Config(), epochs=3, grad_accum=2, lr=2e-4, lr_schedule="cosine", warmup_frac=0.2,
                              max_new_tokens=8)
    pol = HFPolicy(tiny=True, lr=cfg.lr)
    st = getattr(pol, method)(_items(5), cfg)
    lrs, g = st["lrs"], st["grad_norms"]
    assert st["opt_steps"] == len(lrs) == len(g) == 3 * 3  # 3 passes x ceil(5/2) steps
    peak = lrs.index(max(lrs))
    assert 0 < peak < len(lrs) - 1 and max(lrs) == pytest.approx(2e-4)
    assert lrs[0] < lrs[peak] and lrs[-1] < 0.2 * lrs[peak]  # warmup, then decays toward 0
    assert all(b <= a + 1e-12 for a, b in zip(lrs[peak:], lrs[peak + 1:]))
    assert st["grad_norm_max"] == max(g) and st["grad_norm_clipped"] == sum(x > 1 for x in g)


def test_constant_is_default_and_flat():
    cfg = dataclasses.replace(Config(), epochs=2, grad_accum=2, max_new_tokens=8)
    assert cfg.lr_schedule == "constant"
    st = HFPolicy(tiny=True, lr=3e-4).train_sft(_items(4), cfg)
    assert st["lrs"] == [3e-4] * st["opt_steps"]
