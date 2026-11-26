# Papers this builds on

| paper | what it says | how it's used here |
|---|---|---|
| Shenfeld, Damani, Hübotter, Agrawal. **Self-Distillation Enables Continual Learning.** ICML 2026. [arXiv:2601.19897](https://arxiv.org/abs/2601.19897) · [project page](https://self-distillation.github.io/SDFT) | Self-Distillation Fine-Tuning (SDFT): the model, conditioned on a demonstration in context, acts as its own teacher, so training is on-policy. Reports higher new-task accuracy than SFT with much less forgetting, with the gap growing with model size. | The core training method. **This repo is a variant:** the teacher is conditioned on library docs + the model's own verified lessons instead of demonstrations, and it is the frozen initial weights (LoRA off). The loss is implemented here (`nightshift/losses.py`), not taken from the paper's code. |
| Shenfeld, Pari, Agrawal. **RL's Razor: Why Online Reinforcement Learning Forgets Less.** ICLR 2026. [arXiv:2509.04259](https://arxiv.org/abs/2509.04259) | At matched new-task accuracy, on-policy RL forgets less than SFT; among solutions, on-policy training favors the one closest in KL to the original model. | The motivation for on-policy rollouts, and for the `sft` vs `sdft` comparison. |

Numbers above are the papers' own claims, not reproduced here. The point of this repo is to test
the idea in a new setting (an invented library, learned day by day) with its own measurements.
