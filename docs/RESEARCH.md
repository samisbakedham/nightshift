# Related work

This list came out of an earlier brainstorm (Claude + Gemini). **Most of these were described as
2025–2026 papers and have not been checked against the actual papers for this repo.** Treat
titles, authors and the claims attributed to them as leads to verify, not citations.

| idea | reported source | used here as |
|---|---|---|
| Self-distillation fine-tuning (SDFT): student matches its own hint-conditioned outputs to avoid forgetting | "SDFT", Shenfeld et al. (reported 2026) | the core training method |
| On-policy updates forget less because they stay close (in KL) to the current model | "RL's Razor: Why online reinforcement learning forgets less", Shenfeld et al. (reported 2025) | the motivation for on-policy rollouts |
| Dense token-level self-distillation can drift/collapse across sequential domains, esp. with a moving teacher | "Denser ≠ better: limits of on-policy self-distillation for continual post-training" (reported) | why the teacher is fixed |
| Self-distillation can shorten/degrade reasoning; anchoring to a fixed initial teacher helps | "Why does self-distillation (sometimes) degrade the reasoning…" (reported) | why the gate checks reasoning, and the replay anchor |
| Merging LoRA adapters for continual learning | "Merge before forget: a single LoRA continual learning via continual merging" (reported) | future work |
| Regularizing activations/features instead of weights | "From weights to features: SAE-guided activation regularization for LLM continual learning" (reported) | future work |
| Selecting training tasks at test time targeted at weaknesses | "Learning on the job: test-time curricula for targeted RL" (reported) | inspiration for the self-curriculum (theirs selects from a task bank; ours also generates tasks) |

Claims from that brainstorm **not** relied on here, because they weren't verified:
- that Hugging Face TRL ships an `SDFTTrainer` (this repo implements the loss itself);
- specific numbers such as "+4 points at 7B, +7 at 14B" or "4× wall-clock vs SFT".

If you post results, read the actual papers first and cite what they actually say.
