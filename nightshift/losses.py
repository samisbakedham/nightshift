"""The distillation loss, kept tiny and separately tested.

The bug to avoid (from the original sketch):

    loss = (s_logp - t_logp).mean()     # on the SAMPLED token's logprobs only

With a frozen teacher, t_logp carries no gradient, so this just minimizes the
student's log-prob of its own samples: it pushes probability away from whatever
it wrote, ignoring the teacher entirely.

What we do instead: at every completion position, compare the student's and the
teacher's FULL next-token distributions over the vocabulary and minimize their KL.

Alignment: student and teacher see different prompts (the teacher's has docs and
lessons), so their sequences have different lengths. ``completion_logits`` slices
out exactly the positions that predict the completion tokens, so position i in
both tensors predicts completion token i.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def completion_logits(logits: torch.Tensor, prompt_len: int, completion_len: int) -> torch.Tensor:
    """logits: [T, V] for the sequence prompt+completion (T = prompt_len + completion_len).

    The logit at position t predicts token t+1, so completion token i (at sequence
    position prompt_len + i) is predicted by position prompt_len + i - 1.
    """
    assert logits.shape[0] == prompt_len + completion_len, (logits.shape, prompt_len, completion_len)
    return logits[prompt_len - 1: prompt_len - 1 + completion_len]


def token_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor | None = None,
             direction: str = "forward", temperature: float = 1.0) -> torch.Tensor:
    """Mean per-token KL between full next-token distributions.

    student_logits, teacher_logits: [N, V] (already aligned). mask: [N] of 0/1.
    direction="forward":  KL(teacher || student) = sum_v p_t (log p_t - log p_s)   (mode-covering)
    direction="reverse":  KL(student || teacher) = sum_v p_s (log p_s - log p_t)   (mode-seeking)
    The teacher is detached: gradients only flow into the student.
    """
    s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t = F.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    if direction == "forward":
        per_tok = (t.exp() * (t - s)).sum(-1)
    elif direction == "reverse":
        per_tok = (s.exp() * (s - t)).sum(-1)
    else:
        raise ValueError(direction)
    if mask is None:
        return per_tok.mean()
    mask = mask.to(per_tok.dtype)
    return (per_tok * mask).sum() / mask.sum().clamp_min(1.0)
