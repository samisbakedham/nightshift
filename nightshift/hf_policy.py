"""A real model: Hugging Face causal LM + one LoRA adapter (PEFT).

Teacher = the SAME weights with the adapter switched off (``disable_adapter``).
That gives a fixed, never-drifting initial teacher for free: no second copy of
the model in memory, and no EMA teacher that can drift across days.

Engines:
  * ``--engine hf --model Qwen/Qwen2.5-7B-Instruct``   the real experiment (GPU)
  * ``--engine tiny``  a 2-layer random Llama + byte tokenizer, built locally from
    a config (no download). It fails every task; it exists to run the real
    training code paths (rollouts, alignment, KL, LoRA updates, rollback) on a laptop.
"""

from __future__ import annotations

import copy
import random
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from .losses import completion_logits, token_kl
from .policy import Policy, Req, TrainItem
from .prompts import render


class ByteTokenizer:
    """UTF-8 bytes + 3 specials. Only for the tiny smoke-test model."""
    pad_token_id, eos_token_id, bos_token_id = 256, 257, 258
    vocab_size = 259
    chat_template = None

    def encode(self, text: str) -> list[int]:
        return [self.bos_token_id] + list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=True) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFPolicy(Policy):
    def __init__(self, model_name: str | None = None, tiny: bool = False, lora_r: int = 16,
                 lora_alpha: int = 32, lr: float = 1e-4, gen_batch: int = 8, seed: int = 0):
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(seed)
        random.seed(seed)
        self.tiny = tiny
        self.lr = lr
        self.gen_batch = gen_batch
        if tiny:
            from transformers import LlamaConfig, LlamaForCausalLM
            self.name = "tiny-random-llama"
            self.device = "cpu"
            self.tok = ByteTokenizer()
            cfg = LlamaConfig(vocab_size=self.tok.vocab_size, hidden_size=64, intermediate_size=128,
                              num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                              max_position_embeddings=8192, pad_token_id=256, eos_token_id=257, bos_token_id=258)
            base = LlamaForCausalLM(cfg)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.name = model_name
            self.device = _device()
            self.tok = AutoTokenizer.from_pretrained(model_name)
            if self.tok.pad_token_id is None:
                self.tok.pad_token = self.tok.eos_token
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        base.to(self.device)
        lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.model = get_peft_model(base, lcfg)

    # ------------------------------------------------------------------ tokens
    def encode_prompt(self, messages: list[dict]) -> list[int]:
        if getattr(self.tok, "chat_template", None):
            text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            return self.tok(text, add_special_tokens=False).input_ids
        return self.tok.encode(render(messages))

    def encode_completion(self, text: str) -> list[int]:
        if self.tiny:
            return list(text.encode("utf-8")) + [self.tok.eos_token_id]
        return self.tok(text, add_special_tokens=False).input_ids + [self.tok.eos_token_id]

    def count_tokens(self, messages):
        return len(self.encode_prompt(messages))

    # ------------------------------------------------------------------ generation
    def _ctx(self, use_base: bool):
        return self.model.disable_adapter() if use_base else nullcontext()

    @torch.no_grad()
    def _generate_ids(self, prompts: list[list[int]], temperature: float, max_new_tokens: int,
                      use_base: bool = False) -> list[list[int]]:
        self.model.eval()
        out_all = []
        pad, eos = self.tok.pad_token_id, self.tok.eos_token_id
        for i in range(0, len(prompts), self.gen_batch):
            chunk = prompts[i:i + self.gen_batch]
            L = max(map(len, chunk))
            ids = torch.tensor([[pad] * (L - len(p)) + p for p in chunk], device=self.device)
            att = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in chunk], device=self.device)
            # repetition_penalty pinned to 1.0 so sampled rollouts come from the same distribution we distill on
            kw = dict(do_sample=temperature > 0, max_new_tokens=max_new_tokens, pad_token_id=pad, eos_token_id=eos,
                      repetition_penalty=1.0)
            if temperature > 0:
                kw.update(temperature=temperature, top_p=1.0, top_k=0)
            with self._ctx(use_base):
                gen = self.model.generate(input_ids=ids, attention_mask=att, **kw)
            for row in gen[:, L:].tolist():
                if eos in row:
                    row = row[: row.index(eos) + 1]
                row = [t for t in row if t != pad]
                out_all.append(row)
        return out_all

    def generate(self, reqs: list[Req], temperature=0.0, max_new_tokens=512, use_base=False):
        ids = self._generate_ids([self.encode_prompt(r.messages) for r in reqs], temperature, max_new_tokens, use_base)
        return [self.tok.decode(r, skip_special_tokens=True) for r in ids]

    # ------------------------------------------------------------------ training
    def _logits(self, prompt_ids: list[int], comp_ids: list[int]) -> torch.Tensor:
        seq = torch.tensor([prompt_ids + comp_ids], device=self.device)
        out = self.model(input_ids=seq).logits[0]
        return completion_logits(out, len(prompt_ids), len(comp_ids))

    def _opt(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=0.0)

    def train_sdft(self, items: list[TrainItem], cfg) -> dict:
        """On-policy self-distillation.

        For each prompt: the STUDENT (adapter on, closed-book prompt) samples a
        completion y. Then at every position of y we pull the student's full
        next-token distribution toward the frozen TEACHER's (adapter off, prompt
        with docs + lessons), via token_kl. Replay items use the plain prompt for
        the teacher too, which anchors general behavior to the base model.
        """
        opt = self._opt()
        losses, n_tok = [], 0
        for epoch in range(cfg.epochs):
            order = list(range(len(items)))
            random.shuffle(order)
            rolls = self._generate_ids([self.encode_prompt(items[i].student_messages) for i in order],
                                       cfg.rollout_temperature, cfg.max_new_tokens)
            self.model.train()
            opt.zero_grad()
            for step, (i, y) in enumerate(zip(order, rolls), 1):
                if not y:
                    continue
                it = items[i]
                ps, pt = self.encode_prompt(it.student_messages), self.encode_prompt(it.teacher_messages)
                with torch.no_grad(), self.model.disable_adapter():
                    t_logits = self._logits(pt, y)
                s_logits = self._logits(ps, y)
                loss = token_kl(s_logits, t_logits, direction=cfg.kl_direction)
                (loss / cfg.grad_accum).backward()
                losses.append(loss.item())
                n_tok += len(y)
                if step % cfg.grad_accum == 0 or step == len(order):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
        return {"steps": len(losses), "loss_first": losses[0] if losses else None,
                "loss_last": losses[-1] if losses else None,
                "loss_mean": sum(losses) / len(losses) if losses else None, "tokens": n_tok}

    def train_sft(self, items: list[TrainItem], cfg) -> dict:
        """Plain supervised fine-tuning: cross-entropy on given target texts."""
        opt = self._opt()
        losses = []
        self.model.train()
        for epoch in range(cfg.epochs):
            order = list(range(len(items)))
            random.shuffle(order)
            opt.zero_grad()
            for step, i in enumerate(order, 1):
                it = items[i]
                ps, y = self.encode_prompt(it.student_messages), self.encode_completion(it.target)
                logits = self._logits(ps, y)
                loss = F.cross_entropy(logits.float(), torch.tensor(y, device=self.device))
                (loss / cfg.grad_accum).backward()
                losses.append(loss.item())
                if step % cfg.grad_accum == 0 or step == len(order):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
        return {"steps": len(losses), "loss_first": losses[0] if losses else None,
                "loss_last": losses[-1] if losses else None,
                "loss_mean": sum(losses) / len(losses) if losses else None}

    # ------------------------------------------------------------------ checkpoints
    def snapshot(self):
        from peft import get_peft_model_state_dict
        return {k: v.detach().to("cpu", copy=True) for k, v in get_peft_model_state_dict(self.model).items()}

    def restore(self, snap):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, copy.deepcopy(snap))

    def save(self, path: str):
        self.model.save_pretrained(path)
