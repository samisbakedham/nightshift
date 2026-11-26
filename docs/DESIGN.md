# Design notes

## The core idea

In-context learning is instant but temporary: the docs have to ride along in every prompt forever.
Fine-tuning is permanent but, done naively, overwrites old skills. nightshift tries to get both:
learn in context during the day, then *move* what was learned into the weights at night, using
the model's in-context self as its own teacher.

## Why on-policy self-distillation

- **Teacher = the model itself with the hint.** The frozen base model with docs + lessons in its
  prompt already knows the right answer. The student is the same model without the hint. Training
  closes that gap, so the student internalizes what the hint provided.
- **On-policy.** The student trains on text it generated itself, so each update is a small nudge
  to its current behavior instead of a pull toward someone else's text. The hypothesis, tested by
  the `sft` vs `sdft` comparison, is that this forgets less.
- **Full-distribution KL.** At each position we match the whole next-token distribution, which
  carries far more signal than one sampled token and has the correct gradient (see `losses.py`).
- **Fixed teacher.** The teacher is the initial weights (adapter disabled), never an exponential
  moving average of the student, so it can't drift across days.
- **Replay anchor.** Each night also includes general reasoning prompts where the teacher has no
  hint. On those, the loss is just "stay close to the base model".

## What goes into the weights (and what doesn't)

Lessons are the bottleneck. Raw transcripts are not distilled. A lesson is only promoted into the
teacher's context (and so into the weights) if it is:

- **verified**: the closed-book model passes the task with only that lesson in context;
- **surprising and recurring**: `priority = 0.5·surprise + 0.5·min(recurrence, 3)/3 ≥ threshold`,
  where surprise is the student's failure rate on that task family and recurrence counts distinct
  tasks that hit the same error signature (e.g. `tessel.run:attr:map`).

Unpromoted lessons stay in retrieval memory, which the `rag` arm uses. Every lesson stores the task
ids it came from, so you can trace or remove it (`LessonStore.forget_source`).

## The self-curriculum

- Half of each day's practice budget goes to the newly released module (the day's job).
- The rest is sampled across all released task families, weighted by
  `(1 − smoothed pass rate) + exploration bonus`.
- The model also invents exercises for its weakest module. An exercise is accepted only if its
  reference solution runs in the sandbox, uses the module, and a `return None` stub fails the tests.
  Expected outputs come from executing the model's own solution, so accepted tasks are
  self-consistent. They are not guaranteed to be interesting or correct with respect to the prompt
  text; that's a known limitation.

## Controls

- `base` and `rag` bracket the result (floor and ceiling).
- One task type per module is held out of practice, the gate and training; it's the headline metric.
- `sft` and `sdft` share the same teacher and the same filtering rule (teacher-verified tasks only),
  and both get replay on the same 8 reasoning prompts per night: `sdft` as a KL anchor to the base
  model on its own samples, `sft` as cross-entropy on the base model's answers graded correct.
- The regression gate runs in `log` mode for the benchmark: each update is scored (pre-rollback)
  and flagged if it regressed, but never undone. Rolling back would hide exactly the forgetting
  that `sft` vs `sdft` is meant to measure. `--gate enforce` is the deployment mode; `--gate off`
  (`--no-gate`) skips gate scoring to save GPU time.
- The eval, gate and practice splits are seed-disjoint. The gate never uses eval tasks, so rollback
  decisions don't peek at the reported numbers.
- Reasoning eval items are generated, not taken from a public benchmark, so there's no
  contamination question.

## Known risks / what could go wrong in the real run

- **Teacher too weak.** If the base model can't use the docs in context (low `teacher_pass_rate` in
  the night log), there is nothing to distill. This happened locally with Qwen2.5-0.5B (teacher
  solved 0–17%). `run_cloud.sh preflight` checks it first; try Coder-14B if Coder-7B is below 70%.
- **Few training items per night.** Only teacher-verified tasks are trained on. Raise
  `--practice-per-day` if the night log shows tiny `train_items`. Defaults: 5 passes per night on fresh rollouts, `--lr 5e-4`, `--grad-accum 2`, LoRA r=16 (picked by a 1-day test; 1e-3 overfit).
- **Lesson quality.** Real lessons are free text from the model. Verification filters out useless
  ones, but surprise is a cheap proxy (task failure rate), not a token-level loss.
- **Variance.** 60 held-out tasks and 30 reasoning items by day 10 are small. Run more seeds before
  claiming small differences.
- **Sandboxing.** Model code runs in a subprocess with timeouts and rlimits. That's containment,
  not security; use a disposable VM.
- **Speed.** Generation uses plain HF `generate` (bf16 on CUDA, KV cache on, batches of
  `--gen-batch 32`, `--max-new-tokens 384`). The preflight reports how many generations hit the
  token cap. Swapping in vLLM for rollouts and evals is the obvious next speedup.

## Ideas not built yet

- A breaking-change day (an API renamed in v2): does the model update its beliefs or cling to old habits?
- Per-module adapters + merging, instead of one adapter for everything.
- Token-level surprise (student NLL on the teacher's solution) instead of task failure rate.
