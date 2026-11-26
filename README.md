# nightshift

**A model that learns on the job by day and consolidates what it learned into its weights by night.**

A language model is handed `tessel`, a Python library invented for this repo, so no model has seen
it in training. A new module ships every day for 10 days. Each day the model:

1. **works**: attempts tasks that need the new API, closed-book, and gets graded by *running its code*;
2. **reflects**: when it fails, it writes itself a one-line lesson, then proves the lesson works by
   retrying the task with only that lesson in context;
3. **practices**: picks its own weak spots and invents new exercises for itself, which are only
   accepted if its own reference solution runs and a do-nothing stub fails;

and each night it:

4. **consolidates**: distills docs + its verified lessons into a LoRA adapter with on-policy
   self-distillation, then every update is scored on a regression gate (general reasoning + older
   modules). In the benchmark the gate only *logs* regressions, so forgetting stays visible;
   `--gate enforce` rolls regressed updates back instead.

The question: **by day 10, can it use the whole library with zero docs in the prompt, without
forgetting the early modules or getting worse at general reasoning?**

The headline metric is accuracy on **unseen task types**: for every module, one kind of task is
never practiced, never used by the gate, and never trained on. It only appears in the final eval,
so passing it means the model learned the API, not the prompt.

> **Status: the pipeline is built and tested; there are no real results yet.** The numbers in
> `demo/run_artifacts.json` come from a scripted stub and are labeled as such everywhere. Real
> numbers require one GPU run (`scripts/run_cloud.sh`). A local test with Qwen2.5-0.5B ran the
> real path end to end but is too small to learn the library (its teacher solved 0–17% of tasks).

## Results so far (real runs)

| run | model | base | docs in prompt | teacher | student after night 1 |
|---|---|---|---|---|---|
| local, 2 days | Qwen2.5-0.5B | 0/12 | 2/12 | 1/6 | 0/12 |
| Modal A100, day 1 | Qwen2.5-Coder-7B | 0/12 | 8/12 | 13/24 | 0/12 |
| same, after fixing one doc line | Qwen2.5-Coder-7B | 0/12 | 12/12 | 24/24 | 0/12 |
| same, tuned night (5 passes, lr 5e-4) | Qwen2.5-Coder-7B | 0/12 | 12/12 | 24/24 | **12/12** (unseen type 4/4) |

The 7B teacher's first misses were all one mistake caused by a gap in the docs (`knot` didn't say it
returns a plain value). With one weak training pass, the student adopted the teacher's chained style but not yet the
library's names. With five passes at lr 5e-4 (now the default) it solved all 12 held-out tasks
after one night with no docs in its prompt, including the task type it never trained on, and general
reasoning stayed intact (80% base, 93% after). lr 1e-3 over-learned (8/12, unseen 0/4). Full story, charts and the exact code:
the demo page's lab notebook (`demo/`, data in `results/`, built by `scripts/build_journal.py`).

### v1: the full 10 days (Coder-7B, A100, lr 5e-4 × 5 passes, constant LR, gate in log mode)

Score on held-out tasks from every module released so far (6 tasks on day 1, up to 60 on day 10):

| arm | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base (no docs) | 0% | 0% | 0% | 0% | 0% | 0% | 7% | 10% | 9% | 10% | 4% |
| rag (docs in prompt) | 100% | 83% | 94% | 83% | 87% | 83% | 90% | 79% | 83% | 77% | 86% |
| sft | 67% | 67% | 67% | 75% | 43% | 50% | 48% | 50% | 70% | 57% | 59% |
| sdft | 100% | 25% | 44% | 75% | 27% | 47% | 67% | 33% | 50% | 52% | 52% |

Day 10, older modules only: base 11%, rag 78%, sft 59%, sdft 54%. Never-practiced task types: base 10%,
rag 80%, sft 10%, sdft 30%. Reasoning: base/rag 80%; sft 87–100% throughout; sdft fell to 57% on day 5 and
ended at 100%.

**Reading.** sdft learns each new module (100% on day 1, usually 67–100% on the day's module) but its nightly
update often overwrites older ones (day 5: 100% new, 8% older), so it does not beat sft over 10 days. The
gate flagged 5 of sdft's 10 nights (sft: 2), including both of sdft's worst forgetting nights, but it only
logged them, to keep the comparison fair. Eval sets are small, so single-day swings are a handful of tasks.
**Day 5, looked at closely:** not a bad module and not missing replay (night 5 trained on 9 older-module tasks,
including 4 for scroll, and scroll still went 6/6 → 0/6). The update itself destabilized the model: sdft's loss
rose during that night (0.004 → 0.144, also on night 8) and its code came out garbled (`importdef`, invented
names, one call repeated until the token limit). sft broke the same night (an extra `)` in 13 answers). Next:
a gentler nightly update (lower lr, fewer passes, LR decay) and a larger eval set.

The arms ran as separate Modal jobs (same config and seed; each arm restarts from the frozen model). base
comes from `full-coder7b-v1`, which Modal cancelled after 15 minutes for an unknown reason (rag was on day 10). rag
was re-run in full, matched v1's days 1–9 exactly, and is taken from the re-run. Merged with `scripts/merge_runs.py`; provenance is in
`meta.merged_from`. Data: `results/full-coder7b/`.

### v2: gentler nights (lr 2e-4 × 3 passes, per-night cosine decay after 10% warmup)

Rematch for sft and sdft only, identical settings for both, gradient norms logged every step, gate in log mode.
base and rag don't train, so their curves are reused from v1. Data: `results/full-coder7b-v2/` (each arm's
settings in `meta.arm_config`).

| arm | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | mean | v1 mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base (no docs) | 0% | 0% | 0% | 0% | 0% | 0% | 7% | 10% | 9% | 10% | 4% | |
| rag (docs in prompt) | 100% | 83% | 94% | 83% | 87% | 83% | 90% | 79% | 83% | 77% | 86% | |
| sft | 100% | 83% | 100% | 83% | 87% | 72% | 86% | 83% | 70% | 80% | **84%** | 59% |
| sdft | 67% | 33% | 78% | 83% | 67% | 67% | 76% | 67% | 74% | 60% | **67%** | 52% |

10-day means, v2 (v1): older modules sft 87% (58%), sdft 66% (44%); never-practiced task types sft 61% (9%),
sdft 48% (30%); reasoning sft 95% (97%), sdft 83% (86%), base 80%.

**Reading.** The gentler night removed the day-5 collapse in both arms. Day 5, older modules: sdft 8% → 67%,
sft 38% → 92%; sdft reasoning 57% → 83%; unparseable answers on day 5: sdft 8 → 0, sft 13 → 0. sdft's largest
gradient norm per night stayed between 0.4 and 2.1. But sft still beats sdft: closed-book, sft v2 is close to
docs-in-prompt (84% vs 86%) and generalizes better to task types it never practiced (61% vs 48%). Eval sets are
small (6–60 tasks a day), so differences of a few points are noise. Next: a larger eval set, then work out why
on-policy sdft trails sft here.

## The four arms

All arms start from the same weights, get the same daily releases and are scored on the same
held-out eval. (Half of each day's practice is weakness-weighted, so it adapts per arm.)

| arm | memory at test time | learns by | what it shows |
|---|---|---|---|
| `base` | none | nothing | the floor: what the model can guess |
| `rag` | docs + lessons in every prompt | nothing (in-context only) | the ceiling for "just give it the docs"; prompt grows daily |
| `sft` | **none** | nightly fine-tuning on the teacher's verified solutions + reasoning replay | off-policy imitation (known to forget) |
| `sdft` | **none** | nightly on-policy self-distillation + reasoning replay | the method under test |

The **teacher** for both `sft` and `sdft` is identical: the frozen initial weights (LoRA switched
off) with docs + promoted lessons in context. So `sft` vs `sdft` isolates the learning method.

**What each night trains on, identical for both arms (days 1–10):**
- **tessel tasks, new and old:** the day's practice tasks the teacher solved. About 9 of 24 practice tasks a day
  come from *earlier* modules (weakness-weighted), so older modules are replayed implicitly, typically 6–10
  training items a night (25–35%). There is no fixed old-module replay buffer, so coverage varies.
- **reasoning replay:** the same 8 general-reasoning prompts
(chosen by day). `sdft` samples its own answers and is pulled toward the frozen base model's
distribution (token-level KL, no docs on either side). `sft` trains with cross-entropy on the frozen
base model's answers to those prompts, keeping only the ones graded correct. Both train only on
teacher-verified tasks, skip a night when the teacher verified nothing, and use the same nightly update
(v1: 5 passes at lr 5e-4, constant; v2: 3 passes at lr 2e-4, cosine decay, via `--epochs 3 --lr 2e-4 --lr-schedule cosine`).

**Fairness:** the benchmark runs with `--gate log` (the default): updates are scored but never
rolled back, so if `sft` forgets catastrophically, that shows up in the retention and reasoning
charts instead of being quietly undone. The night log records each update's pre-rollback gate
score and whether it *would* have been rejected.

A win looks like: `sdft` ≈ `rag` on unseen-type accuracy with **0 memory tokens**, `sdft` > `sft` on
earlier-module retention, and `sdft` ≈ `base` on general reasoning.

## How it works

```mermaid
flowchart LR
  subgraph Day["Day d (new module ships)"]
    P[curriculum picks tasks<br/>weak spots + new module] --> A[closed-book attempt]
    S[model invents exercises<br/>validated in sandbox] --> A
    A -->|run code vs hidden tests| G{pass?}
    G -- no --> R[reflect → lesson]
    R --> V[verify: retry with<br/>ONLY the lesson]
    V --> L[(lesson store<br/>surprise · recurrence · provenance)]
  end
  subgraph Night
    L -->|promoted lessons + docs| T[frozen teacher<br/>adapter OFF]
    A2[student rollouts<br/>adapter ON, no docs] --> K[per-token KL over full vocab]
    T --> K
    RP[reasoning replay] --> K
    K --> U[LoRA update]
    U --> GT{regression gate}
    GT -- regressed --> RB[log it; rollback only<br/>with --gate enforce]
  end
  GT -- ok --> E[held-out eval: all modules + reasoning]
```

**The loss** (`nightshift/losses.py`): at every position of the student's own sampled completion,
minimize KL between the student's and teacher's *full next-token distributions* (forward KL by
default, `--kl-direction reverse` available). The student and teacher prompts have different
lengths; `completion_logits` aligns them token for token. The tempting shortcut
`(s_logp - t_logp).mean()` on sampled tokens is wrong: with a frozen teacher its gradient doesn't
depend on the teacher at all. There's a test proving that (`tests/test_losses.py`).

**Why the result would be trustworthy:**
- `tessel` is invented, with unusual names (`Run.knot(fn, seed=)` not `reduce`), so the base model can't just guess. (Some task prompts use the module's verbs, e.g. "seal", so prompts carry a little API signal; the `base` arm measures how much.)
- One task type per module is held out entirely; the headline number is measured only on those.
- Every task is graded by executing code against hidden tests. No LLM judge anywhere.
- Eval, gate and practice instances of practiced task types come from disjoint seed ranges, and self-made exercises identical to an eval prompt are rejected (tested).
- Reference solutions never appear in any prompt (tested).
- Every eval prompt and output is saved; the demo shows the exact prompt for each arm.
- No arm's regressions are hidden: the gate logs instead of rolling back in the benchmark.
- The `base` arm is the floor; the `rag` arm is the ceiling.

More detail and the open risks: [`docs/DESIGN.md`](docs/DESIGN.md). Papers this builds on: [`docs/RESEARCH.md`](docs/RESEARCH.md).

## Run it

**Laptop (no GPU), proves the pipeline:**

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python -m pytest -q                                          # 29 tests, ~30s
python -m nightshift.run --engine stub --out-dir runs/dry    # full 10 days, scripted model, ~2 min
python -m nightshift.run --engine tiny --days 2 --practice-per-day 4 --max-new-tokens 24 --out-dir runs/tiny
```

`stub` is a scripted fake model that exercises every code path (lessons, promotion, authored tasks,
gate, artifacts). `tiny` is a real 2-layer random transformer that runs the actual training code
(rollouts, KL, LoRA, rollback). **Neither produces meaningful results.**

**Real run on [Modal](https://modal.com) (easiest; the free Starter plan's $30/month ≈ 12 h of A100-80GB):**

```bash
pip install modal
modal setup                                                  # sign in once
modal run scripts/modal_run.py --mode preflight              # GO / NO-GO + time estimate (~$1-2)
modal run --detach scripts/modal_run.py --mode full          # 10 days; --mode quick = 3 days
```

Options: `--model Qwen/Qwen2.5-Coder-14B-Instruct`, `--gpu L40S|H100` (default `A100-80GB`),
`--extra "--arms base,sdft"`. `--detach` keeps the job alive if your laptop sleeps. Everything is
written to the Modal Volume `nightshift-vol` (model cache + `runs/<name>/`), committed after every
simulated day, so a run cut short by the 12 h timeout or by credits keeps its finished days. When the
job ends, `run_artifacts.json` and `run.log` are downloaded to `runs/<name>/`; if you were disconnected,
fetch them with `modal volume get nightshift-vol runs/<name> runs/`, then `bash scripts/publish_run.sh runs/<name>`.

**Real run on your own box (one 80 GB GPU, A100/H100), default model `Qwen/Qwen2.5-Coder-7B-Instruct`:**

```bash
gh auth login                                   # the repo is private
gh repo clone samisbakedham/nightshift && cd nightshift
tmux                                            # so an SSH drop doesn't kill the run
bash scripts/run_cloud.sh preflight             # tests + day 1 of base/rag/sdft -> GO / NO-GO + time estimate
bash scripts/run_cloud.sh quick                 # 3 days, all 4 arms
bash scripts/run_cloud.sh full                  # 10 days, all 4 arms
bash scripts/publish_run.sh runs/<dir>          # commit + push the results before you kill the pod
```

`preflight` prints a gate summary: `base_acc` (want < 25%: the model can't already guess the API),
`rag_acc`, `teacher_pass_rate` (want ≥ 70%: the docs-in-context teacher can actually solve the
tasks, otherwise there is nothing to distill) and a rough time estimate for the full run. If the
teacher is too weak, use a bigger model: `MODEL=Qwen/Qwen2.5-Coder-14B-Instruct bash scripts/run_cloud.sh preflight`.
Extra flags pass through, e.g. `bash scripts/run_cloud.sh quick --arms base,sdft --gate off`.

`publish_run.sh` saves `run_artifacts.json` (to `demo/` and `results/<run>/`) and the run log. It
does **not** save `traces.jsonl` or the LoRA adapters; copy those off the pod yourself if you want them.

Runtime and cost are **not measured yet**; the preflight prints an estimate. Rough guess: several
hours of generation-bound work for the full run.

Artifacts are rewritten after every simulated day, so a crash loses at most one day.
Outputs: `runs/<timestamp>/run_artifacts.json` (metrics + showcase traces), `traces.jsonl` (every
prompt and output), and LoRA adapters for days 1, 5 and 10.

## Demo page

`demo/` is a static page (no build step) that reads `demo/run_artifacts.json`:
scoreboard, six charts over the 10 days, a day slider, a side-by-side of all four arms on the same
held-out task with a raw-prompt inspector, the lessons the model wrote, and the nightly gate log.
It shows a red **DRY-RUN PLACEHOLDER** banner whenever the artifacts didn't come from a real model.

```bash
python3 -m http.server 4317 --directory demo     # open http://localhost:4317
npx vercel deploy demo                            # or drag the demo/ folder into Vercel
```

## Repo map

```
nightshift/
  env/tessel_lib/tessel/   the invented library (10 modules)
  env/releases.py          release schedule, docs, 30 task templates with reference solutions
  env/tasks.py             task instances + disjoint eval / gate / practice splits
  env/sandbox.py           runs model code in a subprocess with timeout + rlimits
  env/reasoning.py         generated general-reasoning set (eval + replay)
  prompts.py               every prompt, in one place
  lessons.py               reflect → verify → score → promote, with provenance
  curriculum.py            weakness-weighted practice + self-authored task validation
  losses.py                full-vocab token KL + alignment
  policy.py                Policy interface + scripted StubPolicy
  hf_policy.py             HF model + LoRA; teacher = adapter disabled
  experiment.py            the day/night loop, gate, metrics, artifacts
  run.py                   CLI
scripts/run_cloud.sh       GPU run: preflight | quick | full
scripts/publish_run.sh     push a run's results to GitHub
scripts/modal_run.py       same three modes on Modal, with a persistent volume
demo/                      static demo page
tests/                     29 tests
```

## Swapping the environment

Everything domain-specific is in `nightshift/env/`. To teach the model something else, keep the
same `Task` shape (prompt, hidden tests, reference solution) and a release schedule; the lesson,
curriculum, training and demo layers don't care what the tasks are about.
