from nightshift.env.releases import RELEASES
from nightshift.env.sandbox import extract_code, run_tests
from nightshift.env.tasks import PracticeSource, eval_set, make_task, released_templates
from nightshift.experiment import GATE_SEED_BASE

STUB = "def solve(*a, **k):\n    return None\n"


def test_ten_releases_three_templates_each():
    assert [r.day for r in RELEASES] == list(range(1, 11))
    assert all(len(r.templates) == 3 for r in RELEASES)


def test_every_task_is_solvable_and_nontrivial():
    """Reference passes, a do-nothing stub fails, for eval, gate and practice instances."""
    src = PracticeSource()
    tasks = eval_set(10)
    tasks += [make_task(r, t, GATE_SEED_BASE, "gate") for r, t in released_templates(10)]
    tasks += [src.draw(r, t) for r, t in released_templates(10)]
    for t in tasks:
        assert run_tests(t.ref, t.preamble, t.tests).passed, t.id
        assert not run_tests(STUB, t.preamble, t.tests).passed, t.id


def test_splits_are_disjoint():
    ev = {(t.template, tuple(map(tuple, t.tests))) for t in eval_set(10, per_template=2)}
    gate = {(t.template, tuple(map(tuple, t.tests)))
            for t in (make_task(r, tp, GATE_SEED_BASE, "gate") for r, tp in released_templates(10))}
    src = PracticeSource()
    prac = {(t.template, tuple(map(tuple, t.tests)))
            for _ in range(5) for t in (src.draw(r, tp) for r, tp in released_templates(10))}
    assert not ev & gate and not ev & prac


def test_prompts_never_contain_reference():
    from nightshift import prompts
    for t in eval_set(10):
        text = prompts.task_messages(t)[-1]["content"]
        assert "def solve" not in text and t.ref not in text


def test_wrong_api_gives_readable_feedback():
    t = eval_set(1)[0]
    bad = t.ref.replace(".sieve(", ".filter(")
    r = run_tests(bad, t.preamble, t.tests)
    assert not r.passed and "has no attribute 'filter'" in r.feedback


def test_sandbox_timeout_and_crash():
    t = eval_set(1)[0]
    assert run_tests("def solve(*a):\n    while True: pass\n", t.preamble, t.tests, timeout=3).passed is False
    assert "load" in run_tests("this is not python", t.preamble, t.tests).feedback


def test_extract_code_takes_last_block():
    assert extract_code("x\n```python\na=1\n```\nthen\n```python\nb=2\n```") == "b=2\n"
