import json
import random

from nightshift.curriculum import Curriculum, accept_authored
from nightshift.env.releases import RELEASES
from nightshift.env.tasks import PracticeSource
from nightshift.lessons import Lesson, LessonStore, error_signature, parse_lesson

RUN = RELEASES[0]


def test_error_signature_groups_same_mistake():
    a = error_signature("tessel.run", "AttributeError: 'Run' object has no attribute 'map'")
    b = error_signature("tessel.run", "AttributeError: 'Run' object has no attribute 'map' (x)")
    assert a == b == "tessel.run:attr:map"
    assert error_signature("tessel.run", "Wrong answer: got 1, expected 2") == "tessel.run:wrong_answer"


def test_parse_lesson():
    assert parse_lesson("blah\nLesson: [tessel.run] knot needs seed=") == "knot needs seed="
    assert parse_lesson("no lesson here") is None


def test_promotion_requires_verification():
    st = LessonStore(promote_threshold=0.5)
    for i in range(3):
        st.note_failure("tessel.run:attr:map", f"t{i}")
    l = st.add(Lesson("a", "tessel.run", "use thread", "tessel.run:attr:map", 1, ["t0"], verified=False, surprise=1))
    assert not l.promoted
    l = st.add(Lesson("b", "tessel.run", "use thread", "tessel.run:attr:map", 1, ["t1"], verified=True, surprise=1))
    assert l.promoted and l.recurrence == 3 and set(l.sources) == {"t0", "t1"}
    assert st.forget_source("t1") == 1 and not st.lessons


def test_curriculum_weights_weak_templates():
    c = Curriculum(random.Random(0), new_module_share=0.0)
    for _ in range(30):
        c.record("run.sum_sq_even", True)
        c.record("run.braid_shift", False)
    plan = c.plan(1, 200, PracticeSource())
    counts = {t: sum(p.template == t for p in plan) for t in ("run.sum_sq_even", "run.braid_shift", "run.max_fold")}
    assert counts["run.braid_shift"] > 2 * counts["run.sum_sq_even"]
    assert counts["run.max_fold"] == 0  # held-out family is never practiced


def _spec(solution, inputs):
    return json.dumps({"prompt": "Write solve(r) ...", "solution": solution, "inputs": inputs})


def test_accept_authored_good_task():
    sol = "from tessel.run import Run\ndef solve(r):\n    return r.thread(lambda x: x + 1).unravel()\n"
    task, why = accept_authored(_spec(sol, ["(Run.of(1, 2),)", "(Run.of(5),)"]), RUN, "x", 1)
    assert why == "accepted" and task.tests[0][1] == "[2, 3]"


def test_accept_authored_rejects_bad_tasks():
    triv = "from tessel.run import Run\ndef solve(r):\n    return None\n"
    assert accept_authored(_spec(triv, ["(Run.of(1),)"]), RUN, "x", 1)[0] is None          # stub passes
    broken = "from tessel.run import Run\ndef solve(r):\n    return r.map(abs)\n"
    assert accept_authored(_spec(broken, ["(Run.of(1),)"]), RUN, "x", 1)[0] is None        # ref crashes
    nolib = "def solve(r):\n    return 1\n"
    assert accept_authored(_spec(nolib, ["(1,)"]), RUN, "x", 1)[0] is None                 # doesn't use module
    assert accept_authored("sure! here is a task", RUN, "x", 1)[0] is None                 # no JSON
