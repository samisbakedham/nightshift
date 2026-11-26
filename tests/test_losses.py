import pytest

torch = pytest.importorskip("torch")

from nightshift.losses import completion_logits, token_kl  # noqa: E402


def test_kl_zero_iff_equal_and_nonnegative():
    a = torch.randn(5, 11)
    for d in ("forward", "reverse"):
        assert token_kl(a, a, direction=d).item() == pytest.approx(0, abs=1e-6)
        assert token_kl(a, torch.randn(5, 11), direction=d).item() > 0


def test_gradient_flows_only_to_student():
    s = torch.randn(4, 9, requires_grad=True)
    t = torch.randn(4, 9, requires_grad=True)
    token_kl(s, t).backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
    assert t.grad is None


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_student_moves_toward_teacher(direction):
    torch.manual_seed(0)
    teacher = torch.randn(6, 20) * 3
    student = torch.zeros(6, 20, requires_grad=True)
    opt = torch.optim.SGD([student], lr=0.5)
    first = token_kl(student, teacher, direction=direction).item()
    for _ in range(50):
        opt.zero_grad()
        token_kl(student, teacher, direction=direction).backward()
        opt.step()
    last = token_kl(student, teacher, direction=direction).item()
    assert last < first * 0.5
    # the student's argmax now agrees with the teacher's at most positions
    assert (student.argmax(-1) == teacher.argmax(-1)).float().mean() >= 0.8


def test_the_broken_loss_ignores_the_teacher():
    """(s_logp - t_logp).mean() on sampled tokens: gradient identical for ANY teacher."""
    torch.manual_seed(0)
    logits = torch.randn(5, 7, requires_grad=True)
    y = torch.randint(0, 7, (5,))
    grads = []
    for _ in range(2):
        t_logp = torch.log_softmax(torch.randn(5, 7), -1).gather(1, y[:, None]).squeeze(1)  # frozen teacher
        s_logp = torch.log_softmax(logits, -1).gather(1, y[:, None]).squeeze(1)
        (s_logp - t_logp).mean().backward()
        grads.append(logits.grad.clone())
        logits.grad = None
    assert torch.allclose(grads[0], grads[1])  # the teacher had no effect at all


def test_mask():
    s, t = torch.randn(4, 5), torch.randn(4, 5)
    m = torch.tensor([1, 1, 0, 0])
    assert token_kl(s, t, m).item() == pytest.approx(token_kl(s[:2], t[:2]).item(), rel=1e-5)


def test_completion_logits_alignment():
    """Position i of the slice must be the logit that predicts completion token i."""
    P, C, V = 7, 4, 3
    logits = torch.arange(P + C).float()[:, None].expand(P + C, V)  # row index stored in each row
    sl = completion_logits(logits, P, C)
    # completion token i sits at sequence position P+i, predicted by row P+i-1
    assert sl[:, 0].tolist() == [P - 1 + i for i in range(C)]
    # teacher with a longer prompt lines up with the student token for token
    P2 = 12
    logits2 = torch.arange(P2 + C).float()[:, None].expand(P2 + C, V)
    assert (completion_logits(logits2, P2, C)[:, 0] - P2).tolist() == (sl[:, 0] - P).tolist()
