"""DeciderBackend: a decision model as a ladder rung. No torch here -- the model is
injected; each test fails if the request mapping or the probability read is cut."""
import math

from awdecide import Ladder, Question
from awdecide import mcp as mcp_mod
from awdecide.backends import DeciderBackend


class FakeDecider:
    """Speaks the pplx_decider response shape for one question `q`."""

    def __init__(self, answer):
        self.answer, self.seen = answer, []

    def decide(self, request):
        self.seen.append(request)
        return {"answers": {"q": self.answer}, "usage": {"input_tokens": 1, "output_tokens": 0}}


def test_bool_maps_to_noul_and_reads_p_true_as_yes():
    fake = FakeDecider({"type": "noul", "noul": 0.83})
    q = Question.bool(prompt="Is the blocker real?")
    d = Ladder([DeciderBackend(decider=fake)]).decide_one(
        "the blocker names a line that does not exist", q)
    spec = fake.seen[0]["questions"]["q"]
    assert spec == {"type": "noul", "instructions": "Is the blocker real?"}
    assert fake.seen[0]["state"].startswith("the blocker")
    assert d.decided and d.value == "yes" and d.backend == "decider"
    assert math.isclose(d.probabilities["yes"], 0.83) and math.isclose(d.probabilities["no"], 0.17)


def test_choice_keeps_option_names_and_the_model_distribution():
    fake = FakeDecider({"type": "choice", "choice": "deep",
                        "probabilities": {"fast": 0.3, "deep": 0.7}})
    d = Ladder([DeciderBackend(decider=fake)]).decide_one("s", Question.choice(["fast", "deep"]))
    assert fake.seen[0]["questions"]["q"]["criteria"] == {"fast": None, "deep": None}
    assert d.value == "deep" and math.isclose(d.probability, 0.7)


def test_score_levels_are_read_by_index_low_to_high():
    fake = FakeDecider({"type": "score", "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}})
    q = Question.score(["easy", "medium", "hard"])
    d = Ladder([DeciderBackend(decider=fake)]).decide_one("s", q)
    assert fake.seen[0]["questions"]["q"]["criteria"] == ["easy", "medium", "hard"]
    assert d.value == "hard" and math.isclose(d.probabilities["easy"], 0.1)


def test_below_min_confidence_is_undecided_and_falls_through():
    fake = FakeDecider({"type": "noul", "noul": 0.55})
    d = Ladder([DeciderBackend(decider=fake)]).decide_one("s", Question.bool(min_confidence=0.9))
    assert not d.decided and d.value is None


def test_a_broken_model_abstains_instead_of_crashing():
    class Broken:
        def decide(self, request):
            raise RuntimeError("CUDA out of memory")

    d = Ladder([DeciderBackend(decider=Broken())]).decide_one("s", Question.bool())
    assert not d.decided and any("decider: abstained" in r for r in d.reasons)


def test_needs_a_checkpoint_or_an_injected_model():
    try:
        DeciderBackend()
    except ValueError:
        return
    raise AssertionError("a backend with no model and no checkpoint was accepted")


def test_env_puts_the_decider_before_the_chat_brain(monkeypatch, tmp_path):
    monkeypatch.setenv("AWDECIDE_DECIDER_CKPT", str(tmp_path))
    monkeypatch.setenv("AWDECIDE_LLM_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("AWDECIDE_LLM_MODEL", "m")
    rungs = mcp_mod.env_ladder().backends
    assert [r.name for r in rungs] == ["decider", "chat"]
    assert rungs[0].checkpoint == str(tmp_path) and rungs[0].device == "cuda"


def test_a_checkpoint_without_decider_source_fails_loudly(tmp_path):
    b = DeciderBackend(str(tmp_path), device="cpu")
    d = Ladder([b]).decide_one("s", Question.bool())
    assert not d.decided and any("no source/src/pplx_decider" in r for r in d.reasons)
