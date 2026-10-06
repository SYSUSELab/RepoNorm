import random
from dsp_rng_contracts import ObservedRNG

import importlib
from types import MethodType

import dspy
import pytest
from dspy.propose.grounded_proposer import GroundedProposer
from dspy.utils.dummies import DummyLM


class FixedRNG:
    def __init__(self):
        self.calls = 0

    def choice(self, values):
        self.calls += 1
        return "creative"


def make_proposer(rng):
    program = dspy.Predict("question -> answer")
    proposer = GroundedProposer(
        prompt_model=DummyLM([{"proposed_instruction": "unused"}]),
        program=program,
        trainset=[],
        program_aware=False,
        use_dataset_summary=False,
        set_history_randomly=False,
        verbose=True,
        rng=rng,
    )
    return proposer, program


def test_DSP_8f773dc75e34_N001(capsys):
    """A logging change must preserve the existing actual tip context."""
    proposer, program = make_proposer(random.Random(8129))
    proposer.propose_instructions_for_program([], program, None, {}, N=1, T=0.5)
    lines = capsys.readouterr().out.splitlines()
    selected = [line.removeprefix("Selected tip: ") for line in lines if line.startswith("Selected tip: ")]
    assert len(selected) == 1
    tips = importlib.import_module("dspy.propose.grounded_proposer").TIPS
    tip = tips[selected[0]]
    assert proposer.use_tip is bool(tip)
    history = proposer.prompt_model.history
    assert history, "proposal never reached the final LM"
    request = "\n".join(message["content"] for message in history[-1]["messages"])
    if tip:
        assert tip in request


def test_DSP_8f773dc75e34_N002(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("global random source used instead of supplied RNG")

    for name in ("random", "choice", "choices", "uniform", "randrange", "randint", "sample", "shuffle", "getrandbits"):
        monkeypatch.setattr(random, name, forbidden)

    def execute(verbose):
        rng = ObservedRNG(8129)
        proposer, program = make_proposer(rng)
        proposer.verbose = verbose
        observed = []

        def capture(self, **kwargs):
            observed.append(kwargs["tip"])
            return "instruction"

        proposer.propose_instruction_for_predictor = MethodType(capture, proposer)
        result = proposer.propose_instructions_for_program([], program, None, {}, N=3, T=0.5)
        assert rng.events
        return result, observed, rng.getstate()

    # This is the task-local noninterference obligation, not a fixed draw count.
    assert execute(False) == execute(True)
