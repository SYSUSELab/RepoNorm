"""Injected-RNG behavioral witness; no private selector or draw-count contract."""

import random

import dspy
import pytest
from dspy.propose.grounded_proposer import GroundedProposer
from dspy.utils.dummies import DummyLM


class ObservedRNG(random.Random):
    def __init__(self, seed):
        super().__init__(seed)
        self.events = []

    def random(self):
        value = super().random()
        self.events.append(("random", value))
        return value

    def getrandbits(self, count):
        value = super().getrandbits(count)
        self.events.append(("getrandbits", count, value))
        return value


def check_injected_rng(monkeypatch, *, weighted=False):
    """Observe public proposal execution, fresh same-seed reproducibility and use."""
    def forbidden(*args, **kwargs):
        pytest.fail("proposal bypassed the supplied RNG for global randomness")

    for name in ("random", "choice", "choices", "uniform", "randrange", "randint", "sample", "shuffle", "getrandbits"):
        monkeypatch.setattr(random, name, forbidden)

    def execute(seed):
        rng = ObservedRNG(seed)
        program = dspy.Predict("question -> answer")
        kwargs = {"tip_weights": {"none": 0, "creative": 3, "simple": 2}} if weighted else {}
        proposer = GroundedProposer(
            prompt_model=DummyLM([{"proposed_instruction": "instruction"}] * 3),
            program=program,
            trainset=[],
            program_aware=False,
            use_dataset_summary=False,
            set_history_randomly=True,
            verbose=False,
            rng=rng,
            **kwargs,
        )
        result = proposer.propose_instructions_for_program([], program, None, {}, N=3, T=0.5)
        assert rng.events, "public proposal never consumed its supplied RNG"
        return result, rng.events, rng.getstate()

    assert execute(8129) == execute(8129)
