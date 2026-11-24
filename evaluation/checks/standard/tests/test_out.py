import inspect

import numpy as np
import torch
import xgrammar as xgr

from outlines.backends.xgrammar import XGrammarLogitsProcessor
from outlines.processors.base_logits_processor import OutlinesLogitsProcessor


class RecordingProcessor(OutlinesLogitsProcessor):
    def __init__(self):
        super().__init__("numpy")
        self.seen = []

    def process_logits(self, input_ids, logits):
        self.seen.append((input_ids.shape, logits.shape))
        return logits + 1


def test_OUT_N01_shared_adapter_preserves_public_shapes():
    processor = RecordingProcessor()
    one = processor(np.array([1, 2]), np.array([0.0, 2.0, 4.0]))
    two = processor(
        np.array([[1, 2], [3, 4]]),
        np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]]),
    )

    assert one.shape == (3,)
    assert two.shape == (2, 3)
    assert processor.seen == [((1, 2), (1, 3)), ((2, 2), (2, 3))]


def test_OUT_N02_reset_starts_a_fresh_generation():
    compiler = xgr.GrammarCompiler(
        xgr.TokenizerInfo(["a", "b", "c", "d", "<eos>"], stop_token_ids=[4])
    )
    grammar = compiler.compile_regex("ab|cd")
    parameters = inspect.signature(XGrammarLogitsProcessor).parameters
    kwargs = {}
    library = parameters.get("tensor_library_name")
    if library is not None and library.default is inspect.Parameter.empty:
        kwargs["tensor_library_name"] = "torch"
    processor = XGrammarLogitsProcessor(grammar, **kwargs)

    input_ids = torch.tensor([4])
    logits = torch.zeros(5)
    first = processor(input_ids, logits.clone())
    processor.reset()
    restarted = processor(input_ids, logits.clone())

    assert torch.equal(torch.isfinite(first), torch.isfinite(restarted))
    assert first.shape == restarted.shape == (5,)
