"""Tests for inference logic (uncertain band, calibration) using stubs.

We avoid loading the real 66M-parameter model in CI by constructing tiny fake
objects that mimic the interface ``predict_url`` / ``predict_proba`` rely on.
This keeps the tests fast and runnable without trained weights.
"""

import types

import torch

from src import config, predict


class _StubConfig:
    """Mimics the HuggingFace model.config fields we use."""

    def __init__(self, id2label):
        self.id2label = id2label
        self.num_labels = len(id2label)


class _StubModel:
    """A fake model whose forward pass returns fixed logits.

    ``logits`` is a 1 x num_labels tensor; the same value is returned for any
    input, which is all the deterministic tests below need.
    """

    def __init__(self, logits, id2label, temperature=1.0):
        self._logits = torch.tensor([logits], dtype=torch.float32)
        self.config = _StubConfig(id2label)
        self.device = "cpu"
        self.temperature = temperature

    def __call__(self, **_kwargs):
        return types.SimpleNamespace(logits=self._logits)

    def to(self, _device):
        return self


def _stub_tokenizer(url, **_kwargs):
    """Return a minimal batch-like object with a .to() method."""
    obj = {"input_ids": torch.zeros((1, 4), dtype=torch.long)}
    return types.SimpleNamespace(to=lambda _d: obj, **obj)


ID2LABEL = {0: "benign", 1: "defacement", 2: "malware", 3: "phishing"}


def test_predict_proba_sums_to_one():
    model = _StubModel([2.0, 1.0, 0.5, 0.1], ID2LABEL)
    scores = predict.predict_proba("http://x.com", _stub_tokenizer, model)
    assert set(scores) == set(ID2LABEL.values())
    assert abs(sum(scores.values()) - 1.0) < 1e-5


def test_confident_prediction_returns_class():
    # Logits heavily favor class 0 (benign) -> well above threshold.
    model = _StubModel([10.0, 0.0, 0.0, 0.0], ID2LABEL)
    label, conf = predict.predict_url("http://x.com", _stub_tokenizer, model)
    assert label == "benign"
    assert conf > config.UNCERTAIN_THRESHOLD


def test_low_confidence_returns_uncertain():
    # Near-uniform logits -> top prob ~0.25, below the 0.60 threshold.
    model = _StubModel([1.0, 1.0, 1.0, 1.0], ID2LABEL)
    label, conf = predict.predict_url("http://x.com", _stub_tokenizer, model)
    assert label == "uncertain"
    assert conf < config.UNCERTAIN_THRESHOLD


def test_temperature_softens_confidence():
    logits = [6.0, 0.0, 0.0, 0.0]
    cold = _StubModel(logits, ID2LABEL, temperature=1.0)
    warm = _StubModel(logits, ID2LABEL, temperature=3.0)
    conf_cold = max(predict.predict_proba("u", _stub_tokenizer, cold).values())
    conf_warm = max(predict.predict_proba("u", _stub_tokenizer, warm).values())
    # Higher temperature must reduce the peak probability.
    assert conf_warm < conf_cold
