"""Tests for the download module's fast-transfer opt-in (pure env logic,
no network)."""
import importlib.util

import pytest

from dilemma._download import _enable_fast_transfer

_HAS_XET = importlib.util.find_spec("hf_xet") is not None
_HAS_TRANSFER = importlib.util.find_spec("hf_transfer") is not None


@pytest.mark.skipif(not _HAS_XET, reason="hf_xet not installed")
def test_sets_xet_high_performance(monkeypatch):
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    _enable_fast_transfer()
    import os
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"


def test_respects_explicit_opt_out(monkeypatch):
    monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", "0")
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "0")
    _enable_fast_transfer()
    import os
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "0"
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "0"


@pytest.mark.skipif(_HAS_XET or _HAS_TRANSFER,
                    reason="a fast backend is installed")
def test_noop_without_backends(monkeypatch):
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)
    _enable_fast_transfer()
    import os
    assert "HF_XET_HIGH_PERFORMANCE" not in os.environ
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ
