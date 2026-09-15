"""
Tests for vindex.encoder's pure logic: cache path/key computation
(Milestone 3.4) and the MuRIL load guard (Milestone 3.2). Does not load
any real model -- see experiments/scripts/beat_the_baseline.py-style
scripts, or a manual run of calibrated_similarity(), for tests that
exercise real encoding (out of scope for the default fast test suite;
nobody in this project has a GPU and model loading is slow).
"""

import os

import pytest

from vindex.calibration import MURIL_MODEL_NAME
from vindex.encoder import CACHE_ROOT, Encoder, MurilWithoutOverrideError, _cache_path, _sanitize

# --- cache path / key computation ---


def test_sanitize_replaces_slash() -> None:
    assert _sanitize("sentence-transformers/LaBSE") == "sentence-transformers__LaBSE"


def test_cache_path_is_deterministic() -> None:
    p1, _ = _cache_path("some-encoder", "hello world", True)
    p2, _ = _cache_path("some-encoder", "hello world", True)
    assert p1 == p2


def test_cache_path_differs_by_is_query() -> None:
    p1, _ = _cache_path("some-encoder", "hello world", True)
    p2, _ = _cache_path("some-encoder", "hello world", False)
    assert p1 != p2


def test_cache_path_differs_by_text() -> None:
    p1, _ = _cache_path("some-encoder", "hello", True)
    p2, _ = _cache_path("some-encoder", "world", True)
    assert p1 != p2


def test_cache_path_under_cache_root() -> None:
    p, d = _cache_path("some-encoder", "hello", True)
    assert p.startswith(CACHE_ROOT)
    assert d.startswith(CACHE_ROOT)
    assert p.endswith(".npy")


def test_cache_root_is_at_repo_root_not_experiments() -> None:
    # Regression: encoder_cache.py's own CACHE_ROOT computes one level
    # short of the actually-populated cache directory on disk. This
    # module must target the real, populated location.
    assert os.path.basename(CACHE_ROOT) == "encoder_cache"
    assert os.path.basename(os.path.dirname(CACHE_ROOT)) != "experiments"


# --- MuRIL guard ---


def test_muril_without_override_raises() -> None:
    with pytest.raises(MurilWithoutOverrideError):
        Encoder(MURIL_MODEL_NAME)


def test_muril_with_override_does_not_raise() -> None:
    enc = Encoder(MURIL_MODEL_NAME, allow_muril=True)
    assert enc.model_name == MURIL_MODEL_NAME


def test_non_muril_encoder_does_not_raise() -> None:
    enc = Encoder("sentence-transformers/all-MiniLM-L6-v2")
    assert enc.model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_default_model_name_used_when_empty() -> None:
    from vindex.calibration import DEFAULT_ENCODER

    enc = Encoder()
    assert enc.model_name == DEFAULT_ENCODER


# --- cache stats before any encoding ---


def test_cache_stats_before_any_encode_calls() -> None:
    enc = Encoder("sentence-transformers/all-MiniLM-L6-v2")
    stats = enc.cache_stats()
    assert stats == {"hits": 0, "misses": 0, "hit_rate": 0.0}
