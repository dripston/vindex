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

import vindex.encoder
from vindex.calibration import MURIL_MODEL_NAME
from vindex.encoder import CACHE_ROOT, Encoder, MurilWithoutOverrideError, _cache_path, _sanitize

# --- cache path / key computation ---


def test_sanitize_replaces_slash() -> None:
    assert _sanitize("sentence-transformers/LaBSE") == "sentence-transformers__LaBSE"


def test_cache_path_is_deterministic() -> None:
    p1, _ = _cache_path("some-encoder", None, "hello world", True)
    p2, _ = _cache_path("some-encoder", None, "hello world", True)
    assert p1 == p2


def test_cache_path_differs_by_is_query() -> None:
    p1, _ = _cache_path("some-encoder", None, "hello world", True)
    p2, _ = _cache_path("some-encoder", None, "hello world", False)
    assert p1 != p2


def test_cache_path_differs_by_text() -> None:
    p1, _ = _cache_path("some-encoder", None, "hello", True)
    p2, _ = _cache_path("some-encoder", None, "world", True)
    assert p1 != p2


def test_cache_path_under_cache_root() -> None:
    p, d = _cache_path("some-encoder", None, "hello", True)
    assert p.startswith(CACHE_ROOT)
    assert d.startswith(CACHE_ROOT)
    assert p.endswith(".npy")


def test_cache_path_differs_by_revision() -> None:
    # Regression (found by an independent outside review): the cache
    # key used to be derived from encoder_name alone, so a model
    # updated in place under the same repo name would silently reuse
    # a stale cached embedding from the old weights. Different
    # revisions of the same encoder name must land in different cache
    # paths.
    p1, d1 = _cache_path("some-encoder", "abc123", "hello world", True)
    p2, d2 = _cache_path("some-encoder", "def456", "hello world", True)
    assert p1 != p2
    assert d1 != d2


def test_cache_path_with_revision_differs_from_no_revision() -> None:
    p_none, _ = _cache_path("some-encoder", None, "hello world", True)
    p_rev, _ = _cache_path("some-encoder", "abc123", "hello world", True)
    assert p_none != p_rev


def test_cache_path_same_revision_is_deterministic() -> None:
    p1, _ = _cache_path("some-encoder", "abc123", "hello world", True)
    p2, _ = _cache_path("some-encoder", "abc123", "hello world", True)
    assert p1 == p2


def test_cache_root_is_not_relative_to_repo_or_install_path() -> None:
    # Regression: CACHE_ROOT used to be computed as three dirname() hops
    # from __file__, resolving to <repo_root>/encoder_cache/ in a git
    # checkout -- but to somewhere inside the Python install directory
    # (e.g. site-packages' grandparent) for an installed package, where
    # os.makedirs() raises PermissionError on any normal install. Must
    # be a real user cache directory instead, never inside this
    # package's own install location.
    package_dir = os.path.dirname(os.path.abspath(vindex.encoder.__file__))
    assert not CACHE_ROOT.startswith(package_dir)
    assert "vindex" in CACHE_ROOT.lower()


def test_cache_root_honors_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VINDEX_CACHE_DIR", "/tmp/my-custom-vindex-cache")
    assert vindex.encoder._default_cache_root() == "/tmp/my-custom-vindex-cache"


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


# --- revision resolution (model-revision cache pinning) ---


def test_revision_is_none_before_load() -> None:
    enc = Encoder("sentence-transformers/all-MiniLM-L6-v2")
    assert enc.revision is None


def test_resolve_st_revision_returns_none_for_unexpected_shape() -> None:
    # Best-effort: if a SentenceTransformer wrapper never exposes
    # auto_model.config._commit_hash (e.g. a future/unusual model
    # class), resolution must degrade to None (cache keys on name
    # alone) rather than raising out of Encoder.load().
    class _FakeModule:
        def modules(self) -> list[object]:
            return [object()]

    assert Encoder._resolve_st_revision(_FakeModule()) is None


def test_resolve_st_revision_finds_commit_hash() -> None:
    class _Config:
        _commit_hash = "abc123"

    class _AutoModel:
        config = _Config()

    class _Submodule:
        auto_model = _AutoModel()

    class _FakeModel:
        def modules(self) -> list[object]:
            return [self, _Submodule()]

    assert Encoder._resolve_st_revision(_FakeModel()) == "abc123"
