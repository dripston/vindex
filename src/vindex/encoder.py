"""Sentence-encoder loading, encoding, and on-disk embedding cache.

Used by :func:`vindex.calibrated_similarity`. The encoder backends
(sentence-transformers, transformers, torch) are optional, multi-GB
dependencies installed with ``pip install vindex[similarity]``; this
module imports them lazily, so ``import vindex`` never requires them.

The default encoder is ``calibration.DEFAULT_ENCODER``
(paraphrase-multilingual-mpnet-base-v2), the best all-around performer in
the calibration table. Loading ``google/muril-base-cased`` raises
:class:`MurilWithoutOverrideError` unless ``allow_muril=True`` is passed;
see ``calibration.MURIL_WARNING`` for why.

Embeddings are cached on disk, keyed by encoder name, resolved model
revision (the commit hash transformers already resolves at load time),
query/passage role, and the SHA-256 of the text. Including the revision
means a model updated in place on the Hugging Face Hub does not reuse
stale embeddings. Revision detection is best-effort: if a model wrapper
does not expose a commit hash, the cache is keyed on the name alone. For
a hard reproducibility guarantee, point ``VINDEX_CACHE_DIR`` at a fresh
directory whenever you pin a model revision. The cache location is
``$VINDEX_CACHE_DIR`` if set, otherwise the platform's user cache
directory. If the cache is not writable, encoding still succeeds without
caching.
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Any

from vindex.calibration import MURIL_MODEL_NAME, MURIL_WARNING

if TYPE_CHECKING:
    import numpy as np

def _default_cache_root() -> str:
    """Return the user cache directory for encoder embeddings.

    Resolution order: ``$VINDEX_CACHE_DIR``; ``$XDG_CACHE_HOME/vindex``;
    ``%LOCALAPPDATA%/vindex/cache`` on Windows; ``~/.cache/vindex``.
    """
    override = os.environ.get("VINDEX_CACHE_DIR")
    if override:
        return override
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return os.path.join(xdg, "vindex")
    local_appdata = os.environ.get("LOCALAPPDATA")
    if os.name == "nt" and local_appdata:
        return os.path.join(local_appdata, "vindex", "cache")
    return os.path.join(os.path.expanduser("~"), ".cache", "vindex")


CACHE_ROOT = _default_cache_root()

E5_MODELS = frozenset({"intfloat/multilingual-e5-base"})
RAW_TRANSFORMER_MODELS = frozenset({MURIL_MODEL_NAME})


class MurilWithoutOverrideError(ValueError):
    """Raised when loading MuRIL without explicitly allowing it."""


def _sanitize(name: str) -> str:
    return name.replace("/", "__")


def _cache_key(text: str, is_query: bool) -> str:
    h = hashlib.sha256()
    h.update(b"query\x00" if is_query else b"passage\x00")
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def _cache_path(
    encoder_name: str, revision: str | None, text: str, is_query: bool
) -> tuple[str, str]:
    # The revision is part of the directory name, so different weights
    # under the same repo name land in sibling directories.
    key = _cache_key(text, is_query)
    name_part = _sanitize(encoder_name)
    if revision:
        name_part = f"{name_part}@{revision}"
    d = os.path.join(CACHE_ROOT, name_part, key[:2])
    return os.path.join(d, key + ".npy"), d


class Encoder:
    """Loads one sentence encoder and encodes text, with an on-disk cache.

    Provides a uniform interface over sentence-transformers models and raw
    Hugging Face transformer models (MuRIL is not a sentence-transformers
    model and is mean-pooled manually).

    Args:
        model_name: Hugging Face model id. Defaults to
            ``calibration.DEFAULT_ENCODER``.
        allow_muril: Permit loading ``google/muril-base-cased``.

    Raises:
        MurilWithoutOverrideError: If MuRIL is requested without
            ``allow_muril=True``.
    """

    def __init__(self, model_name: str = "", allow_muril: bool = False) -> None:
        from vindex.calibration import DEFAULT_ENCODER

        model_name = model_name or DEFAULT_ENCODER
        if model_name == MURIL_MODEL_NAME and not allow_muril:
            raise MurilWithoutOverrideError(MURIL_WARNING)

        self.model_name = model_name
        self._st_model: Any = None
        self._hf_model: Any = None
        self._hf_tokenizer: Any = None
        self.revision: str | None = None
        self.cache_hits = 0
        self.cache_misses = 0

    def load(self) -> Encoder:
        if self.model_name in RAW_TRANSFORMER_MODELS:
            from transformers import AutoModel, AutoTokenizer

            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._hf_model = AutoModel.from_pretrained(self.model_name)
            self._hf_model.eval()
            self.revision = getattr(self._hf_model.config, "_commit_hash", None)
        else:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.model_name)
            self.revision = self._resolve_st_revision(self._st_model)
        return self

    @staticmethod
    def _resolve_st_revision(st_model: Any) -> str | None:
        # The transformer sub-module of a SentenceTransformer exposes the
        # same resolved config._commit_hash a raw AutoModel would. If the
        # wrapper shape differs, fall back to no revision (cache keys on
        # the name alone) rather than raising.
        try:
            for module in st_model.modules():
                auto_model = getattr(module, "auto_model", None)
                commit_hash = getattr(getattr(auto_model, "config", None), "_commit_hash", None)
                if commit_hash:
                    return str(commit_hash)
        except Exception:
            return None
        return None

    def _prep(self, text: str, is_query: bool) -> str:
        if self.model_name in E5_MODELS:
            prefix = "query: " if is_query else "passage: "
            return f"{prefix}{text}"
        return text

    def _mean_pool(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        import torch

        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def _encode_uncached(self, text: str, is_query: bool) -> np.ndarray[Any, Any]:
        prepped = self._prep(text, is_query)
        if self.model_name in RAW_TRANSFORMER_MODELS:
            import torch

            with torch.no_grad():
                enc = self._hf_tokenizer(
                    [prepped], padding=True, truncation=True, max_length=256, return_tensors="pt"
                )
                out = self._hf_model(**enc)
                pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                return pooled.cpu().numpy()[0]  # type: ignore[no-any-return]
        emb = self._st_model.encode(
            [prepped], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return emb[0]  # type: ignore[no-any-return]

    def encode(self, text: str, is_query: bool = True) -> np.ndarray[Any, Any]:
        """Encode text, using the on-disk cache when available.

        Args:
            text: Text to encode.
            is_query: Encode as a query rather than a passage. Only e5
                models embed the two roles differently, but the role is
                always part of the cache key.

        Returns:
            A unit-normalized embedding vector.
        """
        import numpy as np

        path, d = _cache_path(self.model_name, self.revision, text, is_query)
        if os.path.exists(path):
            self.cache_hits += 1
            result: np.ndarray[Any, Any] = np.load(path)
            return result

        self.cache_misses += 1
        vec = self._encode_uncached(text, is_query)
        try:
            os.makedirs(d, exist_ok=True)
            tmp_base = path[:-4] + ".tmp"
            np.save(tmp_base, vec)
            os.replace(tmp_base + ".npy", path)
        except OSError:
            # Cache directory not writable: return the result uncached.
            pass
        return vec

    def cache_stats(self) -> dict[str, float]:
        total = self.cache_hits + self.cache_misses
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }


def cosine_similarity(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
    import numpy as np

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)
