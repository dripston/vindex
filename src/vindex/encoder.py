"""
Encoder loading, encoding, and disk caching for calibrated_similarity
(Milestone 3.2, 3.4).

sentence-transformers/transformers/torch are NOT core vindex
dependencies -- they are multi-GB installs and Milestone 1/2 shipped
without them on purpose. They are required only if you actually call
calibrated_similarity(); install them via the "similarity" extra
(`pip install vindex[similarity]`). Everything in this module lazy-
imports them so `import vindex` alone never requires them.

DEFAULT ENCODER (Milestone 3.2): sentence-transformers/paraphrase-
multilingual-mpnet-base-v2 -- see calibration.DEFAULT_ENCODER and its
own module docstring for why (it is the best all-around performer in
the calibration table, not the fastest or smallest).

MuRIL WARNING (Milestone 3.2): passing google/muril-base-cased raises
loudly at load time -- not silently degraded output. See
calibration.MURIL_WARNING for the full HindiWiC-cited explanation. Pass
allow_muril=True to load it anyway (e.g. to reproduce or extend the
calibration experiment itself).

CACHING (Milestone 3.4): ported from experiments/scripts/
encoder_cache.py, unchanged in on-disk layout and cache-key scheme
(<repo_root>/encoder_cache/<sanitized_encoder_name>/<sha256(text)[:2]>/
<sha256(text)>.npy, is_query folded into the hash) so a cache warmed by
prior runs is reused, not invalidated, by calling this module --
nobody in this project has a GPU, and encoding is the slow part of
every run.

Note: encoder_cache.py's own CACHE_ROOT computes to
experiments/encoder_cache/ (its own dirname math, one level short of
repo root), but that directory does not exist on disk -- the actually-
populated cache lives at <repo_root>/encoder_cache/, which
encoder_comparison.py's real run wrote to directly (it does not import
CachedEncoder at all). This module targets the real, populated
location -- repo-root encoder_cache/ -- not encoder_cache.py's own
(apparently unused) path formula.
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Any

from vindex.calibration import MURIL_MODEL_NAME, MURIL_WARNING

if TYPE_CHECKING:
    import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_ROOT = os.path.join(REPO_ROOT, "encoder_cache")

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


def _cache_path(encoder_name: str, text: str, is_query: bool) -> tuple[str, str]:
    key = _cache_key(text, is_query)
    d = os.path.join(CACHE_ROOT, _sanitize(encoder_name), key[:2])
    return os.path.join(d, key + ".npy"), d


class Encoder:
    """Loads one sentence encoder and encodes text, with an on-disk cache.

    Uniform interface across sentence-transformers models and raw
    HuggingFace transformer models (MuRIL is not a sentence-transformers
    model and needs manual mean pooling) -- ported from
    experiments/scripts/encoder_comparison.py's EncoderWrapper.
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
        self.cache_hits = 0
        self.cache_misses = 0

    def load(self) -> Encoder:
        if self.model_name in RAW_TRANSFORMER_MODELS:
            from transformers import AutoModel, AutoTokenizer

            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._hf_model = AutoModel.from_pretrained(self.model_name)
            self._hf_model.eval()
        else:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.model_name)
        return self

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

    def _encode_uncached(self, text: str, is_query: bool) -> np.ndarray:
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

    def encode(self, text: str, is_query: bool = True) -> np.ndarray:
        """Encode `text`, using the on-disk cache when available.

        is_query distinguishes query vs passage encoding for models
        (e5) whose embedding depends on that role; for other models the
        embedding is the same either way, but is_query is still folded
        into the cache key unconditionally to stay correct for all
        encoders without special-casing (see encoder_cache.py's
        original docstring).
        """
        import numpy as np

        path, d = _cache_path(self.model_name, text, is_query)
        if os.path.exists(path):
            self.cache_hits += 1
            result: np.ndarray = np.load(path)
            return result

        self.cache_misses += 1
        vec = self._encode_uncached(text, is_query)
        os.makedirs(d, exist_ok=True)
        tmp_base = path[:-4] + ".tmp"
        np.save(tmp_base, vec)
        os.replace(tmp_base + ".npy", path)
        return vec

    def cache_stats(self) -> dict[str, float]:
        total = self.cache_hits + self.cache_misses
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    import numpy as np

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)
