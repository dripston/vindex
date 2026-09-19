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
encoder_cache.py, same cache-key scheme
(<cache_root>/<sanitized_encoder_name>/<sha256(text)[:2]>/
<sha256(text)>.npy, is_query folded into the hash) so repeated calls
with the same (encoder, text, is_query) reuse a cached embedding --
nobody in this project has a GPU, and encoding is the slow part of
every run.

CACHE KEY INCLUDES RESOLVED MODEL REVISION (FIXED -- found by an
independent outside review): the cache key used to be derived only
from the encoder's mutable HuggingFace repo name (e.g.
"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"), not a
pinned revision/commit hash. `judge_model.py`'s judge models are
version-pinned and their exact `judge_model_id` recorded in every
result specifically so scores stay comparable across time -- the
embedding cache had no equivalent protection. If a HuggingFace repo
was updated in place under the same name (weights changed, no name
bump), a stale cached embedding from the old weights would be silently
reused instead of recomputed, and `calibration.CALIBRATION_TABLE`'s
thresholds (fit against the OLD weights) would then be compared against
embeddings from different, new weights with nothing surfacing the
mismatch.

Fixed: `Encoder.load()` now reads the resolved commit hash that
transformers/sentence-transformers already resolve internally
(`config._commit_hash`, exposed identically whether the model is
loaded as a raw `AutoModel` or wrapped inside a `SentenceTransformer`)
and folds it into the cache key, so a rebuild under the same repo name
with different weights gets a different cache key instead of a stale
hit. This costs no extra network call -- it reads a value the loader
already resolved locally, it does not query the Hub separately. If the
hash cannot be determined (e.g. an unusual model wrapper that does not
expose `_commit_hash`), caching degrades to keying on the name alone
(the old behavior) rather than crashing -- so revision pinning is
best-effort, not a guarantee for every possible model class. If you
need a hard reproducibility guarantee regardless, set
`$VINDEX_CACHE_DIR` to a fresh directory whenever you pin a specific
model revision yourself.

CACHE_ROOT LOCATION (fixed after a real bug -- see _default_cache_root's
docstring): NOT a repo-relative path. It used to be computed as three
dirname() hops from __file__, which resolves inside the Python install
directory for anyone who `pip install`'d this package -- and
os.makedirs() there raises PermissionError on any normal, non-root
install. Now a real user cache directory: $VINDEX_CACHE_DIR if set,
else the platform's standard cache location. A cache-write failure
(e.g. read-only filesystem) degrades to no caching rather than
crashing encode() -- see encode()'s try/except.
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Any

from vindex.calibration import MURIL_MODEL_NAME, MURIL_WARNING

if TYPE_CHECKING:
    import numpy as np

def _default_cache_root() -> str:
    """Platform-appropriate user cache directory for vindex's encoder
    cache, honoring VINDEX_CACHE_DIR as an explicit override.

    PACKAGING FIX (a real bug, not a style choice): this used to be
    computed as three dirname() hops from __file__, resolving to
    <repo_root>/encoder_cache/ -- which only exists in a git checkout.
    In an installed package, that path resolves inside the Python
    installation directory (e.g. site-packages' grandparent), and
    encode()'s os.makedirs() call there raises PermissionError on any
    normal (non-root/non-admin) install. Fixed to use a real user cache
    directory: $VINDEX_CACHE_DIR if set, else $XDG_CACHE_HOME/vindex on
    Linux/Mac, else ~/.cache/vindex, else (Windows, no XDG_CACHE_HOME)
    %LOCALAPPDATA%/vindex/cache.
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
    # revision is folded into the directory name (not the file's hash)
    # so different weights under the same repo name land in sibling
    # directories rather than colliding -- see this module's docstring
    # ("CACHE KEY INCLUDES RESOLVED MODEL REVISION").
    key = _cache_key(text, is_query)
    name_part = _sanitize(encoder_name)
    if revision:
        name_part = f"{name_part}@{revision}"
    d = os.path.join(CACHE_ROOT, name_part, key[:2])
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
        # SentenceTransformer wraps one or more sub-modules; the
        # transformer sub-module exposes the same resolved
        # config._commit_hash a raw AutoModel would. Best-effort: if
        # this wrapper shape ever changes, fall back to no revision
        # (cache keys on name alone) rather than raising.
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
        """Encode `text`, using the on-disk cache when available.

        is_query distinguishes query vs passage encoding for models
        (e5) whose embedding depends on that role; for other models the
        embedding is the same either way, but is_query is still folded
        into the cache key unconditionally to stay correct for all
        encoders without special-casing (see encoder_cache.py's
        original docstring).
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
            # Cache directory not writable (e.g. a read-only filesystem,
            # or a permissions issue not fixed by VINDEX_CACHE_DIR) --
            # degrade to no caching rather than crash the whole encode
            # call. The result is still correct, just not persisted for
            # next time.
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
