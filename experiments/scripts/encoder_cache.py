"""
On-disk embedding cache, keyed by (encoder_name, is_query, text_hash).

No GPU on this machine -- encoding hundreds of text pairs through 5
encoders is the expensive part of every run. This cache lets scripts skip
re-encoding text they've already embedded, so a small change (e.g. one new
metric column) doesn't require re-running the full grid.

Cache layout: encoder_cache/<sanitized_encoder_name>/<sha256(text)[:2]>/<sha256(text)>.npy
  -- one .npy file per (encoder, is_query, text) triple. is_query is folded
  into the hash (query/passage prefixing changes the actual embedded string
  for e5 models, but for the OTHER encoders the same text with is_query
  True/False produces the same embedding since _prep() is a no-op for them
  -- to stay correct for ALL encoders without special-casing, is_query is
  included in the cache key unconditionally).

Usage:
    from encoder_cache import CachedEncoder
    enc = CachedEncoder(EncoderWrapper(model_name).load(), model_name)
    vec = enc.encode_one(text, is_query=True)   # -> np.ndarray, cached
"""
import os
import hashlib
import numpy as np

CACHE_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "encoder_cache")


def _sanitize(name):
    return name.replace("/", "__")


def _key(text, is_query):
    h = hashlib.sha256()
    h.update(b"query\x00" if is_query else b"passage\x00")
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def _path_for(encoder_name, text, is_query):
    key = _key(text, is_query)
    d = os.path.join(CACHE_ROOT, _sanitize(encoder_name), key[:2])
    return os.path.join(d, key + ".npy"), d


class CachedEncoder:
    """Wraps an already-.load()-ed EncoderWrapper (from encoder_comparison.py)
    with a disk cache. Does not modify EncoderWrapper; only calls .encode()
    on cache misses."""

    def __init__(self, wrapper, encoder_name):
        self.wrapper = wrapper
        self.encoder_name = encoder_name
        self.hits = 0
        self.misses = 0

    def encode_one(self, text, is_query=True):
        path, d = _path_for(self.encoder_name, text, is_query)
        if os.path.exists(path):
            self.hits += 1
            return np.load(path)

        self.misses += 1
        vec = self.wrapper.encode([text], is_query=is_query)[0]
        os.makedirs(d, exist_ok=True)
        # np.save appends ".npy" to whatever base name it's given, so give
        # it a tmp base without the extension and locate the real output.
        tmp_base = path[:-4] + ".tmp"  # strip ".npy", add ".tmp"
        np.save(tmp_base, vec)
        os.replace(tmp_base + ".npy", path)
        return vec

    def stats(self):
        total = self.hits + self.misses
        rate = self.hits / total if total else 0.0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": rate}


def cache_size_info():
    """Returns (n_files, total_bytes) currently cached, for reporting."""
    n, total = 0, 0
    if not os.path.isdir(CACHE_ROOT):
        return 0, 0
    for root, _, files in os.walk(CACHE_ROOT):
        for f in files:
            if f.endswith(".npy"):
                n += 1
                total += os.path.getsize(os.path.join(root, f))
    return n, total
