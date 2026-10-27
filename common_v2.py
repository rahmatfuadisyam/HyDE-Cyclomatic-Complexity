"""Utilitas bersama eksperimen v2: embedding, loading korpus, pembersihan kode LLM."""
import json
import os
import re

import numpy as np

from config import EMBEDDING_MODEL, EMBEDDING_REVISION, V2_CORPUS_FILE, V2_EMB_FILE

_MODEL = None


class _HashEmbedder:
    """HANYA untuk smoke test pipeline (MAHYDE_FAKE_EMB=1). Jangan dipakai untuk hasil."""
    dim = 384

    def encode(self, texts, normalize_embeddings=True, **kw):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[A-Za-z_]+", t.lower()):
                out[i, hash(tok) % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1
        return out / n


def get_model():
    global _MODEL
    if _MODEL is None:
        if os.environ.get("MAHYDE_FAKE_EMB") == "1":
            print("[!] SMOKE TEST: memakai hashing embedder palsu")
            _MODEL = _HashEmbedder()
        else:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION)  # model + tokenizer pada revisi tetap
    return _MODEL


def embed(texts, batch_size=32):
    m = get_model()
    return np.asarray(m.encode(list(texts), normalize_embeddings=True,
                               batch_size=batch_size, show_progress_bar=len(texts) > 64),
                      dtype=np.float32)


def load_corpus():
    with open(V2_CORPUS_FILE) as f:
        return json.load(f)


def load_embeddings():
    return np.load(V2_EMB_FILE)


def doc_text(fn):
    """Teks yang di-embed untuk satu fungsi: kode sumbernya saja (tanpa header path)."""
    return fn["code"]


_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n(.*?)\n```", re.S | re.M)


def clean_code(text):
    """Buang pagar markdown bila LLM tetap mengeluarkannya."""
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()
