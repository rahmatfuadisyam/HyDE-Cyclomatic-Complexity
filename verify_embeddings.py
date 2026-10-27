"""
Review 5: verifikasi bahwa embedding tersimpan (data/v2/corpus_emb.npy) cocok dengan korpus dan checkpoint.

Memeriksa: (1) hash SHA-256 korpus dan urutan ID fungsi; (2) bentuk matriks embedding; (3) embedding ulang
sampel fungsi dengan BAAI/bge-small-en-v1.5 pada revisi terkunci (config.EMBEDDING_REVISION) dan
membandingkannya dengan baris tersimpan (kosinus minimum). Hasil ditulis ke data/v2/embedding_check.json.

    python verify_embeddings.py            # sampel 64 fungsi (acak, seed 42)
    python verify_embeddings.py --all      # seluruh korpus
"""
import argparse
import hashlib
import json
import os

import numpy as np

from common_v2 import load_corpus, load_embeddings, embed, doc_text
from config import EMBEDDING_MODEL, EMBEDDING_REVISION, V2_CORPUS_FILE, V2_DIR


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    corpus = load_corpus()["functions"]
    E = load_embeddings()
    ids = [f["id"] for f in corpus]
    idx = np.arange(len(corpus)) if a.all else np.sort(np.random.default_rng(42).choice(len(corpus), min(a.n, len(corpus)), replace=False))
    re_emb = embed([doc_text(corpus[i]) for i in idx])
    cos = np.sum(re_emb * E[idx], axis=1)
    out = {"model": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION,
           "corpus_sha256": sha256(V2_CORPUS_FILE), "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
           "n_corpus": len(corpus), "emb_shape": list(E.shape), "rows_match_corpus": E.shape[0] == len(corpus),
           "n_checked": int(len(idx)), "min_cosine": float(cos.min()), "mean_cosine": float(cos.mean()),
           "max_abs_diff": float(np.abs(re_emb - E[idx]).max()),
           "pass": bool(E.shape[0] == len(corpus) and cos.min() > 0.999)}
    json.dump(out, open(os.path.join(V2_DIR, "embedding_check.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("[✓] COCOK" if out["pass"] else "[✗] TIDAK COCOK — periksa revisi model/urutan korpus")


if __name__ == "__main__":
    main()
