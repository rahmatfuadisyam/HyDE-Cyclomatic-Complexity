"""
Verifikasi embedding korpus tersimpan (review 5–6). Gagal -> exit code 1 (menghentikan run_v3_eval.sh).

Pemeriksaan terhadap manifest referensi data/v2/embedding_reference.json (dirilis bersama data):
  (1) SHA-256 korpus, (2) SHA-256 urutan ID fungsi, (3) SHA-256 berkas embedding dan bentuknya,
  (4) model dan revisi encoder di config sama dengan referensi,
  (5) embedding ulang dengan checkpoint terkunci cocok dengan baris tersimpan (kosinus minimum > 0,999).
Hasil ditulis ke data/v2/embedding_check.json dengan cakupan "full" (--all) atau "sample".

    python verify_embeddings.py --all          # audit rilis: seluruh korpus
    python verify_embeddings.py                # pemeriksaan cepat: sampel 64 fungsi (seed 42)
    python verify_embeddings.py --hash-only    # hanya (1)-(4), tanpa memuat model
    python verify_embeddings.py --write-reference   # membuat manifest referensi (hanya oleh peneliti)
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

from config import EMBEDDING_MODEL, EMBEDDING_REVISION, V2_CORPUS_FILE, V2_DIR, V2_EMB_FILE

REF = os.path.join(V2_DIR, "embedding_reference.json")
OUT = os.path.join(V2_DIR, "embedding_check.json")
COS_MIN = 0.999


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def current_hashes():
    corpus = json.load(open(V2_CORPUS_FILE))["functions"]
    E = np.load(V2_EMB_FILE)
    ids = [f["id"] for f in corpus]
    return corpus, E, {"model": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION,
                       "corpus_sha256": sha256_file(V2_CORPUS_FILE),
                       "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                       "embeddings_sha256": sha256_file(V2_EMB_FILE),
                       "n_corpus": len(corpus), "emb_shape": list(E.shape)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--hash-only", action="store_true")
    ap.add_argument("--write-reference", action="store_true")
    a = ap.parse_args()
    corpus, E, cur = current_hashes()
    if a.write_reference:
        json.dump(cur, open(REF, "w"), indent=2)
        print(f"[✓] Manifest referensi ditulis: {REF}")
        return
    if not os.path.exists(REF):
        sys.exit(f"[✗] Manifest referensi tidak ada: {REF}")
    ref = json.load(open(REF))
    checks = {k: cur[k] == ref.get(k) for k in ("model", "revision", "corpus_sha256", "ids_sha256",
                                                "embeddings_sha256", "n_corpus", "emb_shape")}
    checks["rows_match_corpus"] = E.shape[0] == len(corpus)
    out = {"reference": REF, "scope": "hash-only" if a.hash_only else ("full" if a.all else "sample"), **cur,
           "checks": checks}
    if not a.hash_only:
        from common_v2 import embed, doc_text
        idx = (np.arange(len(corpus)) if a.all else
               np.sort(np.random.default_rng(42).choice(len(corpus), min(a.n, len(corpus)), replace=False)))
        re_emb = embed([doc_text(corpus[i]) for i in idx])
        cos = np.sum(re_emb * E[idx], axis=1)
        out.update({"n_checked": int(len(idx)), "min_cosine": float(cos.min()), "mean_cosine": float(cos.mean()),
                    "max_abs_diff": float(np.abs(re_emb - E[idx]).max())})
        checks["reembedding_matches"] = bool(cos.min() > COS_MIN)
    out["pass"] = all(checks.values())
    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps(out, indent=2))
    if not out["pass"]:
        failed = [k for k, v in checks.items() if not v]
        print(f"[✗] VERIFIKASI GAGAL: {failed}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[✓] Verifikasi lulus ({out['scope']})")


if __name__ == "__main__":
    main()
