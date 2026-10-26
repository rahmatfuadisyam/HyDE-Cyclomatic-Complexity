"""
v2 Langkah 2: Embed seluruh korpus dan simpan sebagai matriks numpy (pencarian
exact cosine / brute-force — tanpa ANN, sehingga hasil deterministik).
"""
import json
import os

import numpy as np

from common_v2 import load_corpus, embed, doc_text, get_model
from config import V2_EMB_FILE, V2_DIR, EMBEDDING_MODEL


def main():
    corpus = load_corpus()["functions"]
    texts = [doc_text(f) for f in corpus]
    print(f"[+] Embedding {len(texts)} fungsi dengan {EMBEDDING_MODEL}")
    emb = embed(texts)
    np.save(V2_EMB_FILE, emb)

    # Catat berapa dokumen yang terpotong oleh batas token model (untuk Threats to Validity)
    info = {"n": len(texts), "dim": int(emb.shape[1])}
    m = get_model()
    tok = getattr(m, "tokenizer", None)
    if tok is not None:
        max_len = int(getattr(m, "max_seq_length", 512))
        n_trunc = sum(len(tok(t, add_special_tokens=True)["input_ids"]) > max_len for t in texts)
        info.update({"max_seq_length": max_len, "n_truncated": int(n_trunc),
                     "pct_truncated": round(100 * n_trunc / len(texts), 2)})
    with open(os.path.join(V2_DIR, "index_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[✓] Saved {emb.shape} -> {V2_EMB_FILE}; info: {info}")


if __name__ == "__main__":
    main()
