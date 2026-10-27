"""
v3 Langkah 4c (revisi review kedua): representasi FUNGSI UTUH pada korpus yang SAMA (1.005 unit).

Tujuan: memisahkan pengaruh pemotongan encoder dari perubahan komposisi korpus. Korpus, relevansi,
query, dan hipotesis tidak berubah; yang berubah hanya cakupan teks yang dibaca encoder.

Representasi (encoder SAMA, BAAI/bge-small-en-v1.5):
  first : seperti 04 — encoder membaca <= 512 token pertama (dokumen dan hipotesis)
  mean  : teks dipotong per CHUNK_TOKENS token; embedding = rerata ternormalisasi seluruh potongan
  max   : skor dokumen = kemiripan maksimum antara vektor query/hipotesis dan potongan dokumen
          (vektor hipotesis = rerata potongannya)
Encoder pembanding opsional (EXTRA_ENCODERS, mis. encoder kode berkonteks panjang) dilaporkan terpisah
karena arsitekturnya ikut berubah. Lewati dengan MAHYDE_SKIP_EXTRA=1.

Mode per representasi: dense; per LLM x prompt: hyde (tunggal, rata-rata atas run), hyde_avg (q + semua h).
"""
import json
import os
import re
from importlib import import_module

import numpy as np

from common_v2 import load_corpus, load_embeddings, get_model
from config_v3 import (QUERIES_V2, K, PROMPTS, N_SAMPLES, BGE_QUERY_INSTRUCTION, EMBEDDING_MODEL,
                       V3_HYPO_CACHE, V3_RESULTS_FILE, V3_FULLFUNC_FILE, CHUNK_TOKENS, EXTRA_ENCODERS)

ev4 = import_module("04_evaluate_v3")
Evaluator, unit, mean_metrics = ev4.Evaluator, ev4.unit, ev4.mean_metrics


def spans(text, tok, n):
    """Potong teks menjadi segmen berisi <= n token (offset karakter dari tokenizer bila ada)."""
    if tok is not None:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
    else:  # smoke test: perkiraan token regex
        offs = [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]
    if not offs:
        return [text]
    out = []
    for i in range(0, len(offs), n):
        a, b = offs[i][0], offs[min(i + n, len(offs)) - 1][1]
        out.append(text[a:b])
    return out


def encode(model, texts, bs=32):
    return np.asarray(model.encode(list(texts), normalize_embeddings=True, batch_size=bs,
                                   show_progress_bar=len(texts) > 200), dtype=np.float32)


def chunked(model, tok, texts, n):
    """-> (list of chunk-matrices per text, mean vector per text)."""
    segs = [spans(t, tok, n) for t in texts]
    flat = [s for ss in segs for s in ss]
    emb = encode(model, flat)
    mats, means, i = [], [], 0
    for ss in segs:
        m = emb[i:i + len(ss)]; i += len(ss)
        mats.append(m); means.append(unit(m.mean(0)))
    return mats, np.vstack(means), [len(ss) for ss in segs]


def evaluate(ev, score_fn, q_vecs, H, llms, length=None):
    """score_fn(vec) -> skor per dokumen. H[(L,qid,p)] = list vektor hipotesis (run 0..N-1)."""
    per_query = []
    for qi, (qid, _) in enumerate(QUERIES_V2):
        qv = q_vecs[qi]
        row = {"qid": qid, "modes": {"dense": ev.metrics(ev.rank(score_fn(qv)))}}
        for bn, bv in (length or {}).items():
            row["modes"][bn] = ev.metrics(ev.rank(np.asarray(bv, dtype=float)))
        for L in llms:
            for p in PROMPTS:
                hs = H[(L, qid, p)]
                runs = [ev.metrics(ev.rank(score_fn(h))) for h in hs]
                row["modes"][f"{L}:hyde_{p}"] = mean_metrics(runs)
                row.setdefault("runs", {})[f"{L}:hyde_{p}"] = [r["p_at_k"] for r in runs]
                row["modes"][f"{L}:hyde_{p}_avg"] = ev.metrics(ev.rank(score_fn(unit(np.mean(np.vstack([qv] + hs), 0)))))
        per_query.append(row)
    modes = list(per_query[0]["modes"])
    return per_query, {m: mean_metrics([r["modes"][m] for r in per_query]) for m in modes}


def manifest(*a, **k):
    try:
        return _manifest(*a, **k)
    except Exception as e:  # noqa: BLE001  (mis. embedder palsu pada smoke test)
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def _manifest(model, name, docs, queries, hyps, prefix_q, chunking):
    """Manifest encoder yang dapat diaudit (review 3)."""
    import sentence_transformers as st_
    mods = [type(mm).__name__ for mm in model]
    pool = next((mm for mm in model if type(mm).__name__ == "Pooling"), None)
    rev = None
    try:
        from huggingface_hub import snapshot_download
        rev = os.path.basename(snapshot_download(name, local_files_only=True))
    except Exception as e:  # noqa: BLE001
        rev = f"unknown ({type(e).__name__})"
    tok = getattr(model, "tokenizer", None)
    msl = int(model.max_seq_length)

    def lens(xs):
        if tok is None:
            return None
        L_ = np.array([len(tok(x, add_special_tokens=True)["input_ids"]) for x in xs])
        return {"max": int(L_.max()), "median": float(np.median(L_)), "n_over_window": int((L_ > msl).sum()), "n": len(xs)}

    return {"model": name, "revision": rev, "sentence_transformers": st_.__version__, "modules": mods,
            "pooling": pool.get_pooling_mode_str() if pool is not None else None,
            "normalize": "L2 (normalize_embeddings=True)", "max_seq_length": msl,
            "prefix_query": prefix_q, "prefix_document": "", "prefix_hypothesis": "",
            "tokens_documents": lens(docs), "tokens_queries": lens(queries), "tokens_hypotheses": lens(hyps),
            "chunking": chunking}


def loaded_code_revision(m):
    """Commit kode eksternal yang benar-benar dimuat (dari nama modul transformers_modules.<org>.<repo>.<sha>...)."""
    mod = type(m[0].auto_model).__module__
    return next((x for x in mod.split(".") if len(x) == 40 and all(c in "0123456789abcdef" for c in x)), None)


def load_extra(enc):
    """Review 6: muat encoder pembanding dengan checkpoint DAN kode eksternal terkunci; gagal bila tidak cocok."""
    from sentence_transformers import SentenceTransformer, models
    code_rev = enc.get("code_revision")
    if enc.get("code_repo") and not code_rev:
        raise SystemExit("[✗] code_revision encoder pembanding belum dikunci: jalankan python pin_encoders.py")
    margs = {"trust_remote_code": enc.get("trust_remote_code", False), "revision": enc.get("revision")}
    if code_rev:
        margs["code_revision"] = code_rev
    tr = models.Transformer(enc["name"], max_seq_length=enc.get("max_seq_length"), model_args=margs,
                            tokenizer_args={"trust_remote_code": enc.get("trust_remote_code", False),
                                            "revision": enc.get("revision")})
    pool = models.Pooling(tr.get_word_embedding_dimension(), pooling_mode=enc.get("pooling", "mean"))
    m = SentenceTransformer(modules=[tr, pool])
    loaded = loaded_code_revision(m)
    if code_rev and loaded != code_rev:
        raise SystemExit(f"[✗] kode eksternal yang dimuat ({loaded}) != yang dikunci ({code_rev})")
    return m, loaded


def strata_eval(ev_full, corpus, idx, fn, q_vecs, H, llms, length):
    ev_s = Evaluator([corpus[i] for i in idx])
    pq, sm = evaluate(ev_s, lambda v: fn(v)[idx], q_vecs, H, llms, {k: np.asarray(v)[idx] for k, v in length.items()})
    return {"n_docs": int(len(idx)), "n_relevant": int(ev_s.rel.sum()), "per_query": pq, "summary": sm}


def main():
    corpus = load_corpus()["functions"]
    ev = Evaluator(corpus)
    res = json.load(open(V3_RESULTS_FILE))
    llms = [m["id"] for m in res["metadata"]["llms"]]
    cache = json.load(open(V3_HYPO_CACHE))
    hkeys = {(L, qid, p): [f"{L}|{qid}|{p}|t07|{r}" for r in range(N_SAMPLES)]
             for L in llms for qid, _ in QUERIES_V2 for p in PROMPTS}
    hyp_texts = [cache[k]["code"] for ks in hkeys.values() for k in ks]
    docs = [f["code"] for f in corpus]
    queries = [BGE_QUERY_INSTRUCTION + q for _, q in QUERIES_V2]
    out = {"metadata": {"encoder": EMBEDDING_MODEL, "chunk_tokens": CHUNK_TOKENS, "llms": llms,
                        "n_corpus": len(corpus), "n_relevant": int(ev.rel.sum())}, "representations": {}}

    model = get_model()
    tok = getattr(model, "tokenizer", None)
    E_first = load_embeddings()
    q_vecs = encode(model, queries)
    # hipotesis: first (dibaca terpotong) dan mean (seluruh potongan)
    h_first = encode(model, hyp_texts)
    _, h_mean, h_nchunks = chunked(model, tok, hyp_texts, CHUNK_TOKENS)
    d_mats, d_mean, d_nchunks = chunked(model, tok, docs, CHUNK_TOKENS)
    print(f"[+] Potongan dokumen: {sum(d_nchunks)} (maks {max(d_nchunks)} per fungsi); "
          f"hipotesis: {sum(h_nchunks)}")

    def H_of(vecs):
        it = iter(vecs)
        return {key: [next(it) for _ in ks] for key, ks in hkeys.items()}

    cov = {"n_docs_multi_chunk": int(sum(c > 1 for c in d_nchunks)),
           "n_relevant_multi_chunk": int(sum(c > 1 for c, r in zip(d_nchunks, ev.rel) if r)),
           "n_hyp_multi_chunk": int(sum(c > 1 for c in h_nchunks)), "n_hyp": len(h_nchunks)}
    C_all = np.vstack(d_mats)
    starts = np.concatenate([[0], np.cumsum(d_nchunks)[:-1]]).astype(int)
    reps = {
        "first": (lambda v: E_first @ v, H_of(h_first)),
        "mean": (lambda v: d_mean @ v, H_of(h_mean)),
        "max": (lambda v: np.maximum.reduceat(C_all @ v, starts), H_of(h_mean)),
    }
    tok_len = np.array([len(tok(d, add_special_tokens=True)["input_ids"]) for d in docs]) if tok is not None \
        else np.array([len(re.findall(r"\w+|[^\w\s]", d)) for d in docs])
    dn = np.array(d_nchunks)
    base = {"length": tok_len, "sloc": np.array([f.get("sloc") or 0 for f in corpus], dtype=float)}
    strata_idx = {"1 potongan": np.where(dn == 1)[0], "≥ 2 potongan": np.where(dn >= 2)[0]}
    out["metadata"]["n_chunks_docs"] = d_nchunks
    out["manifest"] = {"bge": manifest(model, EMBEDDING_MODEL, docs, queries, hyp_texts, BGE_QUERY_INSTRUCTION,
                                       {"first": "dokumen & hipotesis dibaca <= max_seq_length token pertama",
                                        "mean": f"potongan berurutan {CHUNK_TOKENS} token (offset tokenizer), "
                                                "rerata embedding potongan lalu dinormalisasi",
                                        "max": "skor = kosinus maksimum vektor query/hipotesis terhadap potongan dokumen; "
                                               "hipotesis = rerata potongan"})}
    out["strata"] = {}
    for name, (fn, H) in reps.items():
        pq, sm = evaluate(ev, fn, q_vecs, H, llms, base)
        out["representations"][name] = {"encoder": EMBEDDING_MODEL, "per_query": pq, "summary": sm, "coverage": cov}
        out["strata"][name] = {k: strata_eval(ev, corpus, ix, fn, q_vecs, H, llms, base) for k, ix in strata_idx.items()}
        print(f"  [{name}] dense P@10={sm['dense']['p_at_k']:.3f}  " + "  ".join(
            f"{L}:fs_avg={sm[f'{L}:hyde_fs_avg']['p_at_k']:.3f}" for L in llms))

    if os.environ.get("MAHYDE_SKIP_EXTRA") != "1" and os.environ.get("MAHYDE_FAKE_EMB") != "1":
        from sentence_transformers import SentenceTransformer
        for enc in EXTRA_ENCODERS:
            try:
                m, code_loaded = load_extra(enc)
                m.max_seq_length = enc.get("max_seq_length", m.max_seq_length)
                t = getattr(m, "tokenizer", None)
                over = int(sum(len(t(d, add_special_tokens=True)["input_ids"]) > m.max_seq_length for d in docs)) if t else None
                bs_ = enc.get("batch_size", 1)   # konteks 8192: batch besar -> buffer atensi > 17 GiB
                order = np.argsort([len(d) for d in docs])
                Ed = np.empty((len(docs), m.get_sentence_embedding_dimension()), dtype=np.float32)
                Ed[order] = encode(m, [docs[i] for i in order], bs=bs_)
                qv = encode(m, [enc.get("query_prefix", "") + q for _, q in QUERIES_V2])
                Hh = H_of(encode(m, hyp_texts, bs=bs_))
                pq, sm = evaluate(ev, lambda v: Ed @ v, qv, Hh, llms, base)
                out["representations"][enc["id"]] = {"encoder": enc["name"], "max_seq_length": m.max_seq_length,
                                                     "n_docs_over_window": over, "per_query": pq, "summary": sm}
                out["strata"][enc["id"]] = {k: strata_eval(ev, corpus, ix, lambda v: Ed @ v, qv, Hh, llms, base)
                                            for k, ix in strata_idx.items()}
                out["manifest"][enc["id"]] = manifest(m, enc["name"], docs,
                                                      [enc.get("query_prefix", "") + q for _, q in QUERIES_V2],
                                                      hyp_texts, enc.get("query_prefix", ""),
                                                      "tanpa pemotongan; seluruh teks dalam jendela encoder")
                out["manifest"][enc["id"]].update({"code_repo": enc.get("code_repo"), "code_revision_pinned": enc.get("code_revision"),
                                                   "code_revision_loaded": code_loaded})
                print(f"  [{enc['id']}] dense P@10={sm['dense']['p_at_k']:.3f}")
            except Exception as e:
                out["representations"][enc["id"]] = {"encoder": enc["name"], "error": str(e)[:300]}
                print(f"  [!] {enc['id']} gagal: {str(e)[:200]}")
    json.dump(out, open(V3_FULLFUNC_FILE, "w"), indent=1)
    print(f"[✓] Saved {V3_FULLFUNC_FILE}")


if __name__ == "__main__":
    main()
