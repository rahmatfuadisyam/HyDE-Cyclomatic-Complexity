"""
v3 Langkah 4: Evaluasi offline multi-LLM (memakai cache v3, korpus & embedding v2).

Mode bersama (tanpa LLM): random, bm25, dense, dense_f, oracle
Mode per LLM  L (kunci "L:mode"):
  hyde_{p}         : 1 hipotesis per ranking, rata-rata metrik atas N_SAMPLES generasi (T=0.7)
  hyde_{p}_f       : idem + filter CC >= FILTER_CC
  hyde_{p}_avg     : formulasi asli HyDE (Gao et al.): mean[q, h_1..h_N]
  hyde_{p}_avg_f   : idem + filter
  hyde_{p}_t0      : 1 hipotesis deterministik (T=0)
  hyde_{p}_avg@n   : dose-response, mean[q, h_1..h_n] untuk n di DOSE_N
dengan p in {fs, zs}.
"""
import json
import math
import re
from itertools import combinations

import numpy as np
from rank_bm25 import BM25Okapi

from common_v2 import load_corpus, load_embeddings, embed
from config_v3 import (QUERIES_V2, K, RELEVANT_CC, FILTER_CC, RANDOM_SEED, BGE_QUERY_INSTRUCTION,
                       EMBEDDING_MODEL, PROMPTS, N_SAMPLES, DETERMINISTIC_RUN, DOSE_N,
                       V3_HYPO_CACHE, V3_MANIFEST, V3_RESULTS_FILE, enabled_llms, MAX_EMPTY_RATE)

GRADE = {"A": 0, "B": 0, "C": 1, "D": 2, "E": 3, "F": 4}
SHARED = ["random", "random_f", "bm25", "dense", "dense_f", "oracle"]


def tokenize(text):
    toks = []
    for w in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text):
        for part in re.split(r"_|(?<=[a-z])(?=[A-Z])", w):
            if len(part) > 1:
                toks.append(part.lower())
    return toks


class Evaluator:
    def __init__(self, corpus):
        self.cc = np.array([f["cc"] for f in corpus])
        self.gain = np.array([2 ** GRADE[f["rank"]] - 1 for f in corpus], dtype=float)
        self.rel = self.cc >= RELEVANT_CC
        self.mask_f = self.cc >= FILTER_CC
        ideal = np.sort(self.gain)[::-1][:K]
        self.idcg = float(np.sum(ideal / np.log2(np.arange(2, K + 2))))

    def metrics(self, order):
        top = order[:K]
        dcg = float(np.sum(self.gain[top] / np.log2(np.arange(2, len(top) + 2))))
        hits = np.where(self.rel[order])[0]
        return {"p_at_k": float(self.rel[top].mean()),
                "ndcg_at_k": dcg / self.idcg if self.idcg else 0.0,
                "mean_cc_at_k": float(self.cc[top].mean()),
                "max_cc_at_k": int(self.cc[top].max()),
                "rr": float(1.0 / (hits[0] + 1)) if len(hits) else 0.0}

    def rank(self, scores, filtered=False):
        s = scores.astype(float).copy()
        if filtered:
            s[~self.mask_f] = -np.inf
        order = np.argsort(-s, kind="stable")
        return order[: int(self.mask_f.sum())] if filtered else order


def exact_rr(N, R):
    """E[1/X], X = posisi relevan pertama pada permutasi seragam N item berisi R relevan (tanpa batas kedalaman).
    P(X = k) = C(N-k, R-1) / C(N, R), k = 1..N-R+1."""
    lc = lambda a, b: math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)
    tot = lc(N, R)
    return float(sum(math.exp(lc(N - k, R - 1) - tot) / k for k in range(1, N - R + 2))) if R else 0.0


def random_expectation(ev, filtered, rng, n_mc=20000):
    """Ekspektasi metrik peringkat acak. P@K, nDCG@K, MeanCC@K dan RR analitik (eksak); MaxCC via Monte Carlo."""
    idx = np.where(ev.mask_f)[0] if filtered else np.arange(len(ev.cc))
    disc = 1 / np.log2(np.arange(2, K + 2))
    out = {"p_at_k": float(ev.rel[idx].mean()),
           "ndcg_at_k": float(ev.gain[idx].mean() * disc.sum() / ev.idcg),
           "mean_cc_at_k": float(ev.cc[idx].mean())}
    mx, rr = [], []
    for _ in range(n_mc):
        o = idx[rng.permutation(len(idx))]
        mx.append(ev.cc[o[:K]].max())
        h = np.where(ev.rel[o])[0]
        rr.append(1 / (h[0] + 1) if len(h) else 0.0)
    out["max_cc_at_k"], out["rr_mc"] = float(np.mean(mx)), float(np.mean(rr))
    out["rr"] = exact_rr(len(idx), int(ev.rel[idx].sum()))
    return out


def subsets(n_total, n, cap=70, seed=0):
    """Semua kombinasi n dari n_total hipotesis (dibatasi `cap` subset acak bila lebih banyak)."""
    combs = list(combinations(range(n_total), n))
    if len(combs) > cap:
        r = np.random.default_rng(seed)
        combs = [combs[i] for i in sorted(r.choice(len(combs), cap, replace=False))]
    return combs


def token_lengths(corpus):
    """Panjang token tiap unit menurut tokenizer encoder (fallback: perkiraan regex)."""
    from common_v2 import get_model
    m = get_model()
    tok = getattr(m, "tokenizer", None)
    max_len = int(getattr(m, "max_seq_length", 512))
    if tok is None:
        return np.array([int(len(re.findall(r"\w+|[^\w\s]", f["code"])) * 1.15) for f in corpus]), max_len, "approx"
    return np.array([len(tok(f["code"], add_special_tokens=True)["input_ids"]) for f in corpus]), max_len, "tokenizer"


def mean_metrics(ms):
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


def unit(v):
    return v / np.linalg.norm(v)


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def keys_for(llm_id, qid, p):
    t07 = [f"{llm_id}|{qid}|{p}|t07|{r}" for r in range(N_SAMPLES)]
    t0 = [f"{llm_id}|{qid}|{p}|t0|0"] if DETERMINISTIC_RUN else []
    return t07, t0


def main():
    data = load_corpus()
    corpus = data["functions"]
    ids = [f["id"] for f in corpus]
    E = load_embeddings()
    ev = Evaluator(corpus)
    cache = json.load(open(V3_HYPO_CACHE))
    manifest = json.load(open(V3_MANIFEST))

    # LLM yang datanya lengkap & lolos gerbang kualitas
    llms, excluded, quality = [], [], {}
    for llm in enabled_llms():
        need = [k for qid, _ in QUERIES_V2 for p in PROMPTS for k in sum(keys_for(llm["id"], qid, p), [])]
        miss = [k for k in need if k not in cache]
        es = [cache[k] for k in need if k in cache]
        q = {"n": len(need), "missing": len(miss),
             "empty": sum(1 for e in es if e.get("empty") or not (e.get("code") or "").strip()),
             "parsed": sum(1 for e in es if e.get("cc_method", "parsed" if e.get("parse_ok") else None) == "parsed"),
             "trimmed": sum(1 for e in es if e.get("cc_method") == "trimmed"),
             "cc_none": sum(1 for e in es if e.get("cc") is None),
             "truncated_output": sum(1 for e in es if e.get("finish_reason") in ("length", "max_tokens")),
             "returned_models": sorted({str(e.get("returned_model")) for e in es})}
        q["empty_rate"] = q["empty"] / max(1, len(es))
        quality[llm["id"]] = q
        if miss:
            excluded.append({"id": llm["id"], "reason": f"{len(miss)} generasi belum ada"})
        elif q["empty_rate"] > MAX_EMPTY_RATE:
            excluded.append({"id": llm["id"], "reason": f"output kosong {q['empty']}/{len(es)} "
                                                        f"({q['empty_rate']:.0%}) > {MAX_EMPTY_RATE:.0%}"})
        else:
            llms.append(llm)
    for x in excluded:
        print(f"[!] {x['id']} DIKELUARKAN: {x['reason']}")
    if not llms:
        raise SystemExit("Tidak ada LLM dengan data lengkap.")
    print(f"[+] Korpus {len(corpus)} | relevan {int(ev.rel.sum())} | pool filter {int(ev.mask_f.sum())} "
          f"| K={K} | LLM: {[m['id'] for m in llms]}")

    # --- embedding semua query & hipotesis (sekali)
    q_emb = embed([BGE_QUERY_INSTRUCTION + q for _, q in QUERIES_V2])
    all_keys = [k for m in llms for qid, _ in QUERIES_V2 for p in PROMPTS for k in sum(keys_for(m["id"], qid, p), [])]
    H = dict(zip(all_keys, embed([cache[k]["code"] for k in all_keys])))

    bm25 = BM25Okapi([tokenize(f["code"]) for f in corpus])
    rng = np.random.default_rng(RANDOM_SEED)
    rand_ms = random_expectation(ev, False, rng)
    rand_f = random_expectation(ev, True, rng)

    # --- analisis pemotongan encoder: subkorpus tanpa unit yang melebihi jendela token
    tok_len, max_len, tok_src = token_lengths(corpus)
    trunc = tok_len > max_len
    keep = np.where(~trunc)[0]
    ev_u = Evaluator([corpus[i] for i in keep])
    E_u = E[keep]
    rand_u = random_expectation(ev_u, False, rng)
    trunc_info = {"source": tok_src, "max_seq_length": max_len, "n_truncated": int(trunc.sum()),
                  "n_relevant_truncated": int((trunc & ev.rel).sum()), "n_relevant": int(ev.rel.sum()),
                  "n_kept": int(len(keep)), "n_relevant_kept": int(ev_u.rel.sum()),
                  "median_tokens_relevant": float(np.median(tok_len[ev.rel])),
                  "median_tokens_nonrelevant": float(np.median(tok_len[~ev.rel]))}
    print(f"[+] Pemotongan encoder ({tok_src}): {trunc_info['n_truncated']} unit > {max_len} token; "
          f"relevan terpotong {trunc_info['n_relevant_truncated']}/{trunc_info['n_relevant']}")
    oracle = ev.metrics(np.argsort(-ev.cc, kind="stable"))

    sloc = np.array([f.get("sloc") or 0 for f in corpus], dtype=float)
    per_query, topk = [], {}
    for qi, (qid, q) in enumerate(QUERIES_V2):
        row = {"qid": qid, "query": q, "modes": {}, "runs": {}, "modes_untrunc": {}}
        qv = q_emb[qi]
        row["modes"]["random"] = rand_ms
        row["modes"]["random_f"] = rand_f
        row["modes_untrunc"]["random"] = rand_u
        row["modes_untrunc"]["dense"] = ev_u.metrics(ev_u.rank(E_u @ qv))
        o = ev.rank(np.asarray(bm25.get_scores(tokenize(q))))
        row["modes"]["bm25"] = ev.metrics(o); topk[f"{qid}|bm25"] = [ids[i] for i in o[:K]]
        for name, filt in (("dense", False), ("dense_f", True)):
            o = ev.rank(E @ qv, filt)
            row["modes"][name] = ev.metrics(o); topk[f"{qid}|{name}"] = [ids[i] for i in o[:K]]
        row["modes"]["oracle"] = oracle
        # baseline panjang (review 3): urutkan menurut jumlah token (tokenizer encoder), terpanjang dahulu
        row["modes"]["length"] = ev.metrics(ev.rank(tok_len.astype(float)))
        row["modes"]["length_f"] = ev.metrics(ev.rank(tok_len.astype(float), True))
        row["modes"]["sloc"] = ev.metrics(ev.rank(sloc))  # baris kode sumber (radon; tanpa komentar/docstring)

        for llm in llms:
            L = llm["id"]
            for p in PROMPTS:
                t07, t0 = keys_for(L, qid, p)
                hs = [H[k] for k in t07]
                for suffix, filt in (("", False), ("_f", True)):
                    runs = []
                    for r, hv in enumerate(hs):
                        o = ev.rank(E @ hv, filt)
                        runs.append(ev.metrics(o))
                        topk[f"{qid}|{L}:hyde_{p}{suffix}|{r}"] = [ids[i] for i in o[:K]]
                    row["modes"][f"{L}:hyde_{p}{suffix}"] = mean_metrics(runs)
                    row["runs"][f"{L}:hyde_{p}{suffix}"] = runs
                    v = unit(np.mean(np.vstack([qv] + hs), axis=0))
                    o = ev.rank(E @ v, filt)
                    row["modes"][f"{L}:hyde_{p}_avg{suffix}"] = ev.metrics(o)
                    topk[f"{qid}|{L}:hyde_{p}_avg{suffix}"] = [ids[i] for i in o[:K]]
                # ablasi review-2: query + SATU hipotesis (dirata-ratakan atas run) -> efek penyertaan query tanpa perataan
                row["modes"][f"{L}:hyde_{p}_q1"] = mean_metrics(
                    [ev.metrics(ev.rank(E @ unit(np.mean(np.vstack([qv, hv]), axis=0)))) for hv in hs])
                # ablasi review-2: bobot query TETAP 1/2 ketika n berubah: unit(0.5 q + 0.5 unit(mean h_1..h_n))
                # review 3: setiap titik n dirata-ratakan atas SEMUA kombinasi n hipotesis (bukan n pertama)
                for n in DOSE_N:
                    if n <= len(hs):
                        ms_fw, ms_avg, ms_nq = [], [], []
                        for sub in subsets(len(hs), n):
                            Hs = np.vstack([hs[i] for i in sub])
                            hb = unit(Hs.mean(0))
                            ms_fw.append(ev.metrics(ev.rank(E @ unit(0.5 * qv + 0.5 * hb))))
                            ms_avg.append(ev.metrics(ev.rank(E @ unit(np.vstack([qv[None], Hs]).mean(0)))))
                            ms_nq.append(ev.metrics(ev.rank(E @ hb)))
                        row["modes"][f"{L}:hyde_{p}_fw@{n}"] = mean_metrics(ms_fw)
                        row["modes"][f"{L}:hyde_{p}_avg@{n}"] = mean_metrics(ms_avg)
                        row["modes"][f"{L}:hyde_{p}_avgnq@{n}"] = mean_metrics(ms_nq)
                # ablasi: rata-rata hipotesis TANPA vektor query (Gao et al., Pers. 7)
                vnq = unit(np.mean(np.vstack(hs), axis=0))
                row["modes"][f"{L}:hyde_{p}_avgnq"] = ev.metrics(ev.rank(E @ vnq))
                if t0:
                    o = ev.rank(E @ H[t0[0]])
                    row["modes"][f"{L}:hyde_{p}_t0"] = ev.metrics(o)
                # subkorpus tanpa unit terpotong (sensitivitas)
                row["modes_untrunc"][f"{L}:hyde_{p}"] = mean_metrics([ev_u.metrics(ev_u.rank(E_u @ hv)) for hv in hs])
                row["modes_untrunc"][f"{L}:hyde_{p}_avg"] = ev_u.metrics(
                    ev_u.rank(E_u @ unit(np.mean(np.vstack([qv] + hs), axis=0))))
        per_query.append(row)

    modes = list(per_query[0]["modes"])
    summary = {m: mean_metrics([r["modes"][m] for r in per_query]) for m in modes}
    modes_u = list(per_query[0]["modes_untrunc"])
    summary_untrunc = {m: mean_metrics([r["modes_untrunc"][m] for r in per_query]) for m in modes_u}

    # --- statistik hipotesis per (LLM, prompt)
    hyp = {}
    for llm in llms:
        L = llm["id"]
        for p in PROMPTS:
            for samp in ("t07", "t0"):
                ks = [k for qid, _ in QUERIES_V2 for k in keys_for(L, qid, p)[0 if samp == "t07" else 1]]
                if not ks:
                    continue
                cc = [cache[k]["cc"] for k in ks if cache[k]["cc"] is not None]
                n_all = len(ks)
                hyp_extra = {
                    "pct_comply_all": float(np.mean([(cache[k]["cc"] or 0) >= RELEVANT_CC for k in ks]) * 100),
                    "pct_cc_none": float(np.mean([cache[k]["cc"] is None for k in ks]) * 100),
                    "pct_trimmed": float(np.mean([cache[k].get("cc_method") == "trimmed" for k in ks]) * 100),
                    "pct_truncated_any": float(np.mean([cache[k].get("finish_reason") in ("length", "max_tokens")
                                                        for k in ks]) * 100),
                    "median_chars": float(np.median([len(cache[k]["code"]) for k in ks])),
                    "pct_comply_parsed_only": float(np.mean([c >= RELEVANT_CC for c, k in
                                                              ((cache[k]["cc"], k) for k in ks)
                                                              if c is not None and cache[k].get("cc_method") == "parsed"]) * 100)
                    if any(cache[k].get("cc_method") == "parsed" for k in ks) else None,
                }
                hyp[f"{L}:{p}:{samp}"] = {**hyp_extra,
                    "n": len(ks), "n_parsed": len(cc),
                    "cc_mean": float(np.mean(cc)) if cc else None,
                    "cc_median": float(np.median(cc)) if cc else None,
                    "cc_sd": float(np.std(cc, ddof=1)) if len(cc) > 1 else None,
                    "cc_min": int(min(cc)) if cc else None, "cc_max": int(max(cc)) if cc else None,
                    "pct_cc_ge_rel": float(np.mean([c >= RELEVANT_CC for c in cc]) * 100) if cc else None,
                    "pct_truncated_output": float(np.mean([cache[k].get("finish_reason") == "length" for k in ks]) * 100),
                    "mean_completion_tokens": float(np.mean([(cache[k].get("usage") or {}).get("completion") or 0 for k in ks])),
                }

    # --- stabilitas antar-generasi & overlap dengan dense_f
    stability, overlap = {}, {}
    for llm in llms:
        L = llm["id"]
        for p in PROMPTS:
            stability[f"{L}:{p}"] = float(np.mean([
                jacc(topk[f"{qid}|{L}:hyde_{p}|{a}"], topk[f"{qid}|{L}:hyde_{p}|{b}"])
                for qid, _ in QUERIES_V2 for a in range(N_SAMPLES) for b in range(a + 1, N_SAMPLES)]))
            overlap[f"{L}:{p}"] = float(np.mean([
                jacc(topk[f"{qid}|dense_f"], topk[f"{qid}|{L}:hyde_{p}_f|{r}"])
                for qid, _ in QUERIES_V2 for r in range(N_SAMPLES)]))

    reps = sorted({f["repo"] for f in corpus})
    out = {
        "metadata": {
            "embedding_model": EMBEDDING_MODEL, "k": K, "relevant_cc": RELEVANT_CC, "filter_cc": FILTER_CC,
            "n_samples": N_SAMPLES, "deterministic_run": DETERMINISTIC_RUN, "dose_n": list(DOSE_N), "dose_subsets": "all combinations (cap 70)",
            "rr_definition": "reciprocal rank of the first relevant unit over the full ranking (no depth cutoff); "
                             "random baseline exact",
            "n_queries": len(QUERIES_V2), "n_corpus": len(corpus), "n_relevant": int(ev.rel.sum()),
            "n_filter_pool": int(ev.mask_f.sum()), "repos": data["repos"],
            "per_repo": {r: {"n": sum(f["repo"] == r for f in corpus),
                             "mean_cc": float(np.mean([f["cc"] for f in corpus if f["repo"] == r])),
                             "n_relevant": sum(f["repo"] == r and f["cc"] >= RELEVANT_CC for f in corpus)}
                         for r in reps},
            "corpus_mean_cc": float(ev.cc.mean()), "corpus_median_cc": float(np.median(ev.cc)),
            "corpus_max_cc": int(ev.cc.max()),
            "rank_dist": {g: int(sum(f["rank"] == g for f in corpus)) for g in "ABCDEF"},
            "llms": [{**{k: m[k] for k in ("id", "label", "model", "open_weight")},
                      "endpoint": m["base_url"], "max_tokens": m.get("max_tokens"),
                      "manifest": manifest.get(m["id"])} for m in llms],
            "excluded_llms": excluded, "data_quality": quality,
        },
        "shared_modes": SHARED, "summary": summary, "summary_untrunc": summary_untrunc,
        "truncation": trunc_info, "per_query": per_query,
        "hypothetical_stats": hyp, "run_stability_jaccard": stability,
        "topk_overlap_dense_f_vs_hyde_f": overlap, "topk": topk,
    }
    json.dump(out, open(V3_RESULTS_FILE, "w"), indent=1)

    # --- ringkasan konsol
    print(f"\n{'mode':<30}{'P@10':>7}{'nDCG':>7}{'MeanCC':>8}{'MaxCC':>7}{'MRR':>7}")
    for m in modes:
        if "@" in m:
            continue
        s = summary[m]
        print(f"{m:<30}{s['p_at_k']:>7.3f}{s['ndcg_at_k']:>7.3f}{s['mean_cc_at_k']:>8.2f}"
              f"{s['max_cc_at_k']:>7.1f}{s['rr']:>7.3f}")
    print("\nDose-response P@10 (avg@n):")
    for llm in llms:
        L = llm["id"]
        for p in PROMPTS:
            cells = []
            for n in DOSE_N:
                key = L + ":hyde_" + p + "_avg@" + str(n)
                cells.append("n=%d:%.3f" % (n, summary[key]["p_at_k"]))
            print("  " + L + ":" + p + "  " + "  ".join(cells))
    print("\nCC hipotesis (T=0.7):", {k: round(v["cc_median"], 1) for k, v in hyp.items()
                                       if k.endswith("t07") and v["cc_median"] is not None})
    print("Stabilitas antar-generasi (Jaccard):", {k: round(v, 2) for k, v in stability.items()})
    print(f"[✓] Saved {V3_RESULTS_FILE}")


if __name__ == "__main__":
    main()
