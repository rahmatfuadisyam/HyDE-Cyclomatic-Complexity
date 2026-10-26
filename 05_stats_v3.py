"""
v3 Langkah 5: Statistik multi-LLM (semua dihitung dari data).

  1. Keluarga "shared"   : Dense vs BM25, efek filter (Wilcoxon, Holm)
  2. Keluarga per LLM    : 12 perbandingan HyDE vs baseline (Wilcoxon, Holm dalam LLM x metrik)
  3. Pooled lintas LLM   : selisih per query dirata-rata atas LLM (robustness), Wilcoxon + Holm
  4. Lintas LLM          : Friedman (>= 3 LLM) / Wilcoxon (2 LLM) + Kendall's W, per mode HyDE
  5. Dose-response N     : Page's L trend test (hipotesis: metrik naik dengan n), Holm
  6. Mixed-effects model : metric ~ prompt * aggregation + llm + (1 | query)   [statsmodels, opsional]
Effect size: matched-pairs rank-biserial; 95% bootstrap CI selisih rata-rata.
"""
import json
import warnings

import numpy as np
from scipy.stats import wilcoxon, rankdata, friedmanchisquare, page_trend_test, spearmanr

from config_v3 import (V3_RESULTS_FILE, V3_STATS_FILE, V3_HYPO_CACHE, N_BOOTSTRAP, RANDOM_SEED,
                       PROMPTS, DOSE_N, N_SAMPLES)

METRICS = ["p_at_k", "ndcg_at_k", "mean_cc_at_k", "max_cc_at_k", "rr"]
PRIMARY = ["p_at_k", "ndcg_at_k"]


def per_llm_comparisons(L):
    c = []
    for p in PROMPTS:
        c += [(f"{L}:hyde_{p}", "dense", f"HyDE-{p} (single) vs Dense"),
              (f"{L}:hyde_{p}_avg", "dense", f"HyDE-{p} (avg, Gao) vs Dense"),
              (f"{L}:hyde_{p}_t0", "dense", f"HyDE-{p} (T=0) vs Dense"),
              (f"{L}:hyde_{p}_f", "dense_f", f"HyDE-{p} (single) + filter vs Dense + filter"),
              (f"{L}:hyde_{p}_avg_f", "dense_f", f"HyDE-{p} (avg) + filter vs Dense + filter")]
    c += [(f"{L}:hyde_fs", f"{L}:hyde_zs", "few-shot vs zero-shot (single)"),
          (f"{L}:hyde_fs_avg", f"{L}:hyde_zs_avg", "few-shot vs zero-shot (avg)")]
    return c


SHARED = [("dense", "bm25", "Dense vs BM25"), ("dense_f", "dense", "Dense + filter vs Dense"),
          ("dense", "random", "Dense vs Random"), ("dense_f", "random_f", "Dense + filter vs Random + filter"),
          ("random_f", "random", "Filter only (random within filter) vs Random")]
# query yang predikatnya setara dengan CC (Q03 = jumlah parameter, Q18 = fungsi 'terlalu banyak tugas',
# Q19 = kedalaman indentasi tidak ekuivalen dengan CC >= 11)
NON_CC_QUERIES = ("Q03", "Q18", "Q19")


def rank_biserial(d):
    d = np.asarray([x for x in d if x != 0])
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def boot_ci(d, rng):
    d = np.asarray(d)
    m = d[rng.integers(0, len(d), (N_BOOTSTRAP, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for i, j in enumerate(order):
        run = max(run, (len(ps) - i) * ps[j])
        adj[j] = min(1.0, run)
    return adj.tolist()


def paired(xa, xb, rng):
    d = xa - xb
    if np.allclose(d, 0):
        stat, p = 0.0, 1.0
    else:
        w = wilcoxon(xa, xb, zero_method="wilcox", alternative="two-sided")
        stat, p = float(w.statistic), float(w.pvalue)
    return {"mean_a": float(xa.mean()), "mean_b": float(xb.mean()), "mean_diff": float(d.mean()),
            "ci95": boot_ci(d, rng), "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
            "losses": int((d < 0).sum()), "W": stat, "p": p, "r_rb": rank_biserial(d)}


def family(rows):
    for row, padj in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = padj
        row["significant"] = padj < 0.05
    return rows


def vec(pq, mode, metric, key="modes"):
    return np.array([r[key][mode][metric] for r in pq])


def onesample(d, rng):
    """Uji satu-sampel (Wilcoxon) bahwa kontras per query = 0, dengan IK bootstrap."""
    d = np.asarray(d)
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d, zero_method="wilcox").pvalue)
    return {"mean_diff": float(d.mean()), "ci95": boot_ci(d, rng), "p": p, "r_rb": rank_biserial(d),
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum())}


def boot_ci90(d, rng):
    d = np.asarray(d)
    m = d[rng.integers(0, len(d), (N_BOOTSTRAP, len(d)))].mean(1)
    return [float(np.percentile(m, 5)), float(np.percentile(m, 95))]


def kendalls_w(chi2, n, k):
    return float(chi2 / (n * (k - 1))) if k > 1 else None


def mixed_model(pq, llms, metric):
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        return {"skipped": "statsmodels/pandas tidak terpasang"}
    rows = []
    for r in pq:
        for L in llms:
            for p in PROMPTS:
                for agg, suf in (("single", ""), ("avg", "_avg")):
                    rows.append({"qid": r["qid"], "llm": L, "prompt": p, "agg": agg,
                                 "y": r["modes"][f"{L}:hyde_{p}{suf}"][metric],
                                 "y_dense": r["modes"]["dense"][metric]})
    df = pd.DataFrame(rows)
    df["delta"] = df["y"] - df["y_dense"]          # perbaikan relatif terhadap Dense pada query yang sama
    formula = "delta ~ C(prompt, Treatment('zs')) * C(agg, Treatment('single'))" + \
              (" + C(llm)" if len(llms) > 1 else "")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = smf.mixedlm(formula, df, groups=df["qid"]).fit(reml=True)
        except Exception as e:
            return {"error": str(e)[:300]}
    ci = fit.conf_int()
    return {"formula": formula + " + (1|query)", "n_obs": int(len(df)),
            "converged": bool(getattr(fit, "converged", True)),
            "params": {k: {"coef": float(fit.params[k]), "se": float(fit.bse[k]), "p": float(fit.pvalues[k]),
                           "ci95": [float(ci.loc[k, 0]), float(ci.loc[k, 1])]}
                       for k in fit.params.index if k != "Group Var"},
            "group_var": float(fit.cov_re.iloc[0, 0])}


def mechanism(pq, llms, metric="p_at_k"):
    """Apakah kompleksitas (CC) / panjang kode hipotetis memprediksi keberhasilan retrieval?
    Tingkat generasi: korelasi Spearman setelah dipusatkan per (LLM, prompt, query) -> hanya
    variasi ANTAR generasi untuk kebutuhan yang sama yang dipakai (menghilangkan efek query/model).
    Tingkat sel: median CC hipotesis per (LLM, prompt) vs rata-rata Δmetric terhadap Dense."""
    cache = json.load(open(V3_HYPO_CACHE))
    out = {"generation_level": {}, "cell_level": {}}
    pooled = {"cc": [], "len": [], "y": []}
    cells = []
    for L in llms:
        g = {"cc": [], "len": [], "y": []}
        for p in PROMPTS:
            cell_cc, cell_d = [], []
            for r in pq:
                runs = r["runs"][f"{L}:hyde_{p}"]
                xs, ls, ys = [], [], []
                for i in range(min(N_SAMPLES, len(runs))):
                    e = cache.get(f"{L}|{r['qid']}|{p}|t07|{i}")
                    if e is None or e.get("cc") is None:
                        continue
                    xs.append(e["cc"]); ls.append(len(e["code"])); ys.append(runs[i][metric])
                    cell_cc.append(e["cc"])
                if len(xs) >= 2:
                    for arr, key in ((xs, "cc"), (ls, "len"), (ys, "y")):
                        a = np.asarray(arr, float)
                        g[key] += (a - a.mean()).tolist()
                cell_d.append(r["modes"][f"{L}:hyde_{p}"][metric] - r["modes"]["dense"][metric])
            cells.append({"llm": L, "prompt": p, "median_cc": float(np.median(cell_cc)) if cell_cc else None,
                          "mean_delta": float(np.mean(cell_d))})
        res = {}
        for key in ("cc", "len"):
            if np.std(g[key]) > 0 and np.std(g["y"]) > 0:
                rho, pv = spearmanr(g[key], g["y"])
                res[key] = {"rho": float(rho), "p": float(pv), "n": len(g["y"])}
        out["generation_level"][L] = res
        for k in pooled:
            pooled[k] += g[k]
    for key in ("cc", "len"):
        rho, pv = spearmanr(pooled[key], pooled["y"])
        out["generation_level"].setdefault("_pooled", {})[key] = {"rho": float(rho), "p": float(pv),
                                                                   "n": len(pooled["y"])}
    ok = [c for c in cells if c["median_cc"] is not None]
    rho, pv = spearmanr([c["median_cc"] for c in ok], [c["mean_delta"] for c in ok])
    out["cell_level"] = {"cells": cells, "rho": float(rho), "p": float(pv), "n": len(ok)}
    return out


def main():
    res = json.load(open(V3_RESULTS_FILE))
    pq = res["per_query"]
    llms = [m["id"] for m in res["metadata"]["llms"]]
    rng = np.random.default_rng(RANDOM_SEED)
    out = {"n_queries": len(pq), "llms": llms, "shared": {}, "per_llm": {}, "pooled": {},
           "cross_llm": {}, "dose_response": {}, "mixed_model": {}, "mechanism": {}, "aggregation": {},
           "interaction": {}, "equivalence": {}, "sensitivity_cc_queries": {}, "sensitivity_untrunc": {},
           "dose_response_noquery": {}}
    has_nq = f"{llms[0]}:hyde_fs_avgnq" in pq[0]["modes"]
    has_u = bool(pq[0].get("modes_untrunc"))
    pq_cc = [r for r in pq if r["qid"] not in NON_CC_QUERIES]
    out["sensitivity_cc_queries_excluded"] = list(NON_CC_QUERIES)

    for metric in METRICS:
        # 1. shared
        out["shared"][metric] = family([{"a": a, "b": b, "desc": d, **paired(vec(pq, a, metric), vec(pq, b, metric), rng)}
                                        for a, b, d in SHARED])
        # 2. per LLM
        for L in llms:
            out["per_llm"].setdefault(L, {})[metric] = family(
                [{"a": a, "b": b, "desc": d, **paired(vec(pq, a, metric), vec(pq, b, metric), rng)}
                 for a, b, d in per_llm_comparisons(L)])
        # 3. pooled: rata-rata atas LLM per query, lalu bandingkan dengan baseline
        pooled = []
        for suf, base in (("", "dense"), ("_avg", "dense"), ("_t0", "dense"), ("_f", "dense_f"), ("_avg_f", "dense_f")):
            for p in PROMPTS:
                xa = np.mean([vec(pq, f"{L}:hyde_{p}{suf}", metric) for L in llms], axis=0)
                pooled.append({"a": f"pooled:hyde_{p}{suf}", "b": base,
                               "desc": f"HyDE-{p}{suf} (mean over {len(llms)} LLMs) vs {base}",
                               **paired(xa, vec(pq, base, metric), rng)})
        out["pooled"][metric] = family(pooled)
        # 3b. agregasi: avg (Gao) vs single per LLM dan pooled
        # dekomposisi: (i) avg+query vs single; (ii) avg tanpa query vs single (efek jumlah sampel);
        # (iii) avg+query vs avg tanpa query (efek penyertaan vektor query)
        contrasts = [("_avg", "", "avg+query vs single")]
        if has_nq:
            contrasts += [("_avgnq", "", "avg (no query) vs single"), ("_avg", "_avgnq", "avg+query vs avg (no query)")]
        agg = []
        for sa, sb, desc in contrasts:
            for p in PROMPTS:
                for L in llms:
                    agg.append({"a": f"{L}:hyde_{p}{sa}", "b": f"{L}:hyde_{p}{sb}", "desc": f"{L} {p}: {desc}",
                                **paired(vec(pq, f"{L}:hyde_{p}{sa}", metric), vec(pq, f"{L}:hyde_{p}{sb}", metric), rng)})
                xa = np.mean([vec(pq, f"{L}:hyde_{p}{sa}", metric) for L in llms], axis=0)
                xb = np.mean([vec(pq, f"{L}:hyde_{p}{sb}", metric) for L in llms], axis=0)
                agg.append({"a": f"pooled:hyde_{p}{sa}", "b": f"pooled:hyde_{p}{sb}", "desc": f"pooled {p}: {desc}",
                            **paired(xa, xb, rng)})
        out["aggregation"][metric] = family(agg)
        # 3c. interaksi HyDE x filter: (HyDE_f - Dense_f) - (HyDE - Dense), per query
        inter = []
        for suf in ("", "_avg"):
            for p in PROMPTS:
                for L in llms + ["pooled"]:
                    def g(mode):
                        if L == "pooled":
                            return np.mean([vec(pq, f"{x}:{mode}", metric) for x in llms], axis=0)
                        return vec(pq, f"{L}:{mode}", metric)
                    d = (g(f"hyde_{p}{suf}_f") - vec(pq, "dense_f", metric)) - (g(f"hyde_{p}{suf}") - vec(pq, "dense", metric))
                    inter.append({"a": f"{L}:hyde_{p}{suf}", "b": "interaction", "desc": f"{L} {p}{suf}: HyDE x filter",
                                  **onesample(d, rng)})
        out["interaction"][metric] = family(inter)
        # 3d. kesetaraan Dense vs acak: IK 90% (logika TOST; batas ekuivalensi ditentukan pembaca)
        d = vec(pq, "dense", metric) - vec(pq, "random", metric)
        out["equivalence"][metric] = {"contrast": "dense - random", "mean_diff": float(d.mean()),
                                      "ci90": boot_ci90(d, rng), "ci95": boot_ci(d, rng),
                                      "note": "equivalence holds only for margins larger than max(|ci90|)"}
        # 3e. sensitivitas: tanpa query non-CC
        sens = []
        for suf in ("", "_avg"):
            for p in PROMPTS:
                xa = np.mean([vec(pq_cc, f"{L}:hyde_{p}{suf}", metric) for L in llms], axis=0)
                sens.append({"a": f"pooled:hyde_{p}{suf}", "b": "dense", "desc": f"CC-only queries: HyDE-{p}{suf} vs Dense",
                             **paired(xa, vec(pq_cc, "dense", metric), rng)})
        sens.append({"a": "dense", "b": "random", "desc": "CC-only queries: Dense vs Random",
                     **paired(vec(pq_cc, "dense", metric), vec(pq_cc, "random", metric), rng)})
        out["sensitivity_cc_queries"][metric] = family(sens)
        # 3f. sensitivitas: subkorpus tanpa unit terpotong encoder
        if has_u:
            su = []
            for suf in ("", "_avg"):
                for p in PROMPTS:
                    xa = np.mean([vec(pq, f"{L}:hyde_{p}{suf}", metric, "modes_untrunc") for L in llms], axis=0)
                    su.append({"a": f"pooled:hyde_{p}{suf}", "b": "dense", "desc": f"untruncated corpus: HyDE-{p}{suf} vs Dense",
                               **paired(xa, vec(pq, "dense", metric, "modes_untrunc"), rng)})
            su.append({"a": "dense", "b": "random", "desc": "untruncated corpus: Dense vs Random",
                       **paired(vec(pq, "dense", metric, "modes_untrunc"), vec(pq, "random", metric, "modes_untrunc"), rng)})
            out["sensitivity_untrunc"][metric] = family(su)
        # 4. lintas LLM
        if len(llms) >= 2:
            cross = []
            for p in PROMPTS:
                for suf in ("", "_avg", "_t0", "_f"):
                    cols = [vec(pq, f"{L}:hyde_{p}{suf}", metric) for L in llms]
                    if len(llms) >= 3:
                        try:
                            chi2, pv = friedmanchisquare(*cols)
                            cross.append({"mode": f"hyde_{p}{suf}", "test": "Friedman", "chi2": float(chi2),
                                          "p": float(pv), "kendalls_w": kendalls_w(chi2, len(pq), len(llms)),
                                          "means": {L: float(c.mean()) for L, c in zip(llms, cols)}})
                        except ValueError:   # semua nilai identik
                            cross.append({"mode": f"hyde_{p}{suf}", "test": "Friedman", "chi2": 0.0, "p": 1.0,
                                          "kendalls_w": 0.0, "means": {L: float(c.mean()) for L, c in zip(llms, cols)}})
                    else:
                        r = paired(cols[0], cols[1], rng)
                        cross.append({"mode": f"hyde_{p}{suf}", "test": "Wilcoxon", "p": r["p"],
                                      "r_rb": r["r_rb"], "means": {L: float(c.mean()) for L, c in zip(llms, cols)}})
            out["cross_llm"][metric] = family(cross)
        # 5. dose-response
        dose = []
        ns = [n for n in DOSE_N if f"{llms[0]}:hyde_fs_avg@{n}" in pq[0]["modes"]]
        if len(ns) >= 3:
            for L in llms:
                for p in PROMPTS:
                    M = np.column_stack([vec(pq, f"{L}:hyde_{p}_avg@{n}", metric) for n in ns])
                    try:
                        pt = page_trend_test(M)
                        stat, pv = float(pt.statistic), float(pt.pvalue)
                    except Exception:
                        stat, pv = 0.0, 1.0
                    dose.append({"llm": L, "prompt": p, "n": ns, "means": M.mean(0).tolist(),
                                 "page_L": stat, "p": pv})
            out["dose_response"][metric] = family(dose)
            if has_nq:
                dnq = []
                for L in llms:
                    for p in PROMPTS:
                        Mq = np.column_stack([vec(pq, f"{L}:hyde_{p}_avgnq@{n}", metric) for n in ns])
                        try:
                            pt = page_trend_test(Mq)
                            stat_, pv_ = float(pt.statistic), float(pt.pvalue)
                        except Exception:
                            stat_, pv_ = 0.0, 1.0
                        dnq.append({"llm": L, "prompt": p, "n": ns, "means": Mq.mean(0).tolist(), "page_L": stat_, "p": pv_})
                out["dose_response_noquery"][metric] = family(dnq)
        # 6. mixed model (metrik primer saja)
        if metric in PRIMARY:
            out["mixed_model"][metric] = mixed_model(pq, llms, metric)
            out["mechanism"][metric] = mechanism(pq, llms, metric)

    json.dump(out, open(V3_STATS_FILE, "w"), indent=1)

    # ---- ringkasan konsol (P@10)
    m = "p_at_k"
    def line(r, a=None):
        return (f"  {(a or r['a']):<28} vs {r['b']:<16} Δ={r['mean_diff']:+.3f} "
                f"CI[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] W/T/L={r['wins']}/{r['ties']}/{r['losses']} "
                f"p_holm={r['p_holm']:.4f} r={r['r_rb']:+.2f}{' *' if r['significant'] else ''}")
    print("== SHARED (P@10) =="); [print(line(r)) for r in out["shared"][m]]
    for L in llms:
        print(f"== {L} (P@10) =="); [print(line(r)) for r in out["per_llm"][L][m]]
    print("== POOLED over LLMs (P@10) =="); [print(line(r)) for r in out["pooled"][m]]
    print("== AGREGASI (P@10) =="); [print(line(r)) for r in out["aggregation"][m] if r["a"].startswith("pooled")]
    print("== INTERAKSI HyDE x filter (P@10) ==")
    for r in out["interaction"][m]:
        if r["a"].startswith("pooled"):
            print(f"  {r['desc']:<40} Δ={r['mean_diff']:+.3f} CI[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] p_holm={r['p_holm']:.4f}")
    e = out["equivalence"][m]
    print(f"== KESETARAAN dense-random: {e['mean_diff']:+.3f} IK90 [{e['ci90'][0]:+.3f},{e['ci90'][1]:+.3f}]")
    print("== SENSITIVITAS query CC saja =="); [print(line(r)) for r in out["sensitivity_cc_queries"][m]]
    if out["sensitivity_untrunc"]:
        print("== SENSITIVITAS korpus tanpa unit terpotong =="); [print(line(r)) for r in out["sensitivity_untrunc"][m]]
    if out["cross_llm"]:
        print("== CROSS-LLM (P@10) ==")
        for r in out["cross_llm"][m]:
            print(f"  {r['mode']:<14} {r['test']} p_holm={r['p_holm']:.4f} means={ {k: round(v, 3) for k, v in r['means'].items()} }")
    if out["dose_response"]:
        print("== DOSE-RESPONSE (P@10, Page's L) ==")
        for r in out["dose_response"][m]:
            print(f"  {r['llm']}:{r['prompt']} n={r['n']} means={[round(x, 3) for x in r['means']]} "
                  f"p_holm={r['p_holm']:.4f}{' *' if r['significant'] else ''}")
    mm = out["mixed_model"].get(m, {})
    if "params" in mm:
        print(f"== MIXED MODEL (ΔP@10 vs Dense) {mm['formula']} ==")
        for k, v in mm["params"].items():
            print(f"  {k:<60} {v['coef']:+.3f} (SE {v['se']:.3f}) p={v['p']:.4f}")
    else:
        print("== MIXED MODEL ==", mm)
    me = out["mechanism"]["p_at_k"]
    print("== MEKANISME: CC/panjang hipotesis vs P@10 (antar-generasi, dipusatkan per query) ==")
    for L, v in me["generation_level"].items():
        print(f"  {L:<18} " + "  ".join(f"{k}: rho={x['rho']:+.3f} p={x['p']:.4f} n={x['n']}" for k, x in v.items()))
    print(f"  sel LLM×prompt: median CC vs ΔP@10  rho={me['cell_level']['rho']:+.3f} "
          f"p={me['cell_level']['p']:.4f} n={me['cell_level']['n']}")
    print(f"\n[✓] Saved {V3_STATS_FILE}")


if __name__ == "__main__":
    main()
