"""
v3 Langkah 5b (revisi review kedua): statistik tambahan — SEMUA dianalisis setelah review dan
diberi label "pasca-review" di naskah.

Keluarga Holm (masing-masing dikoreksi di dalam keluarganya sendiri):
  F_abl  : ablasi penyertaan query & jumlah hipotesis (gabungan atas generator)
  F_fw   : tren bobot-query-tetap terhadap n (uji L Page per generator x prompt)
  F_grp  : kelompok query (HyDE few-shot/zero-shot multi-sampel vs Dense; Dense vs acak)
  F_rep_<rep> : per representasi fungsi utuh (HyDE vs Dense; Dense vs acak)
  F_repc : kontras representasi terhadap 'first' (selisih efek HyDE dan selisih Dense)
Korelasi CC hipotesis–P@10: IK 95% bootstrap KLASTER (resampling query, seluruh titiknya ikut).
Bootstrap: unit = query, N_BOOT resampling, interval persentil, seed RANDOM_SEED.
"""
import json
import os
from importlib import import_module

import numpy as np
from scipy.stats import spearmanr, page_trend_test

from config_v3 import (V3_RESULTS_FILE, V3_STATS_FILE, V3_HYPO_CACHE, V3_FULLFUNC_FILE, V3_SUPP_STATS_FILE,
                       PROMPTS, N_SAMPLES, DOSE_N, QUERY_GROUPS, RANDOM_SEED, N_BOOT, SESOI_P10, RELEVANT_CC)

s5 = import_module("05_stats_v3")
paired, family, vec, onesample, boot_ci90 = s5.paired, s5.family, s5.vec, s5.onesample, s5.boot_ci90
M = "p_at_k"
K_TOP = 10


def pooled_vec(pq, llms, suffix, key="modes"):
    return np.mean([vec(pq, f"{L}:hyde_{suffix}", M, key) for L in llms], axis=0)


def cluster_rho(pq, llms, cache, rng, n_boot, xkey="cc"):
    """Spearman CC-hipotesis (atau panjang, xkey='len') vs P@10 (dipusatkan per sel), IK bootstrap klaster per query."""
    pts = {}  # qid -> list of (cc_c, y_c)
    for L in llms:
        for p in PROMPTS:
            for r in pq:
                runs = r["runs"][f"{L}:hyde_{p}"]
                xs, ys = [], []
                for i in range(min(N_SAMPLES, len(runs))):
                    e = cache.get(f"{L}|{r['qid']}|{p}|t07|{i}")
                    if e is None or e.get("cc") is None:
                        continue
                    xs.append(e["cc"] if xkey == "cc" else len(e["code"])); ys.append(runs[i][M])
                if len(xs) >= 2:
                    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
                    pts.setdefault(r["qid"], []).extend(zip(xs - xs.mean(), ys - ys.mean()))
    qids = sorted(pts)
    allp = np.array([t for q in qids for t in pts[q]])
    rho = float(spearmanr(allp[:, 0], allp[:, 1]).correlation)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(qids), len(qids), replace=True)
        b = np.array([t for i in pick for t in pts[qids[i]]])
        if np.std(b[:, 0]) > 0 and np.std(b[:, 1]) > 0:
            boots.append(spearmanr(b[:, 0], b[:, 1]).correlation)
    return {"rho": rho, "n_points": int(len(allp)), "n_clusters": len(qids),
            "ci95_cluster": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_boot": len(boots)}


def rho_from_pts(pts, rng, n_boot):
    qids = sorted(q for q in pts if pts[q])
    if not qids:
        return None
    allp = np.array([t for q in qids for t in pts[q]])
    if len(allp) < 5 or np.std(allp[:, 0]) == 0 or np.std(allp[:, 1]) == 0:
        return None
    rho = float(spearmanr(allp[:, 0], allp[:, 1]).correlation)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(qids), len(qids), replace=True)
        b = np.array([t for i in pick for t in pts[qids[i]]])
        if np.std(b[:, 0]) > 0 and np.std(b[:, 1]) > 0:
            boots.append(spearmanr(b[:, 0], b[:, 1]).correlation)
    return {"rho": rho, "n_points": int(len(allp)), "n_clusters": len(qids),
            "ci95_cluster": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "n_boot": len(boots)}


def compliance_assoc(pq, llms, cache, rng, n_boot, runs_of):
    """Review 3: (1) patuh (CC >= RELEVANT_CC) vs tidak patuh di dalam sel generator-query-prompt;
    (2) di antara hipotesis yang patuh, Spearman CC (dipusatkan per sel) vs P@10, IK bootstrap klaster per query.
    CC tak terukur dihitung tidak patuh dan tidak masuk analisis (2)."""
    diffs, pts, n_cells, n_both = {}, {}, 0, 0
    n_comp = n_non = 0
    for r in pq:
        for L in llms:
            for p in PROMPTS:
                ys = runs_of(r, L, p)
                if not ys:
                    continue
                n_cells += 1
                comp, non, xs, yc = [], [], [], []
                for i, y in enumerate(ys[:N_SAMPLES]):
                    e = cache.get(f"{L}|{r['qid']}|{p}|t07|{i}") or {}
                    cc = e.get("cc")
                    if cc is not None and cc >= RELEVANT_CC:
                        comp.append(y); xs.append(cc); yc.append(y)
                    else:
                        non.append(y)
                n_comp += len(comp); n_non += len(non)
                if comp and non:
                    n_both += 1
                    diffs.setdefault(r["qid"], []).append(float(np.mean(comp) - np.mean(non)))
                if len(xs) >= 2:
                    xs, yc = np.asarray(xs, float), np.asarray(yc, float)
                    pts.setdefault(r["qid"], []).extend(zip(xs - xs.mean(), yc - yc.mean()))
    qd = np.array([np.mean(v) for v in diffs.values()])
    binary = ({"n_queries": int(len(qd)), "n_cells_both": n_both, "n_cells": n_cells, "n_compliant": n_comp,
               "n_noncompliant": n_non, **onesample(qd, rng)} if len(qd) >= 5 else
              {"n_queries": int(len(qd)), "n_cells_both": n_both, "n_cells": n_cells})
    return {"binary": binary, "rho_compliant": rho_from_pts(pts, rng, n_boot)}


def main():
    res = json.load(open(V3_RESULTS_FILE))
    pq = res["per_query"]
    llms = [m["id"] for m in res["metadata"]["llms"]]
    cache = json.load(open(V3_HYPO_CACHE))
    rng = np.random.default_rng(RANDOM_SEED)
    out = {"post_review": True, "bootstrap": {"unit": "query", "n_boot": N_BOOT, "interval": "percentile",
                                               "seed": RANDOM_SEED},
           "sesoi_p10": SESOI_P10, "families": {}}

    # ---- F_abl: ablasi penyertaan query dan jumlah hipotesis
    has_q1 = f"{llms[0]}:hyde_fs_q1" in pq[0]["modes"]
    abl = []
    for p in PROMPTS:
        if has_q1:
            abl.append({"a": f"pooled:hyde_{p}_q1", "b": f"pooled:hyde_{p}", "desc": f"{p}: query + 1 hipotesis vs 1 hipotesis",
                        **paired(pooled_vec(pq, llms, f"{p}_q1"), pooled_vec(pq, llms, p), rng)})
            abl.append({"a": f"pooled:hyde_{p}_avg", "b": f"pooled:hyde_{p}_q1", "desc": f"{p}: query + 8 vs query + 1 (bobot query berubah)",
                        **paired(pooled_vec(pq, llms, f"{p}_avg"), pooled_vec(pq, llms, f"{p}_q1"), rng)})
        fw = [f"{p}_fw@{n}" for n in DOSE_N]
        if f"{llms[0]}:hyde_{fw[0]}" in pq[0]["modes"]:
            abl.append({"a": f"pooled:hyde_{fw[-1]}", "b": f"pooled:hyde_{fw[0]}",
                        "desc": f"{p}: bobot query tetap, n = {DOSE_N[-1]} vs n = {DOSE_N[0]}",
                        **paired(pooled_vec(pq, llms, fw[-1]), pooled_vec(pq, llms, fw[0]), rng)})
    out["ablation"] = family(abl) if abl else []
    out["families"]["F_abl"] = len(abl)

    # ---- F_fw: tren bobot tetap
    fwr = []
    if f"{llms[0]}:hyde_fs_fw@{DOSE_N[0]}" in pq[0]["modes"]:
        for L in llms:
            for p in PROMPTS:
                Mx = np.column_stack([vec(pq, f"{L}:hyde_{p}_fw@{n}", M) for n in DOSE_N])
                try:
                    t = page_trend_test(Mx); st_, pv = float(t.statistic), float(t.pvalue)
                except Exception:
                    st_, pv = 0.0, 1.0
                fwr.append({"llm": L, "prompt": p, "n": list(DOSE_N), "means": Mx.mean(0).tolist(), "page_L": st_, "p": pv})
        fwr = family(fwr)
    out["dose_fixed_weight"] = fwr
    out["families"]["F_fw"] = len(fwr)
    # rerata gabungan per n untuk kedua formulasi (deskriptif)
    out["dose_means_pooled"] = {}
    for p in PROMPTS:
        for form in ("avg", "avgnq", "fw"):
            k0 = f"{llms[0]}:hyde_{p}_{form}@{DOSE_N[0]}"
            if k0 in pq[0]["modes"]:
                out["dose_means_pooled"][f"{p}_{form}"] = [float(pooled_vec(pq, llms, f"{p}_{form}@{n}").mean()) for n in DOSE_N]

    # ---- F_grp: kelompok query
    grp = []
    for g, info in QUERY_GROUPS.items():
        sub = [r for r in pq if r["qid"] in info["qids"]]
        if len(sub) < 2:
            continue
        for suf, lab in (("fs_avg", "HyDE few-shot multi-sampel vs Dense"), ("zs_avg", "HyDE zero-shot multi-sampel vs Dense")):
            grp.append({"group": g, "a": f"pooled:hyde_{suf}", "b": "dense", "desc": f"{g}: {lab}", "n_queries": len(sub),
                        **paired(pooled_vec(sub, llms, suf), vec(sub, "dense", M), rng)})
        grp.append({"group": g, "a": "dense", "b": "random", "desc": f"{g}: Dense vs acak", "n_queries": len(sub),
                    **paired(vec(sub, "dense", M), vec(sub, "random", M), rng)})
    out["query_groups"] = family(grp)
    out["families"]["F_grp"] = len(grp)

    # ---- kesetaraan Dense vs acak terhadap SESOI
    d = vec(pq, "dense", M) - vec(pq, "random", M)
    ci90 = boot_ci90(d, rng)
    out["equivalence_sesoi"] = {"mean_diff": float(d.mean()), "ci90": ci90, "sesoi": SESOI_P10,
                                "equivalent": bool(-SESOI_P10 < ci90[0] and ci90[1] < SESOI_P10)}

    # ---- korelasi klaster
    out["cc_rho_cluster"] = cluster_rho(pq, llms, cache, rng, min(N_BOOT, 5000))
    # review 5: koefisien per generator dan untuk panjang hipotesis dengan IK bootstrap klaster per query
    out["rho_cluster_by_generator"] = {L: {"cc": cluster_rho(pq, [L], cache, rng, 2000),
                                           "len": cluster_rho(pq, [L], cache, rng, 2000, "len")} for L in llms}
    out["len_rho_cluster"] = cluster_rho(pq, llms, cache, rng, min(N_BOOT, 5000), "len")

    # ---- kepatuhan generasi (ringkas dari 04)
    out["compliance"] = {k: {kk: v.get(kk) for kk in ("n", "n_parsed", "pct_comply_all", "pct_cc_ge_rel", "pct_cc_none",
                                                       "pct_trimmed", "pct_truncated_any", "median_chars", "cc_median")}
                         for k, v in res["hypothetical_stats"].items() if k.endswith(":t07")}

    # ---- fungsi utuh
    if os.path.exists(V3_FULLFUNC_FILE):
        ff = json.load(open(V3_FULLFUNC_FILE))
        reps = {k: v for k, v in ff["representations"].items() if "per_query" in v}
        rand = {r["qid"]: r["modes"]["random"][M] for r in pq}
        out["fullfunc"] = {"coverage": next(iter(reps.values())).get("coverage"), "reps": {}, "contrast_vs_first": []}
        for name, rep in reps.items():
            rq = rep["per_query"]
            rows = []
            for suf in ("fs", "fs_avg", "zs", "zs_avg"):
                rows.append({"a": f"pooled:hyde_{suf}", "b": "dense", "desc": f"{name}: HyDE {suf} vs Dense",
                             **paired(pooled_vec(rq, llms, suf), vec(rq, "dense", M), rng)})
            rows.append({"a": "dense", "b": "random", "desc": f"{name}: Dense vs acak",
                         **paired(vec(rq, "dense", M), np.array([rand[r["qid"]] for r in rq]), rng)})
            out["fullfunc"]["reps"][name] = {"encoder": rep.get("encoder"), "tests": family(rows),
                                             "summary_dense": rep["summary"]["dense"][M],
                                             "summary_fs_avg_pooled": float(np.mean([rep["summary"][f"{L}:hyde_fs_avg"][M] for L in llms]))}
            out["families"][f"F_rep_{name}"] = len(rows)
        first = reps.get("first")
        con = []
        if first:
            fq = first["per_query"]
            eff_first = pooled_vec(fq, llms, "fs_avg") - vec(fq, "dense", M)
            for name, rep in reps.items():
                if name == "first":
                    continue
                rq = rep["per_query"]
                eff = pooled_vec(rq, llms, "fs_avg") - vec(rq, "dense", M)
                con.append({"a": name, "b": "first", "desc": f"efek HyDE few-shot multi-sampel: {name} - first",
                            **onesample(eff - eff_first, rng)})
                con.append({"a": f"{name}:dense", "b": "first:dense", "desc": f"Dense: {name} - first",
                            **paired(vec(rq, "dense", M), vec(fq, "dense", M), rng)})
            con = family(con)
        out["fullfunc"]["contrast_vs_first"] = con
        out["families"]["F_repc"] = len(con)
        out["fullfunc"]["errors"] = {k: v["error"] for k, v in ff["representations"].items() if "error" in v}

    # ---- review 3: baseline panjang (gabungan atas generator untuk HyDE)
    lens_rows = []
    if "length" in pq[0]["modes"]:
        for base, lab in (("length", "panjang token"), ("sloc", "SLOC")):
            if base not in pq[0]["modes"]:
                continue
            lens_rows.append({"a": "pooled:hyde_fs_avg", "b": base, "desc": f"HyDE few-shot multi-sampel vs {lab}",
                              **paired(pooled_vec(pq, llms, "fs_avg"), vec(pq, base, M), rng)})
            lens_rows.append({"a": "dense", "b": base, "desc": f"Dense vs {lab}",
                              **paired(vec(pq, "dense", M), vec(pq, base, M), rng)})
        out["length_baselines"] = {"summary": {k: res["summary"][k] for k in ("length", "sloc", "length_f") if k in res["summary"]},
                                   "tests": family(lens_rows)}
        out["families"]["F_len"] = len(lens_rows)

    # ---- review 3: kepatuhan biner vs CC kontinu, per representasi
    comp_rep = {"first": compliance_assoc(pq, llms, cache, rng, min(N_BOOT, 2000),
                                          lambda r, L, p: [x[M] for x in r["runs"].get(f"{L}:hyde_{p}", [])])}
    strata_out, strata_rows = {}, []
    if os.path.exists(V3_FULLFUNC_FILE):
        for name, rep in ff["representations"].items():
            if name == "first" or "per_query" not in rep or "runs" not in rep["per_query"][0]:
                continue
            byq = {r["qid"]: r for r in rep["per_query"]}
            comp_rep[name] = compliance_assoc(pq, llms, cache, rng, min(N_BOOT, 2000),
                                              lambda r, L, p, byq=byq: byq[r["qid"]]["runs"].get(f"{L}:hyde_{p}", []))
        # ---- review 3: strata jumlah potongan (kandidat dibatasi pada strata; korpus & relevansi tetap)
        for name, st_ in (ff.get("strata") or {}).items():
            strata_out[name] = {}
            for sname, sv in st_.items():
                sq = sv["per_query"]
                summ = {"n_docs": sv["n_docs"], "n_relevant": sv["n_relevant"],
                        "random_p10": sv["n_relevant"] / sv["n_docs"] if sv["n_docs"] else None,
                        "dense": sv["summary"]["dense"][M],
                        "fs_avg_pooled": float(np.mean([sv["summary"][f"{L}:hyde_fs_avg"][M] for L in llms])),
                        "length": sv["summary"].get("length", {}).get(M), "sloc": sv["summary"].get("sloc", {}).get(M)}
                strata_out[name][sname] = summ
                if sv["n_relevant"] > 0:
                    strata_rows.append({"rep": name, "stratum": sname, "a": "pooled:hyde_fs_avg", "b": "dense",
                                        "desc": f"{name} | {sname}: HyDE few-shot multi-sampel vs Dense",
                                        **paired(pooled_vec(sq, llms, "fs_avg"), vec(sq, "dense", M), rng)})
    # ---- review 3: komposisi 10 teratas (representasi utama): proporsi unit multi-potongan
    if os.path.exists(V3_FULLFUNC_FILE) and ff.get("metadata", {}).get("n_chunks_docs") and res.get("topk"):
        from common_v2 import load_corpus
        corp = load_corpus()["functions"]
        ncd = dict(zip([f["id"] for f in corp], ff["metadata"]["n_chunks_docs"]))
        ccd = {f["id"]: f["cc"] for f in corp}
        tk = res["topk"]
        qids = [r["qid"] for r in pq]

        def share(mode):
            return np.array([np.mean([ncd[i] > 1 for i in tk[f"{q}|{mode}"]]) for q in qids])

        def prec_split(mode):
            mu = [ccd[i] >= RELEVANT_CC for q in qids for i in tk[f"{q}|{mode}"] if ncd[i] > 1]
            si = [ccd[i] >= RELEVANT_CC for q in qids for i in tk[f"{q}|{mode}"] if ncd[i] == 1]
            return {"n_multi": len(mu), "prec_multi": float(np.mean(mu)) if mu else None,
                    "n_single": len(si), "prec_single": float(np.mean(si)) if si else None}

        sh_d = share("dense")
        sh_h = np.mean([share(f"{L}:hyde_fs_avg") for L in llms], axis=0)
        cells = []
        for L in llms:
            for p_ in PROMPTS:
                cells.append({"llm": L, "prompt": p_, "share_multi": float(share(f"{L}:hyde_{p_}_avg").mean()),
                              "p10": float(res["summary"][f"{L}:hyde_{p_}_avg"][M]), **prec_split(f"{L}:hyde_{p_}_avg")})
        rr_ = spearmanr([c["share_multi"] for c in cells], [c["p10"] for c in cells])
        out["topk_composition"] = {
            "multi_chunk_corpus_share": float(np.mean([v > 1 for v in ncd.values()])),
            "dense": {"share_multi": float(sh_d.mean()), **prec_split("dense")},
            "bm25": {"share_multi": float(share("bm25").mean())},
            "hyde_fs_avg_pooled": {"share_multi": float(sh_h.mean())},
            "test_share_hyde_vs_dense": paired(sh_h, sh_d, rng),
            "cells": cells, "cell_rho_share_p10": {"rho": float(rr_.correlation), "p": float(rr_.pvalue), "n": len(cells)}}
    # ---- review 4: dekomposisi perubahan P@10 dari RANKING GLOBAL (tanpa retrieval ulang per strata)
    #   P@10 = sum_s share_s * prec_s ; Delta = sum_s [(sh_H - sh_D) * (pr_H + pr_D)/2]   (komposisi)
    #                                         + sum_s [(pr_H - pr_D) * (sh_H + sh_D)/2]   (di dalam strata)
    #   (dekomposisi titik-tengah, eksak per query; prec tak terdefinisi bila strata kosong -> diisi prec metode lain,
    #    yang tidak mengubah identitas karena sumbangannya dikalikan share 0)
    if os.path.exists(V3_FULLFUNC_FILE) and ff.get("metadata", {}).get("n_chunks_docs") and res.get("topk"):
        def stratum_of(nc_, scheme):
            if scheme == "biner":
                return "1 potongan" if nc_ == 1 else "≥ 2 potongan"
            return "1" if nc_ == 1 else ("2" if nc_ == 2 else ("3–4" if nc_ <= 4 else "≥ 5"))

        def per_q(mode, q, scheme, strata):
            ids = tk[f"{q}|{mode}"]
            o = {}
            for st_ in strata:
                sel = [i for i in ids if stratum_of(ncd[i], scheme) == st_]
                o[st_] = (len(sel) / len(ids), (np.mean([ccd[i] >= RELEVANT_CC for i in sel]) if sel else None),
                          int(sum(ccd[i] >= RELEVANT_CC for i in sel)), len(sel))
            return o

        decomp = {}
        for scheme, strata in (("biner", ["1 potongan", "≥ 2 potongan"]), ("rinci", ["1", "2", "3–4", "≥ 5"])):
            comp_q, within_q, delta_q = [], [], []
            prof = {}
            for q in qids:
                D = per_q("dense", q, scheme, strata)
                cq = wq = 0.0
                for L in llms:
                    Hq = per_q(f"{L}:hyde_fs_avg", q, scheme, strata)
                    for st_ in strata:
                        shH, prH = Hq[st_][0], Hq[st_][1]
                        shD, prD = D[st_][0], D[st_][1]
                        if prH is None and prD is None:
                            continue
                        prH = prD if prH is None else prH
                        prD = prH if prD is None else prD
                        cq += (shH - shD) * (prH + prD) / 2 / len(llms)
                        wq += (prH - prD) * (shH + shD) / 2 / len(llms)
                comp_q.append(cq); within_q.append(wq); delta_q.append(cq + wq)
            comp_q, within_q, delta_q = map(np.asarray, (comp_q, within_q, delta_q))
            # profil per metode: proporsi 10 teratas, jumlah relevan, relevansi kondisional per strata
            for mode, lab in [("dense", "Dense")] + [(f"{L}:hyde_fs_avg", L) for L in llms]:
                rows_ = {}
                for st_ in strata:
                    tot = [per_q(mode, q, scheme, strata)[st_] for q in qids]
                    n_in = sum(t[3] for t in tot); r_in = sum(t[2] for t in tot)
                    rows_[st_] = {"share": n_in / (len(qids) * K_TOP), "n": n_in, "n_relevant": r_in,
                                  "cond_prec": r_in / n_in if n_in else None}
                prof[lab] = rows_
            idx = rng.integers(0, len(qids), (N_BOOT, len(qids)))
            bc, bw, bd = comp_q[idx].mean(1), within_q[idx].mean(1), delta_q[idx].mean(1)
            frac = bc / np.where(bd == 0, np.nan, bd)
            pct = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
            base = {}
            for st_ in strata:
                ids_st = [i for i in ncd if stratum_of(ncd[i], scheme) == st_]
                base[st_] = {"n_docs": len(ids_st), "n_relevant": int(sum(ccd[i] >= RELEVANT_CC for i in ids_st))}
            decomp[scheme] = {"strata": strata, "corpus": base, "profile": prof,
                              "delta": {"mean": float(delta_q.mean()), "ci95": pct(bd)},
                              "composition": {"mean": float(comp_q.mean()), "ci95": pct(bc)},
                              "within": {"mean": float(within_q.mean()), "ci95": pct(bw)},
                              "composition_fraction": {"ratio_of_means": float(comp_q.mean() / delta_q.mean()) if delta_q.mean() else None,
                                                       "ci95": pct(frac)}}
        # review 5: sensitivitas terhadap aturan pengisian presisi kondisional pada strata kosong
        def decomp_conv(scheme, strata, conv):
            prev = {}
            for st_ in strata:
                ids_st = [i for i in ncd if stratum_of(ncd[i], scheme) == st_]
                prev[st_] = float(np.mean([ccd[i] >= RELEVANT_CC for i in ids_st])) if ids_st else 0.0
            cq_, wq_ = [], []
            n_pairs = n_one = n_both = 0
            for q in qids:
                D = per_q("dense", q, scheme, strata)
                c_ = w_ = 0.0
                for L in llms:
                    Hq = per_q(f"{L}:hyde_fs_avg", q, scheme, strata)
                    for st_ in strata:
                        n_pairs += 1
                        shH, prH = Hq[st_][0], Hq[st_][1]
                        shD, prD = D[st_][0], D[st_][1]
                        if prH is None and prD is None:
                            n_both += 1
                            continue
                        if prH is None or prD is None:
                            n_one += 1
                        fill = {"pembanding": None, "nol": 0.0, "prevalensi": prev[st_]}[conv]
                        if prH is None:
                            prH = prD if fill is None else fill
                        if prD is None:
                            prD = prH if fill is None else fill
                        c_ += (shH - shD) * (prH + prD) / 2 / len(llms)
                        w_ += (prH - prD) * (shH + shD) / 2 / len(llms)
                cq_.append(c_); wq_.append(w_)
            cq_, wq_ = np.asarray(cq_), np.asarray(wq_)
            idx_ = rng.integers(0, len(qids), (N_BOOT, len(qids)))
            pc_ = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
            return {"composition": {"mean": float(cq_.mean()), "ci95": pc_(cq_[idx_].mean(1))},
                    "within": {"mean": float(wq_.mean()), "ci95": pc_(wq_[idx_].mean(1))},
                    "total": float((cq_ + wq_).mean()), "n_pairs": n_pairs, "n_one_empty": n_one, "n_both_empty": n_both}

        sens_ = {}
        for scheme, strata in (("biner", ["1 potongan", "≥ 2 potongan"]), ("rinci", ["1", "2", "3–4", "≥ 5"])):
            sens_[scheme] = {conv: decomp_conv(scheme, strata, conv) for conv in ("pembanding", "nol", "prevalensi")}
        out["decomposition_sensitivity"] = sens_
        out["decomposition_global"] = {"method": "midpoint (Kitagawa) decomposition of per-query P@10 difference, "
                                                 "HyDE few-shot multi-sample (mean over generators) minus Dense, BGE first-512",
                                       "schemes": decomp}

        # ---- review 4: baseline panjang biner (>= 2 potongan) dari strata 04c (representasi awal fungsi)
        st_first = (ff.get("strata") or {}).get("first", {}).get("≥ 2 potongan")
        if st_first:
            sq = {r["qid"]: r for r in st_first["per_query"]}
            hy_glob = pooled_vec(pq, llms, "fs_avg")
            rnd_multi = np.full(len(pq), st_first["n_relevant"] / st_first["n_docs"])
            dn_multi = np.array([sq[r["qid"]]["modes"]["dense"][M] for r in pq])
            rows_lb = [
                {"a": "pooled:hyde_fs_avg", "b": "random|multi", "desc": "HyDE few-shot multi-sampel (global) vs acak di antara unit ≥ 2 potongan",
                 **paired(hy_glob, rnd_multi, rng)},
                {"a": "pooled:hyde_fs_avg", "b": "dense|multi", "desc": "HyDE few-shot multi-sampel (global) vs Dense di antara unit ≥ 2 potongan",
                 **paired(hy_glob, dn_multi, rng)},
                {"a": "dense|multi", "b": "random|multi", "desc": "Dense di antara unit ≥ 2 potongan vs acak di antara unit ≥ 2 potongan",
                 **paired(dn_multi, rnd_multi, rng)}]
            out["length_binary_baseline"] = {"threshold_tokens": int(ff["metadata"].get("chunk_tokens", 480)),
                                             "random_multi_expected": float(rnd_multi[0]),
                                             "dense_multi": float(dn_multi.mean()), "hyde_global": float(hy_glob.mean()),
                                             "tests": family(rows_lb)}
            out["families"]["F_lenbin"] = len(rows_lb)
    out["compliance_assoc"] = comp_rep
    bin_rows = [{"rep": k, "desc": f"{k}: patuh vs tidak patuh", **v["binary"]} for k, v in comp_rep.items()
                if "p" in v["binary"]]
    out["compliance_binary_tests"] = family(bin_rows) if bin_rows else []
    out["families"]["F_comp"] = len(bin_rows)
    out["strata"] = {"summary": strata_out, "tests": family(strata_rows) if strata_rows else []}
    out["families"]["F_strata"] = len(strata_rows)

    # ---- daftar keluarga dari 05 (untuk spesifikasi di naskah)
    st = json.load(open(V3_STATS_FILE))
    fam5 = {"shared": len(st["shared"][M]), "pooled": len(st["pooled"][M]), "aggregation": len(st["aggregation"][M]),
            "interaction": len(st.get("interaction", {}).get(M, [])),
            "sensitivity_query": len(st.get("sensitivity_cc_queries", {}).get(M, [])),
            "sensitivity_untrunc": len(st.get("sensitivity_untrunc", {}).get(M, [])),
            "cross_llm": len(st.get("cross_llm", {}).get(M, [])), "dose": len(st.get("dose_response", {}).get(M, [])),
            "per_generator": {L: len(v[M]) for L, v in st["per_llm"].items()}}
    out["families_main"] = fam5
    json.dump(out, open(V3_SUPP_STATS_FILE, "w"), indent=1)

    # ---- ringkasan konsol
    print("== ABLASI ==")
    for r in out["ablation"]:
        print(f"  {r['desc']:<52} Δ={r['mean_diff']:+.3f} CI[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] p_holm={r['p_holm']:.4f}")
    print("== RERATA DOSIS (gabungan) ==", {k: [round(x, 3) for x in v] for k, v in out["dose_means_pooled"].items()})
    print("== KELOMPOK QUERY ==")
    for r in out["query_groups"]:
        print(f"  {r['desc']:<60} n={r['n_queries']} Δ={r['mean_diff']:+.3f} CI[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] p_holm={r['p_holm']:.4f}")
    e = out["equivalence_sesoi"]
    print(f"== KESETARAAN (SESOI ±{SESOI_P10}): IK90 [{e['ci90'][0]:+.3f},{e['ci90'][1]:+.3f}] setara={e['equivalent']}")
    c = out["cc_rho_cluster"]
    print(f"== RHO KLASTER: {c['rho']:+.3f} IK95 [{c['ci95_cluster'][0]:+.3f},{c['ci95_cluster'][1]:+.3f}] "
          f"({c['n_points']} titik, {c['n_clusters']} query)")
    if "fullfunc" in out:
        print("== FUNGSI UTUH ==", out["fullfunc"]["coverage"])
        for name, v in out["fullfunc"]["reps"].items():
            t = {r["a"]: r for r in v["tests"]}
            r = t["pooled:hyde_fs_avg"]
            print(f"  [{name}] dense={v['summary_dense']:.3f} fs_avg={v['summary_fs_avg_pooled']:.3f} "
                  f"Δ={r['mean_diff']:+.3f} CI[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] p_holm={r['p_holm']:.4f}")
        for r in out["fullfunc"]["contrast_vs_first"]:
            print(f"  {r['desc']:<50} Δ={r['mean_diff']:+.3f} CI[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] p_holm={r['p_holm']:.4f}")
        if out["fullfunc"]["errors"]:
            print("  error:", out["fullfunc"]["errors"])
    for r in out.get("length_baselines", {}).get("tests", []):
        print(f"  [panjang] {r['desc']:<45} Δ={r['mean_diff']:+.3f} p_holm={r['p_holm']:.4f}")
    print("== KEPATUHAN BINER / RHO PATUH ==")
    for k, v in out["compliance_assoc"].items():
        b, rc = v["binary"], v["rho_compliant"]
        print(f"  [{k}] patuh-tidak: " + (f"Δ={b['mean_diff']:+.3f} CI[{b['ci95'][0]:+.3f},{b['ci95'][1]:+.3f}] "
              f"n_q={b['n_queries']} sel={b['n_cells_both']}" if "mean_diff" in b else str(b))
              + (f" | rho_patuh={rc['rho']:+.3f} [{rc['ci95_cluster'][0]:+.3f},{rc['ci95_cluster'][1]:+.3f}]" if rc else ""))
    print("== STRATA POTONGAN ==")
    for r in out["strata"]["tests"]:
        print(f"  {r['desc']:<60} Δ={r['mean_diff']:+.3f} CI[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] p_holm={r['p_holm']:.4f}")
    for k, v in out["strata"]["summary"].items():
        print("   ", k, {s: {kk: (round(x, 3) if isinstance(x, float) else x) for kk, x in d.items()} for s, d in v.items()})
    if "topk_composition" in out:
        tc = out["topk_composition"]
        print("== KOMPOSISI 10 TERATAS (multi-potongan) ==", {k: tc[k] for k in ("multi_chunk_corpus_share", "dense", "bm25", "hyde_fs_avg_pooled")},
              "rho sel", tc["cell_rho_share_p10"])
    if "decomposition_global" in out:
        for sc, d in out["decomposition_global"]["schemes"].items():
            print(f"== DEKOMPOSISI ({sc}) Δ={d['delta']['mean']:+.3f} komposisi={d['composition']['mean']:+.3f} "
                  f"{[round(x,3) for x in d['composition']['ci95']]} di-dalam={d['within']['mean']:+.3f} "
                  f"{[round(x,3) for x in d['within']['ci95']]} fraksi={d['composition_fraction']}")
            for lab, rows_ in d["profile"].items():
                print("   ", lab, {k: (round(v['share'], 3), v['n_relevant'], None if v['cond_prec'] is None else round(v['cond_prec'], 3)) for k, v in rows_.items()})
    if "decomposition_sensitivity" in out:
        print("== SENSITIVITAS DEKOMPOSISI ==")
        for sc, d in out["decomposition_sensitivity"].items():
            for cv, r in d.items():
                print(f"  {sc:6s} {cv:11s} komposisi={r['composition']['mean']:+.4f} {[round(x,3) for x in r['composition']['ci95']]} "
                      f"di-dalam={r['within']['mean']:+.4f} {[round(x,3) for x in r['within']['ci95']]} total={r['total']:+.4f} "
                      f"kosong satu={r['n_one_empty']}/{r['n_pairs']} kosong dua={r['n_both_empty']}")
    print("== RHO KLASTER PER GENERATOR ==", {L: {k: (round(v['rho'], 3), [round(x, 3) for x in v['ci95_cluster']]) for k, v in d.items() if v}
                                            for L, d in out["rho_cluster_by_generator"].items()},
          "len gabungan", round(out["len_rho_cluster"]["rho"], 3), [round(x, 3) for x in out["len_rho_cluster"]["ci95_cluster"]])
    if "length_binary_baseline" in out:
        lb = out["length_binary_baseline"]
        print("== BASELINE PANJANG BINER ==", {k: lb[k] for k in ("random_multi_expected", "dense_multi", "hyde_global")})
        for r in lb["tests"]:
            print(f"  {r['desc']:<75} Δ={r['mean_diff']:+.3f} CI[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] p_holm={r['p_holm']:.4f}")
    print("== KELUARGA ==", out["families"], out["families_main"])
    print(f"[✓] Saved {V3_SUPP_STATS_FILE}")


if __name__ == "__main__":
    main()
