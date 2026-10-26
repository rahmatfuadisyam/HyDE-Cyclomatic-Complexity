"""
Generator naskah v3 — VERSI BAHASA INDONESIA, revisi pasca-review (jalur A: klaim defensibel).
SEMUA angka dihitung dari data/v3/eval_results_v3.json & stats_v3.json. Tidak ada angka diketik tangan.
Konvensi: desimal koma (0,080), pemisah ribuan titik (1.005), IK = interval kepercayaan.

Pemakaian:
  MAHYDE_LANG=id python 06_visualize_v3.py        # figure berlabel Indonesia -> figures/v3/id/
  python make_paper_v3_id.py [output.docx]        # default: paper_mahyde_v3_id.docx
"""
import json
import math
import os
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, RGBColor

from config_v3 import LLMS as CFG_LLMS
from config_v3 import (V3_RESULTS_FILE, V3_STATS_FILE, V3_HYPO_CACHE, V3_FIG_DIR, QUERIES_V2,
                       HYDE_ZS_PROMPT_TEMPLATE, SAMPLE_TEMPERATURE, MAX_TOKENS)

OUT = sys.argv[1] if len(sys.argv) > 1 else "paper_mahyde_v3_id.docx"
FIG_DIR = os.path.join(V3_FIG_DIR, "id") if os.path.isdir(os.path.join(V3_FIG_DIR, "id")) else V3_FIG_DIR

# =============================================================================== data
R = json.load(open(V3_RESULTS_FILE))
S = json.load(open(V3_STATS_FILE))
CACHE = json.load(open(V3_HYPO_CACHE))
MD, SM, HYP = R["metadata"], R["summary"], R["hypothetical_stats"]
SU = R.get("summary_untrunc", {})
TR = R.get("truncation", {})
K, NQ, NS = MD["k"], MD["n_queries"], MD["n_samples"]
LID = [m["id"] for m in MD["llms"]]
LAB = {**{m["id"]: m["label"] for m in CFG_LLMS}, **{m["id"]: m["label"] for m in MD["llms"]}}
DQ = {k: v for k, v in MD.get("data_quality", {}).items() if k in LID}
M = "p_at_k"
PN = {"fs": "few-shot", "zs": "zero-shot"}
N_HYP = sum(1 for e in CACHE.values() if e["llm"] in LID)
N_STOCH = sum(1 for k, e in CACHE.items() if e["llm"] in LID and k.split("|")[3] == "t07")
N_T0 = sum(1 for k, e in CACHE.items() if e["llm"] in LID and k.split("|")[3] == "t0")
NON_CC = S.get("sensitivity_cc_queries_excluded", [])
from config_v3 import V3_SUPP_STATS_FILE, QUERY_GROUPS, SESOI_P10, N_BOOT, REPLICATION_DOI, CHUNK_TOKENS  # noqa: E402
SUPP = json.load(open(V3_SUPP_STATS_FILE)) if os.path.exists(V3_SUPP_STATS_FILE) else {}
QG_OF = {q: g for g, v in QUERY_GROUPS.items() for q in v["qids"]}


# =============================================================================== format
def n(x, d=3, sign=False):
    return (f"{x:+.{d}f}" if sign else f"{x:.{d}f}").replace(".", ",")


def ribu(x):
    return f"{x:,}".replace(",", ".")


def pnum(p):
    return n(p, 4) if 0.0495 <= p < 0.0505 else n(p, 3)


def pval(p):
    return "< 0,001" if p < 0.001 else "= " + pnum(p)


def ci(lo, hi, d=3):
    return f"[{n(lo, d, True)}; {n(hi, d, True)}]"


def stat(r):
    return (f"Δ = {n(r['mean_diff'], 3, True)}, IK 95% {ci(*r['ci95'])}, "
            f"p_Holm {pval(r['p_holm'])}, r_rb = {n(r['r_rb'], 2, True)}")


def status(r):
    if r["significant"]:
        return "peningkatan terdeteksi" if r["mean_diff"] > 0 else "penurunan terdeteksi"
    return "perbedaan belum terdeteksi"


def fisher_ci(rho, nn):
    z, se = math.atanh(max(min(rho, 0.999), -0.999)), 1 / math.sqrt(max(nn - 3, 1))
    return math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)


def names(ids):
    lab = [LAB[i] for i in ids]
    if not lab:
        return "tidak ada"
    return lab[0] if len(lab) == 1 else ", ".join(lab[:-1]) + " dan " + lab[-1]


def row(family, a, b):
    return next(r for r in family if r["a"] == a and r["b"] == b)


def find(family, a_prefix, desc_contains):
    return next(r for r in family if r["a"] == a_prefix and desc_contains in r["desc"])


# =============================================================================== hasil kunci
shared = S["shared"][M]
r_rand = row(shared, "dense", "random")
r_bm25 = row(shared, "dense", "bm25")
r_df = row(shared, "dense_f", "dense")
r_df_rf = row(shared, "dense_f", "random_f")
r_rf = row(shared, "random_f", "random")
pool = {r["a"].split(":", 1)[1]: r for r in S["pooled"][M]}
AGG = S["aggregation"][M]
agg = {}
for p in ("fs", "zs"):
    agg[(p, "avg_vs_single")] = row(AGG, f"pooled:hyde_{p}_avg", f"pooled:hyde_{p}")
    if any(r["a"] == f"pooled:hyde_{p}_avgnq" for r in AGG):
        agg[(p, "nq_vs_single")] = row(AGG, f"pooled:hyde_{p}_avgnq", f"pooled:hyde_{p}")
        agg[(p, "avg_vs_nq")] = row(AGG, f"pooled:hyde_{p}_avg", f"pooled:hyde_{p}_avgnq")
INTER = S.get("interaction", {}).get(M, [])
inter_pool = {r["desc"].split(":")[0].replace("pooled ", ""): r for r in INTER if r["a"].startswith("pooled")}
EQ = S.get("equivalence", {}).get(M)
SENS = {r["a"]: r for r in S.get("sensitivity_cc_queries", {}).get(M, [])}
SENU = {r["a"]: r for r in S.get("sensitivity_untrunc", {}).get(M, [])}
cross = {r["mode"]: r for r in S["cross_llm"].get(M, [])}
dose = S["dose_response"].get(M, [])
dose_nq = S.get("dose_response_noquery", {}).get(M, [])
mech, mixed, perllm = S["mechanism"][M], S["mixed_model"][M], S["per_llm"]
gl = mech["generation_level"]


def per(L, suf, base):
    return row(perllm[L][M], f"{L}:hyde_{suf}", base)


def cfg_label(key):
    L, mode = key.split(":", 1)
    m = mode.replace("hyde_", "")
    parts = [PN[m[:2]], "multi-sampel" if "_avg" in m else "T = 0" if "_t0" in m else "tunggal"]
    if m.endswith("_f"):
        parts.append("+ filter")
    return f"{LAB[L]}, HyDE {' '.join(parts)}"


fsavg = {L: SM[f"{L}:hyde_fs_avg"][M] for L in LID}
top_v, low_v = max(fsavg.values()), min(fsavg.values())
best_list = [L for L in LID if abs(fsavg[L] - top_v) < 1e-9]
worst_list = [L for L in LID if abs(fsavg[L] - low_v) < 1e-9]
sig_any = [L for L in LID if any(per(L, s, "dense")["significant"] and per(L, s, "dense")["mean_diff"] > 0
                                for s in ("fs", "zs", "fs_avg", "zs_avg", "fs_t0", "zs_t0"))]
nonsig = [L for L in LID if L not in sig_any]
cands = [k for k in SM if ":" in k and "@" not in k]
best_val = max(SM[k][M] for k in cands)
best_cfgs = [k for k in cands if abs(SM[k][M] - best_val) < 1e-9]
best_unf = max(SM[k][M] for k in cands if not k.endswith("_f"))
pc, pl = gl["_pooled"]["cc"], gl["_pooled"]["len"]
pc_ci = fisher_ci(pc["rho"], pc["n"])
cell = mech["cell_level"]
neg_sig = [L for L in LID if "cc" in gl.get(L, {}) and gl[L]["cc"]["rho"] < 0 and gl[L]["cc"]["p"] < .05]
pos_sig = [L for L in LID if "cc" in gl.get(L, {}) and gl[L]["cc"]["rho"] > 0 and gl[L]["cc"]["p"] < .05]
cross_sig = [m for m, r in cross.items() if r["significant"]]
dose_sig = [d for d in dose if d["significant"]]
dose_nq_sig = [d for d in dose_nq if d["significant"]]
hyp_cc = {(L, p): HYP.get(f"{L}:{p}:t07", {}).get("cc_median") for L in LID for p in ("fs", "zs")}
most_complex = max(LID, key=lambda L: hyp_cc[(L, "fs")] or 0)
cc_none = {L: DQ[L]["cc_none"] for L in LID if DQ.get(L, {}).get("cc_none")}
s_fs = pool["hyde_fs"]["significant"] and pool["hyde_fs"]["mean_diff"] > 0
s_zs = pool["hyde_zs"]["significant"] and pool["hyde_zs"]["mean_diff"] > 0
s_avg = pool["hyde_fs_avg"]["significant"] and pool["hyde_fs_avg"]["mean_diff"] > 0
sens_ok = bool(SENS.get("pooled:hyde_fs_avg", {}).get("significant")) and \
    (not SENU or bool(SENU.get("pooled:hyde_fs_avg", {}).get("significant")))
in_f = pool["hyde_fs_avg_f"]["significant"] and pool["hyde_fs_avg_f"]["mean_diff"] > 0
sens_q = bool(SENS.get("pooled:hyde_fs_avg", {}).get("significant"))
sens_u = bool(SENU.get("pooled:hyde_fs_avg", {}).get("significant")) if SENU else None
r_u = SENU.get("pooled:hyde_fs_avg")
q_drives = (("fs", "avg_vs_nq") in agg and agg[("fs", "avg_vs_nq")]["significant"]
            and not agg[("fs", "nq_vs_single")]["significant"])


def sens_sentence():
    """Kalimat sensitivitas yang eksplisit per analisis."""
    a = ("Peningkatan ini bertahan ketika query yang predikatnya tidak setara dengan CC dikeluarkan"
         if sens_q else "Peningkatan ini tidak terdeteksi ketika query yang predikatnya tidak setara dengan CC dikeluarkan")
    if sens_u is None:
        return a + ". "
    if sens_u:
        return a + ", dan juga pada subkorpus tanpa fungsi yang terpotong oleh encoder. "
    return (a + "; namun pada subkorpus tanpa fungsi yang terpotong oleh encoder, yang hanya memuat "
            f"{TR['n_relevant_kept']} fungsi relevan, selisihnya mengecil menjadi {n(r_u['mean_diff'], 3, True)} "
            f"(IK 95% {ci(*r_u['ci95'])}) dan tidak terdeteksi. ")


def rand_phrase(r):
    if r["significant"] and r["mean_diff"] > 0:
        return "Dense lebih baik dari acak"
    if r["significant"]:
        return "Dense lebih buruk dari acak"
    if EQS and EQS.get("equivalent"):
        return "peningkatan atas acak belum terdeteksi"
    return "peningkatan atas acak belum terdeteksi, dan kesetaraan juga tidak dapat diklaim"


# ------------------------------------------------------------------ hasil pasca-review (05b)
FF = SUPP.get("fullfunc") or {}
FFR = FF.get("reps", {})
REP_LAB = {"first": "BGE, awal fungsi (≤ 512 token; utama)", "mean": f"BGE, seluruh token: rerata potongan {CHUNK_TOKENS} token",
           "max": f"BGE, seluruh token: skor maksimum potongan {CHUNK_TOKENS} token"}


def ff_row(rep, a):
    return next((r for r in FFR.get(rep, {}).get("tests", []) if r["a"] == a), None)


ff_main = {rep: ff_row(rep, "pooled:hyde_fs_avg") for rep in FFR if rep != "first"}
from config_v3 import V3_FULLFUNC_FILE  # noqa: E402
_FFJ = json.load(open(V3_FULLFUNC_FILE)) if os.path.exists(V3_FULLFUNC_FILE) else {}
FFE = _FFJ.get("representations", {})
FFM = _FFJ.get("manifest", {})
EXK = [k for k in FFR if k not in ("first", "mean", "max") and ff_row(k, "pooled:hyde_fs_avg")]
EX = EXK[0] if EXK else None
EXN = FFR[EX]["encoder"].split("/")[-1] if EX else ""
ex_fs = ff_row(EX, "pooled:hyde_fs_avg") if EX else None
ex_hold = bool(ex_fs and ex_fs["significant"] and ex_fs["mean_diff"] > 0)


def ex_sentence(lang="id"):
    if not EX:
        return ""
    if lang == "en":
        return (f"With a long-context code encoder ({EXN}, 8,192 tokens), the multi-sample few-shot effect was "
                + (f"larger (Δ = {ex_fs['mean_diff']:+.3f}). " if ex_hold else "not detected. "))
    return (f"Dengan encoder kode berkonteks panjang ({EXN}, 8.192 token), efek HyDE few-shot multi-sampel "
            + (f"lebih besar ({stat(ex_fs)}). " if ex_hold else f"tidak terdeteksi ({stat(ex_fs)}). "))
ff_whole = {k: v for k, v in ff_main.items() if k in ("mean", "max") and v}
ff_hold = any(v["significant"] and v["mean_diff"] > 0 for v in ff_whole.values())
ff_fail = bool(ff_whole) and not ff_hold


def ff_sentence():
    if not ff_whole:
        return ""
    det = "; ".join(f"{'rerata' if k == 'mean' else 'maksimum'} potongan: {stat(v)}" for k, v in ff_whole.items())
    if ff_hold and all(v["significant"] and v["mean_diff"] > 0 for v in ff_whole.values()):
        return (f"Ketika seluruh token fungsi dicakup melalui potongan {CHUNK_TOKENS} token pada korpus yang sama, "
                f"peningkatan HyDE few-shot multi-sampel tetap terdeteksi ({det}). ")
    if ff_hold:
        return (f"Ketika seluruh token fungsi dicakup melalui potongan {CHUNK_TOKENS} token pada korpus yang sama, "
                f"peningkatan HyDE few-shot multi-sampel hanya terdeteksi pada sebagian cara agregasi ({det}). ")
    return (f"Ketika seluruh token fungsi dicakup melalui potongan {CHUNK_TOKENS} token pada korpus yang sama, "
            f"peningkatan HyDE few-shot multi-sampel tidak lagi terdeteksi ({det}). ")


TRUNC_SENT = ("Hasil sensitif terhadap penghapusan fungsi panjang. Karena penghapusan tersebut juga mengubah komposisi "
              "korpus dan distribusi relevansi, analisis ini belum memisahkan pengaruh pemotongan dari perubahan kandidat "
              "retrieval. ")


def agg_sentence():
    if ("fs", "avg_vs_nq") not in agg:
        return ""
    a, b = agg[("fs", "avg_vs_nq")], agg[("fs", "nq_vs_single")]
    s = ""
    if a["significant"] and a["mean_diff"] > 0:
        s += (f"Pada konfigurasi few-shot dengan {'delapan' if NS == 8 else NS} hipotesis, penyertaan query memberikan "
              f"tambahan P@{K} sebesar {n(a['mean_diff'])}. ")
    else:
        s += f"Pada konfigurasi few-shot, tambahan dari penyertaan query belum terdeteksi ({stat(a)}). "
    s += ("Manfaat perataan hipotesis tanpa query belum terdeteksi. " if not b["significant"]
          else f"Perataan hipotesis tanpa query juga mengubah P@{K} ({stat(b)}). ")
    return s


COMP = SUPP.get("compliance", {})
comp = {(L, p): COMP.get(f"{L}:{p}:t07") for L in LID for p in ("fs", "zs")}
comp_vals = [v["pct_comply_all"] for v in comp.values() if v and v.get("pct_comply_all") is not None]
below = [(L, p) for (L, p), v in comp.items() if v and (v.get("cc_median") or 0) < MD["relevant_cc"]]
CR = SUPP.get("cc_rho_cluster")
rho_ci = CR["ci95_cluster"] if CR else None
rci = rho_ci if rho_ci else pc_ci
rci_lab = "IK 95% bootstrap klaster per query" if rho_ci else "IK 95%"
EQS = SUPP.get("equivalence_sesoi")
ABL = {r["desc"]: r for r in SUPP.get("ablation", [])}
DMP = SUPP.get("dose_means_pooled", {})
QGR = SUPP.get("query_groups", [])
FAMM, FAMS = SUPP.get("families_main", {}), SUPP.get("families", {})
CA = SUPP.get("compliance_assoc", {})
CBT = {x["rep"]: x for x in SUPP.get("compliance_binary_tests", [])}
LENB = SUPP.get("length_baselines") or {}
STRA = SUPP.get("strata") or {}
import config_v3 as _c3  # noqa: E402
REPLICATION_URL = getattr(_c3, "REPLICATION_URL", "")
EX_GEN = ({L: FFE[EX]["summary"][f"{L}:hyde_fs_avg"][M] for L in LID if f"{L}:hyde_fs_avg" in FFE.get(EX, {}).get("summary", {})}
          if EX else {})
REP_SHORT = {"first": "awal fungsi", "mean": "rerata potongan", "max": "maksimum potongan"}
TC = SUPP.get("topk_composition")
_ST = STRA.get("tests", [])
st_pos = [x for x in _ST if x["significant"] and x["mean_diff"] > 0]
st_neg = [x for x in _ST if x["significant"] and x["mean_diff"] < 0]
_sf = (STRA.get("summary") or {}).get("first", {})
S_MULTI, S_SINGLE = _sf.get("≥ 2 potongan"), _sf.get("1 potongan")
comp_shift = bool(TC and TC["test_share_hyde_vs_dense"]["p"] < .05 and TC["test_share_hyde_vs_dense"]["mean_diff"] > 0)


def comp_text(short=False):
    """Kalimat efek komposisi (review 3)."""
    if not (TC and S_MULTI):
        return ""
    d, h = TC["dense"]["share_multi"], TC["hyde_fs_avg_pooled"]["share_multi"]
    cr = TC["cell_rho_share_p10"]
    if short:
        return (f"pada encoder utama HyDE terutama menggeser sepuluh hasil teratas ke fungsi panjang "
                f"({n(d * 100, 1)}% menjadi {n(h * 100, 1)}% unit multi-potongan), tempat proporsi fungsi relevan jauh "
                f"lebih tinggi, dan proporsi ini berkorelasi dengan P@{K} antar-konfigurasi generator "
                f"(ρ = {n(cr['rho'], 2, True)}, n = {cr['n']})")
    return (f"Unit multi-potongan hanya {n(TC['multi_chunk_corpus_share'] * 100, 1)}% korpus tetapi memuat "
            f"{S_MULTI['n_relevant']} dari {MD['n_relevant']} unit relevan (proporsi relevan "
            f"{n(S_MULTI['random_p10'] * 100, 1)}%, dibanding {n(S_SINGLE['random_p10'] * 100, 1)}% pada unit satu "
            f"potongan). Di antara sepuluh hasil teratas pada encoder utama, proporsi unit multi-potongan adalah "
            f"{n(d * 100, 1)}% untuk Dense dan {n(h * 100, 1)}% untuk HyDE few-shot multi-sampel "
            f"(selisih {n(TC['test_share_hyde_vs_dense']['mean_diff'] * 100, 1, True)} poin persentase, IK 95% "
            f"{ci(TC['test_share_hyde_vs_dense']['ci95'][0] * 100, TC['test_share_hyde_vs_dense']['ci95'][1] * 100, 1)}, "
            f"Wilcoxon p {pval(TC['test_share_hyde_vs_dense']['p'])}). Antar-sel generator–prompt, proporsi ini berkorelasi dengan "
            f"P@{K} (ρ = {n(cr['rho'], 2, True)}, p {pval(cr['p'])}, n = {cr['n']}; deskriptif).")


def rep_name(k):
    return REP_SHORT.get(k, (FFR.get(k, {}).get("encoder") or k).split("/")[-1])


def comp_sentence(rep="first"):
    """Kepatuhan biner dan CC kontinu di antara hipotesis patuh (review 3)."""
    b, rc = CBT.get(rep), (CA.get(rep) or {}).get("rho_compliant")
    out = ""
    if b:
        out += (f"hipotesis yang memenuhi CC ≥ {MD['relevant_cc']} {('tidak menunjukkan P@' + str(K) + ' yang berbeda dari') if not b['significant'] else ('menghasilkan P@' + str(K) + (' lebih tinggi' if b['mean_diff'] > 0 else ' lebih rendah') + ' daripada')} "
                f"hipotesis yang tidak memenuhi di dalam sel yang sama ({stat(b)}; {b['n_cells_both']} sel)")
    if rc:
        out += ((", dan " if out else "") + "di antara hipotesis yang patuh, kenaikan CC "
                + ("tidak menunjukkan hubungan positif yang konsisten dengan P@" + str(K)
                   if not (rc["ci95_cluster"][0] > 0) else "berasosiasi positif dengan P@" + str(K))
                + f" (ρ = {n(rc['rho'], 3, True)}, IK 95% klaster {ci(*rc['ci95_cluster'])})")
    return out

# =============================================================================== pustaka
REFS = {
    "lewis2020": "P. Lewis et al., “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in Proc. NeurIPS, 2020.",
    "gao2023survey": "Y. Gao et al., “Retrieval-augmented generation for large language models: A survey,” arXiv:2312.10997, 2023.",
    "gao2023hyde": "L. Gao, X. Ma, J. Lin, and J. Callan, “Precise zero-shot dense retrieval without relevance labels,” in Proc. ACL, 2023, pp. 1762–1777.",
    "nogueira2019": "R. Nogueira, W. Yang, J. Lin, and K. Cho, “Document expansion by query prediction,” arXiv:1904.08375, 2019.",
    "wang2023": "L. Wang, N. Yang, and F. Wei, “Query2doc: Query expansion with large language models,” in Proc. EMNLP, 2023.",
    "li2024": "H. Li et al., “Rewriting the code: A simple method for large language model augmented code search,” in Proc. ACL, 2024.",
    "limpanukorn2025": "B. Limpanukorn, Y. Wang, Z. Patterson, P. Garg, M. K. Ramanathan, X. Ma, A. Deoras, and M. Kim, “Structural code search using natural language queries,” arXiv:2507.02107, 2025.",
    "geng2025": "J. Geng et al., “CoQuIR: A comprehensive benchmark for code quality-aware information retrieval,” arXiv:2506.11066, 2025.",
    "husain2019": "H. Husain et al., “CodeSearchNet challenge: Evaluating the state of semantic code search,” arXiv:1909.09436, 2019.",
    "feng2020": "Z. Feng et al., “CodeBERT: A pre-trained model for programming and natural languages,” in Findings of EMNLP, 2020.",
    "guo2021": "D. Guo et al., “GraphCodeBERT: Pre-training code representations with data flow,” in Proc. ICLR, 2021.",
    "guo2022": "D. Guo et al., “UniXcoder: Unified cross-modal pre-training for code representation,” in Proc. ACL, 2022.",
    "allamanis2018": "M. Allamanis, M. Brockschmidt, and M. Khademi, “Learning to represent programs with graphs,” in Proc. ICLR, 2018.",
    "mccabe1976": "T. J. McCabe, “A complexity measure,” IEEE Trans. Softw. Eng., vol. SE-2, no. 4, pp. 308–320, 1976.",
    "campbell2018": "G. A. Campbell, “Cognitive complexity: An overview and evaluation,” in Proc. Int. Conf. Technical Debt (TechDebt), 2018, pp. 57–58.",
    "cunningham1992": "W. Cunningham, “The WyCash portfolio management system,” in Addendum to Proc. OOPSLA, 1992.",
    "goodhue1995": "D. L. Goodhue and R. L. Thompson, “Task-technology fit and individual performance,” MIS Quarterly, vol. 19, no. 2, pp. 213–236, 1995.",
    "robertson2009": "S. Robertson and H. Zaragoza, “The probabilistic relevance framework: BM25 and beyond,” Found. Trends Inf. Retr., vol. 3, no. 4, pp. 333–389, 2009.",
    "xiao2023": "S. Xiao et al., “C-Pack: Packaged resources to advance general Chinese embedding,” arXiv:2309.07597, 2023.",
    "jarvelin2002": "K. Järvelin and J. Kekäläinen, “Cumulated gain-based evaluation of IR techniques,” ACM Trans. Inf. Syst., vol. 20, no. 4, pp. 422–446, 2002.",
    "radon": "M. Lacchia, “Radon: A Python tool that computes various metrics from the source code,” v6.0.1. [Online]. Tersedia: https://github.com/rubik/radon",
    "wilcoxon1945": "F. Wilcoxon, “Individual comparisons by ranking methods,” Biometrics Bulletin, vol. 1, no. 6, pp. 80–83, 1945.",
    "holm1979": "S. Holm, “A simple sequentially rejective multiple test procedure,” Scand. J. Statist., vol. 6, no. 2, pp. 65–70, 1979.",
    "kerby2014": "D. S. Kerby, “The simple difference formula: An approach to teaching nonparametric correlation,” Comprehensive Psychology, vol. 3, 2014.",
    "friedman1937": "M. Friedman, “The use of ranks to avoid the assumption of normality implicit in the analysis of variance,” J. Amer. Statist. Assoc., vol. 32, no. 200, pp. 675–701, 1937.",
    "page1963": "E. B. Page, “Ordered hypotheses for multiple treatments: A significance test for linear ranks,” J. Amer. Statist. Assoc., vol. 58, no. 301, pp. 216–230, 1963.",
    "seabold2010": "S. Seabold and J. Perktold, “Statsmodels: Econometric and statistical modeling with Python,” in Proc. 9th Python in Science Conf., 2010.",
    "arcuri2011": "A. Arcuri and L. Briand, “A practical guide for using statistical tests to assess randomized algorithms in software engineering,” in Proc. ICSE, 2011, pp. 1–10.",
    "lakens2017": "D. Lakens, “Equivalence tests: A practical primer for t tests, correlations, and meta-analyses,” Social Psychological and Personality Science, vol. 8, no. 4, pp. 355–362, 2017.",
    "hewitt2019": "J. Hewitt and P. Liang, “Designing and interpreting probes with control tasks,” in Proc. EMNLP-IJCNLP, 2019.",
    "gunther2023": "M. Günther et al., “Jina Embeddings 2: 8192-token general-purpose text embeddings for long documents,” arXiv:2310.19923, 2023.",
    "hevner2004": "A. R. Hevner, S. T. March, J. Park, and S. Ram, “Design science in information systems research,” MIS Quarterly, vol. 28, no. 1, pp. 75–105, 2004.",
}
_order = []


def c(*keys):
    out = []
    for k in keys:
        if k not in _order:
            _order.append(k)
        out.append(_order.index(k) + 1)
    return "[" + "], [".join(str(x) for x in out) + "]"


# =============================================================================== docx helpers
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.font.size = Pt(10.5)
st.paragraph_format.space_after = Pt(4)
for sname in ("Heading 1", "Heading 2", "Heading 3"):
    hs = doc.styles[sname]
    hs.font.name = "Times New Roman"
    hs.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    hs.font.color.rgb = RGBColor(0, 0, 0)
doc.styles["Heading 1"].font.size = Pt(12)
doc.styles["Heading 2"].font.size = Pt(11)
for sec in doc.sections:
    sec.left_margin = sec.right_margin = Inches(0.9)
    sec.top_margin = sec.bottom_margin = Inches(0.9)
_fig, _tab = [0], [0]


def P(text="", bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None):
    p = doc.add_paragraph()
    if text:
        r = p.add_run(text)
        r.bold, r.italic = bold, italic
        if size:
            r.font.size = Pt(size)
    p.alignment = align
    return p


def PR(parts):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for text, fmt in parts:
        r = p.add_run(text)
        r.bold, r.italic = "b" in fmt, "i" in fmt
    return p


def H(text, level=1):
    return doc.add_heading(text, level=level)


def B(text, lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if lead:
        p.add_run(lead).bold = True
    p.add_run(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def _border(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.first_child_found_in("w:tcBorders")
    if b is None:
        b = OxmlElement("w:tcBorders")
        tcPr.append(b)
    for edge, val in kw.items():
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), "6")
        el.set(qn("w:color"), "000000")
        b.append(el)


LEFT_COLS = ("Query", "Bukti", "Kajian", "Pertanyaan", "Temuan", "Status bukti", "Keluaran dan syarat lanjut",
             "Pembedaan", "Operasionalisasi", "Diukur di studi ini?", "Representasi", "Intervensi", "Kebutuhan",
             "Ciri rumusan", "Keluarga uji", "Suku model", "Kontras")


def T(header, rows, caption, widths, bold_rows=(), font=8.5, note=None):
    _tab[0] += 1
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Tabel {_tab[0]}. ")
    r.bold = True
    r.font.size = Pt(9)
    cap.add_run(caption).font.size = Pt(9)
    cap.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, w in enumerate(widths):
        t.columns[j].width = Inches(w)
    for i, rr in enumerate([header] + rows):
        for j, v in enumerate(rr):
            cl = t.rows[i].cells[j]
            cl.width = Inches(widths[j])
            cl.text = ""
            run = cl.paragraphs[0].add_run(str(v))
            run.font.size = Pt(font)
            run.bold = i == 0 or (i - 1) in bold_rows
            left = j == 0 or header[j] in LEFT_COLS
            cl.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if left else WD_ALIGN_PARAGRAPH.CENTER
            if i < len(rows):
                cl.paragraphs[0].paragraph_format.keep_with_next = True
            if i == 0:
                _border(cl, top="single", bottom="single")
            elif i == len(rows):
                _border(cl, bottom="single")
    for i, row_ in enumerate(t.rows):
        trPr = row_._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        trPr.append(cs)
        if i == 0:
            th = OxmlElement("w:tblHeader")
            trPr.append(th)
    if note:
        doc.add_paragraph().add_run(note).font.size = Pt(8)
    doc.add_paragraph()
    return _tab[0]


def F(fname, caption, width=6.4):
    _fig[0] += 1
    path = os.path.join(FIG_DIR, fname)
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.paragraphs[-1].paragraph_format.keep_with_next = True
    else:
        P(f"[gambar tidak ditemukan: {path}]", italic=True)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Gambar {_fig[0]}. ")
    r.bold = True
    r.font.size = Pt(9)
    cap.add_run(caption).font.size = Pt(9)
    return _fig[0]


def dag(r):
    return "†" if r["significant"] else ""


# =============================================================================== judul & abstrak
P("Manfaat dan Batas HyDE pada Pencarian Fungsi Python Berdasarkan Cyclomatic Complexity",
  bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=15)
P("Benefits and Limits of HyDE for Retrieving Python Functions by Cyclomatic Complexity",
  italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=11)
P("Rahmat", align=WD_ALIGN_PARAGRAPH.CENTER, size=11)
P("Program Studi Ilmu Komputer, Fakultas MIPA, Universitas Pancasakti Makassar, Indonesia\n"
  "rahmatfuadisyam@gmail.com", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=9.5)

trunc_rel = TR.get("n_relevant_truncated")
H("Abstrak", 1)
P(
    "Hypothetical Document Embeddings (HyDE) mengganti query pendek dengan dokumen hipotetis buatan large language "
    "model (LLM) sebelum pencarian berbasis embedding. Kami mengukur manfaat dan batas HyDE pada benchmark diagnostik "
    "yang relevansinya ditentukan secara pasti oleh cyclomatic complexity (CC): "
    f"{ribu(MD['n_corpus'])} fungsi produksi dari tiga pustaka Python, {NQ} query dalam {len(QUERY_GROUPS)} kelompok "
    f"rumusan, {len(LID)} konfigurasi generator, dua rancangan prompt, dan {ribu(N_HYP)} respons generasi "
    f"({ribu(N_STOCH)} stokastik + {ribu(N_T0)} temperature 0). Embedding query langsung mencapai P@{K} = "
    f"{n(SM['dense'][M])} dibanding {n(SM['random'][M])} untuk peringkat acak ({rand_phrase(r_rand)}). Pada "
    "konfigurasi dasar (BGE, 512 token pertama), HyDE few-shot multi-sampel meningkatkan P@" + str(K)
    + f" ({stat(pool['hyde_fs_avg'])})"
    + ((", dan peningkatan agregat ini tetap terdeteksi ketika seluruh token fungsi dicakup melalui potongan"
        + (f" serta dengan encoder kode berkonteks panjang (Δ = {n(ex_fs['mean_diff'], 3, True)})" if ex_hold else "")
        + ". ") if ff_hold else ". ")
    + ((f"Namun, di dalam strata jumlah potongan yang sama, keunggulan HyDE atas Dense hanya terdeteksi pada "
        f"{len(st_pos)} dari {len(_ST)} perbandingan" + (f" dan berbalik pada {len(st_neg)}" if st_neg else "")
        + "; " + comp_text(short=True) + ". ") if (_ST and comp_shift) else "")
    + ((f"Urutan konfigurasi generator juga berubah antar-encoder (misalnya {LAB[worst_list[0]]}: {n(low_v)} dengan BGE, "
        f"{n(EX_GEN[worst_list[0]])} dengan {EXN}). ") if EX_GEN and worst_list else "")
    + ((f"Keunggulan formulasi multi-sampel berasal dari penyertaan query "
        f"({n(agg[('fs', 'avg_vs_nq')]['mean_diff'], 3, True)}), bukan dari jumlah hipotesis. ") if q_drives else "")
    + f"CC hipotesis tidak berfungsi sebagai proksi keberhasilan retrieval pada konfigurasi dasar, sedangkan metadata CC "
    f"memberi manfaat langsung (Dense + filter: {n(SM['dense_f'][M])}; pengurutan CC: {n(SM['oracle'][M])}). "
    + ("Manfaat HyDE untuk predikat struktural dengan demikian bergantung pada konfigurasi dan sebagian besar mengikuti "
       "preferensi terhadap fungsi panjang, sehingga belum dapat ditafsirkan sebagai pembacaan struktur. "
       if comp_shift else "")
    + "Benchmark, seluruh teks hipotesis, dan skrip analisis "
    + ("dirilis sebagai paket replikasi." if REPLICATION_URL else "disiapkan sebagai paket replikasi.")
)
PR([("Kata kunci: ", "b"), ("hypothetical document embeddings; pencarian kode; retrieval-augmented generation; "
                            "metrik perangkat lunak; cyclomatic complexity; studi empiris", "i")])
H("Abstract", 1)


def en(x, d=3, sign=False):
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


P(
    "Hypothetical Document Embeddings (HyDE) replace a short query with an LLM-written hypothetical document before "
    "dense retrieval. We measure the benefits and limits of HyDE on a diagnostic benchmark whose relevance is defined "
    f"exactly by cyclomatic complexity (CC): {MD['n_corpus']:,} production functions from three Python libraries, "
    f"{NQ} queries in {len(QUERY_GROUPS)} wording groups, {len(LID)} generator configurations, two prompt designs and "
    f"{N_HYP:,} generation responses ({N_STOCH:,} stochastic + {N_T0:,} at temperature 0). Direct query embedding "
    f"reached P@{K} = {en(SM['dense'][M])} versus {en(SM['random'][M])} for a random ranking (no improvement detected; "
    f"equivalence not established). In the base configuration (BGE, first 512 tokens), multi-sample few-shot HyDE "
    f"improved P@{K} (Δ = {en(pool['hyde_fs_avg']['mean_diff'], 3, True)})"
    + ((", and this aggregate gain persisted when all tokens were covered through chunks"
        + (f" and with a long-context code encoder (Δ = {en(ex_fs['mean_diff'], 3, True)})" if ex_hold else "") + ". ")
       if ff_hold else ". ")
    + ((f"Within strata of equal chunk count, however, HyDE outperformed dense retrieval in only {len(st_pos)} of "
        f"{len(_ST)} comparisons; with the base encoder HyDE mainly shifted the top ten toward long functions "
        f"({en(TC['dense']['share_multi'] * 100, 1)}% to {en(TC['hyde_fs_avg_pooled']['share_multi'] * 100, 1)}% "
        f"multi-chunk units), where relevant functions are far more frequent, and this share tracked P@{K} across "
        f"generator configurations (ρ = {en(TC['cell_rho_share_p10']['rho'], 2, True)}). ") if (_ST and comp_shift) else "")
    + ((f"Generator rankings also changed across encoders ({LAB[worst_list[0]]}: {en(low_v)} with BGE, "
        f"{en(EX_GEN[worst_list[0]])} with {EXN}). ") if EX_GEN and worst_list else "")
    + ((f"The multi-sample advantage came from including the query "
        f"({en(agg[('fs', 'avg_vs_nq')]['mean_diff'], 3, True)}), not from the number of hypotheses. ") if q_drives else "")
    + f"Hypothesis CC was not a proxy for retrieval success in the base configuration, whereas CC metadata helped "
    f"directly (Dense + filter: {en(SM['dense_f'][M])}; sorting by CC: {en(SM['oracle'][M])}). "
    + ("The benefit of HyDE for a structural predicate is therefore configuration-dependent and largely follows a "
       "preference for long functions; it cannot yet be read as structural matching. " if comp_shift else "")
    + "The benchmark, all hypothesis texts and analysis scripts are "
    + ("released." if REPLICATION_URL else "prepared as a replication package."), italic=True)
PR([("Keywords: ", "b"), ("hypothetical document embeddings; code search; retrieval-augmented generation; software "
                          "metrics; cyclomatic complexity; empirical study", "i")])

# =============================================================================== 1 pendahuluan
H("1. Pendahuluan", 1)
P("Retrieval-augmented generation (RAG) menambatkan large language model (LLM) pada pengetahuan eksternal dengan "
  f"mengambil konteks yang relevan sebelum generasi {c('lewis2020', 'gao2023survey')}. Dalam rekayasa perangkat lunak, "
  "asisten pemrograman mengambil potongan kode dari vector store untuk menjelaskan, menguji, atau merefaktor sistem. "
  "Sebagian kebutuhan pengembang bersifat struktural, misalnya menemukan fungsi dengan percabangan berlebihan sebagai "
  f"kandidat refaktor atau titik rawan utang teknis {c('cunningham1992')}. Cyclomatic complexity (CC) "
  f"{c('mccabe1976')} adalah salah satu operasionalisasi yang paling sering dipakai untuk sifat ini, walaupun CC "
  "tidak identik dengan konsep seperti “sulit diuji” atau “prioritas refaktor”.")
P("Jika kebutuhannya semata-mata mencari CC tinggi dan CC tersedia sebagai metadata, cara terbaik jelas adalah "
  "menghitung CC lalu mengurutkannya. Penelitian ini tidak mengusulkan HyDE sebagai pengganti cara tersebut. "
  "Kami memakai kebutuhan CC murni sebagai benchmark diagnostik: karena relevansi ditentukan secara pasti, kita "
  "dapat mengukur seberapa jauh generasi di sisi query mampu membawa sifat struktural ke dalam retrieval berbasis "
  "embedding, kondisi yang dihadapi pipeline RAG yang hanya mengindeks teks atau agen yang hanya menerima maksud "
  "dalam bahasa alami. Kebutuhan yang lebih realistis, yaitu kebutuhan campuran semantik dan struktural (misalnya "
  "“fungsi autentikasi dengan banyak metode login dan percabangan tinggi”), memerlukan penilaian relevansi "
  "oleh ahli dan menjadi sasaran studi lanjutan (Bagian 8).")
P(f"Hypothetical Document Embeddings (HyDE) {c('gao2023hyde')} meminta LLM menulis dokumen hipotetis yang memenuhi "
  "query, lalu memakai embedding dokumen tersebut sebagai vektor pencarian. Untuk kode, LLM dapat menulis fungsi "
  f"hipotetis yang sengaja kompleks. Asumsi yang mendasarinya adalah bahwa sifat penentu relevansi dari dokumen "
  "hipotetis tetap terbawa dalam proses retrieval. Untuk kebutuhan struktural, asumsi ini perlu diuji, karena "
  "kompleksitas alur kendali bukan hal yang secara langsung dibicarakan oleh token-token kode.")
P("Penelitian ini menjawab empat pertanyaan:")
B(" Seberapa besar HyDE meningkatkan retrieval fungsi ber-CC tinggi dibanding embedding query langsung, dan apakah "
  "hasilnya bertahan terhadap kelompok rumusan query, cara fungsi panjang direpresentasikan, dan pilihan encoder?", "RQ1.")
B(" Bagaimana faktor generasi, yaitu konfigurasi generator (eksploratif), rancangan prompt, penyertaan vektor query, "
  "dan jumlah sampel, memengaruhi manfaat tersebut?", "RQ2.")
B(" Seberapa patuh generator terhadap predikat CC, dan apakah CC kode hipotetis dapat berfungsi sebagai proksi "
  "keberhasilan retrieval yang dihasilkannya?", "RQ3.")
B(" Bagaimana HyDE berhubungan dengan metadata CC, termasuk apakah terdapat interaksi antara HyDE dan filter?", "RQ4.")
if EX and ex_hold:
    P("Temuan utama kami ada dua. Pertama, manfaat HyDE dan urutan konfigurasi generator bergantung pada encoder dan "
      "cara fungsi direpresentasikan: konfigurasi generator yang paling lemah dengan satu encoder dapat menjadi yang "
      "terkuat dengan encoder lain."
      + (" Kedua, pada encoder utama sebagian besar peningkatan HyDE berasal dari pergeseran hasil teratas ke fungsi "
         "yang lebih panjang, tempat fungsi relevan lebih banyak; di antara fungsi dengan jumlah potongan yang sama, "
         "keunggulan HyDE umumnya tidak terdeteksi." if comp_shift else "")
      + " Bukti ini bersifat diagnostik; mekanisme penyebabnya belum diidentifikasi.")
P("Kontribusi penelitian ini ada tiga: (i) benchmark diagnostik beserta materi replikasinya, yaitu korpus pada "
  f"commit tetap, {ribu(N_HYP)} respons generasi beserta metadatanya, hasil per query, dan skrip analisis; (ii) "
  "estimasi manfaat HyDE dengan ukuran efek dan interval kepercayaan, beserta batasnya: kepekaan terhadap pemilihan "
  "query, terhadap cara fungsi panjang direpresentasikan, dan terhadap pasangan generator–encoder; dan (iii) dekomposisi "
  "formulasi multi-sampel menjadi efek penyertaan query dan efek jumlah hipotesis. Pengujian mekanisme dan kebutuhan "
  "campuran semantik–struktural kami tempatkan sebagai agenda penelitian (Bagian 8).")

# =============================================================================== 2 literatur
H("2. Penelitian Terkait", 1)
H("2.1 Ekspansi query dan dokumen dengan model bahasa", 2)
P(f"doc2query {c('nogueira2019')} menambahkan query hasil prediksi ke dokumen sebelum pengindeksan. Query2doc "
  f"{c('wang2023')} menggabungkan dokumen semu hasil LLM dengan query. HyDE {c('gao2023hyde')} meng-embed N dokumen "
  "hipotetis dan merata-ratakannya, dengan dua formulasi: tanpa vektor query dan dengan vektor query diperlakukan "
  "sebagai salah satu hipotesis. Gao dkk. juga telah membandingkan beberapa model generator (InstructGPT, Cohere, "
  "dan FLAN-T5) dan menemukan bahwa model yang lebih besar memberi peningkatan lebih besar. Ketergantungan pada "
  "generator dengan demikian bukan temuan baru; yang kami uji adalah apakah ketergantungan itu juga muncul ketika "
  "kriteria relevansinya struktural.")
H("2.2 Retrieval kode berbantuan LLM", 2)
P(f"Pencarian kode memetakan query bahasa alami ke kode {c('husain2019')} dengan encoder pralatih seperti CodeBERT "
  f"{c('feng2020')}, GraphCodeBERT {c('guo2021')}, dan UniXcoder {c('guo2022')}; representasi berbasis graf "
  f"{c('allamanis2018')} mengodekan ketergantungan kendali dan data secara eksplisit. ReCo {c('li2024')} memperluas "
  "kerangka generation-augmented retrieval: selain membuat contoh kode dari query, ReCo menulis ulang kode di dalam "
  "codebase untuk menormalkan gaya, karena perbedaan gaya antara kode buatan LLM dan kode asli menurunkan kemiripan "
  "embedding. Temuan ini relevan sebagai penjelasan alternatif bagi hasil kami: perbedaan antar-generator dapat "
  "berasal dari keselarasan gaya kode, bukan dari sifat struktural.")
H("2.3 Pencarian kode berbasis struktur dan kualitas", 2)
P(f"Limpanukorn dkk. {c('limpanukorn2025')} menerjemahkan query bahasa alami menjadi query struktural (Semgrep dan "
  "GQL) dengan bantuan LLM, sehingga predikat struktural dieksekusi secara eksplisit alih-alih dicocokkan lewat "
  f"embedding. CoQuIR {c('geng2025')} mengevaluasi apakah sistem retrieval kode memperhitungkan kualitas "
  "(ketepatan, efisiensi, keamanan, dan keterpeliharaan) dan menemukan bahwa retriever terbaik pun sering gagal "
  "membedakan kode berkualitas rendah. Kedua karya ini menunjukkan bahwa retrieval berbasis struktur dan kualitas "
  "sudah menjadi agenda riset; kontribusi kami lebih sempit, yaitu pengukuran terkendali atas HyDE pada predikat "
  f"metrik numerik. CC dihitung dengan radon {c('radon')}; cognitive complexity {c('campbell2018')} adalah "
  "alternatif yang lebih menekankan nesting.")
H("2.4 Posisi penelitian", 2)
T(["Kajian", "Kebutuhan", "Representasi", "Intervensi", "Bukti", "Kondisi batas yang diuji"],
  [["doc2query " + c("nogueira2019"), "Topikal", "Teks", "Ekspansi dokumen", "Akurasi retrieval", "—"],
   ["Query2doc " + c("wang2023"), "Topikal", "Teks (sparse/dense)", "Ekspansi query", "Akurasi retrieval", "Ukuran model"],
   ["HyDE " + c("gao2023hyde"), "Topikal", "Teks (dense)", "Ekspansi query (N dokumen)", "Akurasi retrieval", "Generator"],
   ["ReCo " + c("li2024"), "Semantik", "Kode", "Ekspansi query + normalisasi korpus", "Akurasi retrieval", "Gaya kode"],
   ["Structural Code Search " + c("limpanukorn2025"), "Pola sintaksis", "Query struktural (Semgrep/GQL)",
    "Penerjemahan predikat", "Presisi/recall", "—"],
   ["CoQuIR " + c("geng2025"), "Semantik + kualitas", "Teks/kode", "Benchmark + pelatihan", "Akurasi berbasis kualitas", "Dimensi kualitas"],
   ["Penelitian ini", "Metrik numerik (CC)", "Teks (dense) + metadata", "Ekspansi query + filter",
    "Akurasi retrieval", "Generator, prompt, N, query, pemotongan"]],
  "Matriks literatur menurut kebutuhan, representasi, intervensi, jenis bukti, dan kondisi batas.",
  widths=[1.3, 0.9, 1.0, 1.1, 0.9, 1.2], bold_rows=(6,), font=7.5)

# =============================================================================== 3 kerangka
H("3. Kerangka Konseptual", 1)
P(f"Kami memakai task–technology fit {c('goodhue1995')} sebagai kerangka awal, bukan sebagai teori yang diuji "
  "secara penuh. Gagasannya: manfaat sebuah teknologi bergantung pada kesesuaian antara karakteristik tugas dan "
  "kemampuan teknologi. Di tingkat artefak retrieval, tugas adalah jenis kebutuhan informasi (di sini: predikat CC), "
  "sedangkan teknologi adalah representasi indeks beserta intervensi di sisi query. Agar argumen tidak melingkar, "
  "kesesuaian tidak didefinisikan melalui P@K. Kami memisahkan empat konstruk (Tabel 2); penelitian ini hanya "
  "mengukur konstruk pertama dan keempat secara langsung." .replace("(Tabel 2)", f"(Tabel {_tab[0] + 1})"))
T(["Konstruk", "Operasionalisasi", "Diukur di studi ini?"],
  [["Ketepatan generasi", "Apakah kode hipotetis memenuhi predikat query (CC hipotesis ≥ ambang)?", "Ya"],
   ["Kepekaan representasi", "Apakah perubahan struktur kode menghasilkan perubahan embedding yang terukur?",
    "Tidak (Studi lanjutan 1)"],
   ["Kesesuaian pemeringkatan", "Apakah perubahan embedding menaikkan peringkat dokumen relevan?",
    "Tidak langsung (Studi lanjutan 1)"],
   ["Hasil retrieval", f"P@{K}, nDCG@{K}, MRR, kepatuhan terhadap predikat", "Ya"]],
  "Konstruk dalam rantai generasi–representasi–pemeringkatan.", widths=[1.5, 3.6, 1.4], font=8)
P("Dari kerangka ini kami menurunkan ekspektasi, bukan klaim kausal: (a) bila representasi kurang peka terhadap "
  "struktur, embedding query langsung tidak akan jauh lebih baik dari acak; (b) HyDE dapat membantu bila dokumen "
  "hipotetis membawa petunjuk yang berkorelasi dengan relevansi, baik struktural maupun leksikal atau gaya; (c) bila "
  "petunjuk itu bersifat struktural, hipotesis yang memenuhi predikat diharapkan lebih sering menghasilkan retrieval "
  "yang berhasil daripada yang tidak memenuhi; karena relevansi didefinisikan dengan ambang, hubungan ini tidak harus "
  "monoton terhadap besarnya CC. "
  "Perbedaan antar-generator diperlakukan sebagai pertanyaan eksploratif.")

# =============================================================================== 4 metode
H("4. Metode", 1)
H("4.1 Korpus", 2)
pr = MD["per_repo"]
P("Requests, Flask, dan FastAPI dikloning pada commit tetap dan diurai dengan radon. Hanya kode produksi yang "
  "dipertahankan (pengujian, dokumentasi, tutorial, contoh, dan skrip dikeluarkan). Unit retrieval adalah fungsi "
  f"atau method; blok kelas dibuang dan duplikat identik dihapus (Tabel {_tab[0] + 1}).")
T(["Repositori", "Commit", "Unit", "Rerata CC", f"CC ≥ {MD['relevant_cc']}"],
  [[r.capitalize(), MD["repos"][r]["commit"][:10], ribu(pr[r]["n"]), n(pr[r]["mean_cc"], 2), pr[r]["n_relevant"]]
   for r in sorted(pr)] + [["Total", "—", ribu(MD["n_corpus"]), n(MD["corpus_mean_cc"], 2), MD["n_relevant"]]],
  "Korpus (hanya kode produksi).", widths=[1.3, 1.1, 0.8, 0.8, 0.9], bold_rows=(len(pr),))
rd = MD["rank_dist"]
P(f"Distribusi CC sangat menceng (median {MD['corpus_median_cc']:.0f}, maksimum {MD['corpus_max_cc']}); peringkat "
  f"radon A–F berisi {rd['A']}, {rd['B']}, {rd['C']}, {rd['D']}, {rd['E']}, dan {rd['F']} unit. Sebanyak "
  f"{MD['n_relevant']} unit ({n(MD['n_relevant'] / MD['n_corpus'] * 100, 1)}%) relevan.")
if TR:
    P(f"Encoder memiliki jendela {TR['max_seq_length']} token. Menurut tokenizer encoder, {TR['n_truncated']} unit "
      f"melebihi jendela tersebut, termasuk {TR['n_relevant_truncated']} dari {TR['n_relevant']} unit relevan; median "
      f"panjang unit relevan adalah {TR['median_tokens_relevant']:.0f} token, dibanding "
      f"{TR['median_tokens_nonrelevant']:.0f} token untuk unit lain. Dengan kata lain, sebagian besar fungsi yang "
      "paling kompleks tidak terbaca utuh oleh encoder; karena itu kami menambahkan analisis sensitivitas pada "
      "subkorpus tanpa unit yang terpotong (Bagian 4.6).")
H("4.2 Query, relevansi, dan metrik", 2)
P(f"Kami menyusun {NQ} query yang menyatakan kebutuhan kompleksitas percabangan dengan rumusan berbeda (Lampiran A); "
  f"lima berasal dari studi pilot. Unit relevan bila CC ≥ {MD['relevant_cc']}. Karena relevansi sama untuk "
  "semua query, ke-20 query adalah parafrasa dari satu kebutuhan substantif; desain ini mengukur kepekaan "
  "terhadap perumusan bahasa dan tidak mewakili banyak kebutuhan independen. Query dikelompokkan menurut cara "
  f"rumusannya menunjuk kompleksitas (Tabel {_tab[0] + 1}). Kelompok terakhir menyebut predikat yang tidak setara "
  "dengan CC (jumlah parameter, fungsi yang melakukan terlalu banyak hal, kedalaman indentasi); hasil utama karena itu "
  "juga diuji ulang tanpa kelompok ini. Pengelompokan ditetapkan setelah review pertama, sehingga analisis per "
  "kelompok bersifat eksploratif.")
tqg = T(["Kelompok", "Ciri rumusan", "Query", "n"],
  [[g, v["label"], ", ".join(v["qids"]), len(v["qids"])] for g, v in QUERY_GROUPS.items()],
  "Kelompok rumusan query (teks lengkap di Lampiran A).", widths=[1.1, 3.2, 1.8, 0.4], font=8)
P(f"Gain bertingkat nDCG {c('jarvelin2002')} adalah 2^g − 1 dengan g = 0 untuk peringkat A–B dan g = 1–4 untuk "
  f"peringkat C–F. Kami melaporkan P@{K} (utama), nDCG@{K}, dan MRR. MRR adalah rerata reciprocal rank unit relevan "
  "pertama atas seluruh peringkat, tanpa batas kedalaman. Ekspektasi acak dihitung secara analitik: untuk N unit "
  "dengan R unit relevan, posisi relevan pertama X pada permutasi seragam memenuhi P(X = k) = C(N − k, R − 1)/C(N, R), "
  f"sehingga E[1/X] = Σₖ P(X = k)/k; nilainya {n(SM['random']['rr'])} untuk korpus penuh dan "
  f"{n(SM['random_f']['rr'])} untuk kumpulan yang difilter. P@{K} dan nDCG@{K} acak dihitung dari proporsi dan rerata "
  "gain.")
H("4.3 Konfigurasi retrieval", 2)
B(f" ekspektasi analitik peringkat acak atas seluruh korpus (P@{K} = {n(SM['random'][M], 4)}) dan atas kumpulan "
  f"yang difilter (acak + filter, P@{K} = {n(SM['random_f'][M], 4)}); BM25 {c('robertson2009')}.", "Baseline:")
B(f" query (dengan instruksi retrieval BGE) di-embed dengan {MD['embedding_model']} {c('xiao2023')}; pencarian "
  "kosinus eksak.", "Dense:")
B(f" satu fungsi hipotetis menggantikan vektor query; metrik dirata-ratakan atas {NS} generasi.", "HyDE tunggal:")
B(f" rerata ternormalisasi dari vektor query dan {NS} vektor hipotesis (Pers. 8 Gao dkk.).", "HyDE multi-sampel (+query):")
B(f" rerata {NS} vektor hipotesis tanpa vektor query (Pers. 7 Gao dkk.); dipakai untuk memisahkan efek jumlah "
  "sampel dari efek penyertaan query.", "HyDE multi-sampel (tanpa query):")
B(f" rerata ternormalisasi vektor query dan satu vektor hipotesis; metrik dirata-ratakan atas {NS} hipotesis. "
  "Kontras dengan HyDE tunggal memisahkan efek penyertaan query tanpa perataan banyak hipotesis.",
  "HyDE query + satu hipotesis (pasca-review):")
B(f" vektor 0,5·q + 0,5·h̄ₙ, dengan h̄ₙ rerata ternormalisasi dari n hipotesis (n ∈ {{{', '.join(str(x) for x in MD['dose_n'])}}}). "
  f"Setiap titik n dirata-ratakan atas semua kombinasi n dari {NS} hipotesis (n = 1: {NS}; n = 2: {NS * (NS - 1) // 2}; "
  f"n = 4: 70 kombinasi), demikian pula kurva jumlah hipotesis dengan dan tanpa query, sehingga titik n = 1 identik "
  "dengan kondisi query + satu hipotesis dan tidak bergantung pada urutan generasi. "
  "Pada Pers. 8, bobot query adalah 1/(n + 1) sehingga turun ketika n bertambah; formulasi ini menahan bobot query "
  "tetap agar efek jumlah hipotesis tidak tercampur dengan perubahan bobot query.",
  "HyDE bobot query tetap (pasca-review):")
B(" satu hipotesis pada temperature 0.", "HyDE T = 0:")
B(f" konfigurasi mana pun yang dibatasi pada unit dengan CC ≥ {MD['filter_cc']} ({MD['n_filter_pool']} unit).",
  "+ filter:")
B(" pengurutan berdasarkan CC; baseline operasional yang sah karena relevansi ditentukan oleh CC.", "Pengurutan CC:")
H("4.4 Generator dan prompt", 2)
P("Prompt few-shot menampilkan satu contoh fungsi bisnis bernesting dalam dan mensyaratkan minimal empat tingkat "
  "nesting, perulangan, dan penanganan eksepsi. Prompt ini sekaligus memuat contoh dan membatasi struktur, sehingga "
  "kedua efek tersebut belum terpisah dalam desain ini. Prompt zero-shot tidak membatasi apa pun: “"
  + HYDE_ZS_PROMPT_TEMPLATE.split("\n")[0] + f"”. Untuk setiap generator, query, dan prompt diambil {NS} sampel "
  f"pada temperature {n(SAMPLE_TEMPERATURE, 1)} (seed s + i) dan satu sampel pada temperature 0. Pagar markdown "
  "dibuang; CC diukur dengan radon, dan untuk keluaran yang terpotong batas token, baris akhir dipangkas hingga kode "
  "dapat diurai (hasilnya batas bawah).")
gen_rows = [[LAB[g], ", ".join(q["returned_models"]), f"{q['n'] - q['empty']}/{q['n']}", q["parsed"], q["trimmed"],
             q["cc_none"], q["truncated_output"]] for g, q in DQ.items()]
T(["Generator", "ID model yang dikembalikan", "Tidak kosong", "Terurai", "Dipangkas", "CC tak terukur", "Terpotong"],
  gen_rows, "Konfigurasi generator dan kualitas keluaran. Label generator adalah label layanan; identitas model di "
            "baliknya tidak dapat diverifikasi secara independen (Bagian 7).",
  widths=[1.4, 1.7, 0.7, 0.6, 0.7, 0.7, 0.7], font=7.5)
if cc_none:
    P("Keluaran yang CC-nya tidak dapat diukur (" + "; ".join(f"{LAB[L]}: {v}" for L, v in cc_none.items())
      + ") seluruhnya terpotong batas token dan tetap gagal diurai setelah pemangkasan. Keluaran ini tetap dipakai "
        "dalam retrieval (teksnya tetap di-embed) tetapi tidak disertakan dalam analisis CC hipotesis; denominator "
        "setiap analisis CC dilaporkan bersama hasilnya.")
H("4.5 Analisis statistik", 2)
P(f"Unit analisis adalah query (n = {NQ}). Untuk konfigurasi multi-sampel, metrik dihitung sekali per query; untuk "
  f"HyDE tunggal, metrik dirata-ratakan atas {NS} hipotesis; untuk hasil gabungan, metrik per query dirata-ratakan "
  f"lagi atas {len(LID)} generator. Dengan demikian setiap uji memakai {NQ} pasangan dan tidak memperlakukan "
  "hipotesis atau generator sebagai pengamatan independen. Kami memakai uji peringkat bertanda Wilcoxon dua sisi "
  f"{c('wilcoxon1945')}, korelasi rank-biserial r_rb {c('kerby2014')} sebagai ukuran efek, dan IK 95% bootstrap "
  f"persentil untuk selisih rerata dengan {ribu(N_BOOT)} resampling query (seed 42) {c('arcuri2011')}. Koreksi Holm "
  f"{c('holm1979')} diterapkan di dalam setiap keluarga yang ditetapkan sebelum analisis (Tabel {_tab[0] + 1}); "
  "analisis yang ditambahkan setelah review pertama diberi label pasca-review dan dikoreksi di dalam keluarganya "
  "sendiri. Hasil yang tidak signifikan dilaporkan sebagai “perbedaan belum terdeteksi”, bukan sebagai kesetaraan. "
  f"Untuk klaim kesetaraan kami memakai batas efek terkecil yang bermakna (SESOI) ±{n(SESOI_P10, 2)} pada P@{K}, "
  "yaitu setengah dokumen relevan di antara sepuluh hasil teratas; selisih yang lebih kecil dari ini kami anggap "
  f"tidak bermakna secara praktis bagi pengguna daftar sepuluh hasil. Kesetaraan diklaim bila IK 90% berada seluruhnya di dalam batas tersebut {c('lakens2017')}. "
  "Batas ini ditetapkan pasca-review, sehingga uji kesetaraan bersifat eksploratif. Perbedaan antar-generator diuji "
  f"dengan uji Friedman {c('friedman1937')} dan tren jumlah sampel dengan uji L Page {c('page1963')}. Model linear "
  f"campuran {c('seabold2010')} memodelkan ΔP@{K} terhadap Dense per query, generator, prompt, dan formulasi "
  "(tunggal/multi-sampel) dengan efek tetap prompt × formulasi dan generator (referensi: "
  + LAB[sorted(LID)[0]] + ") serta intersep acak per query. Interaksi HyDE × filter diuji sebagai kontras per query "
  "(HyDE+filter − Dense+filter) − (HyDE − Dense). Asosiasi CC hipotesis dengan P@K dihitung dengan Spearman setelah "
  "keduanya dipusatkan di dalam setiap sel generator–query–prompt; karena hipotesis dari query yang sama tidak "
  "independen, IK-nya dihitung dengan bootstrap klaster yang meresampel query.")
_fam_lab = {"shared": "Baseline bersama", "pooled": "HyDE gabungan vs Dense", "aggregation": "Dekomposisi agregasi",
            "interaction": "Interaksi HyDE × filter", "sensitivity_query": "Sensitivitas: tanpa query bukan CC",
            "sensitivity_untrunc": "Sensitivitas: subkorpus tanpa pemotongan", "cross_llm": "Friedman antar-generator",
            "dose": "Page: jumlah hipotesis", "F_abl": "Ablasi query + satu hipotesis dan bobot tetap",
            "F_fw": "Page: bobot query tetap", "F_grp": "Kelompok query", "F_rep_first": "Representasi: BGE awal fungsi",
            "F_rep_mean": "Representasi: BGE rerata potongan", "F_rep_max": "Representasi: BGE maksimum potongan",
            "F_repc": "Representasi: kontras terhadap awal fungsi", "F_len": "Baseline panjang",
            "F_comp": "Kepatuhan biner (per representasi)", "F_strata": "Strata jumlah potongan"}
fam_rows = [[_fam_lab.get(k, k), str(v), "utama"] for k, v in FAMM.items() if k != "per_generator"]
if "per_generator" in FAMM:
    fam_rows.append(["Per generator (satu keluarga per generator)",
                     " / ".join(str(v) for v in FAMM["per_generator"].values()), "utama"])
fam_rows += [[_fam_lab.get(k, k), str(v), "pasca-review"] for k, v in FAMS.items()
             if not k.startswith("F_rep_") or k[6:] in ("first", "mean", "max")]
fam_rows += [[f"Representasi: {k[6:]}", str(v), "pasca-review"] for k, v in FAMS.items()
             if k.startswith("F_rep_") and k[6:] not in ("first", "mean", "max")]
if fam_rows:
    T(["Keluarga uji", "Jumlah uji", "Status"], fam_rows,
      "Keluarga koreksi Holm. Keluarga utama dihitung terpisah untuk setiap metrik; keluarga pasca-review hanya untuk "
      f"P@{K}.", widths=[3.6, 1.2, 1.2], font=8)
H("4.6 Analisis sensitivitas", 2)
B(f" hasil gabungan diuji ulang tanpa kelompok query berpredikat bukan CC ({', '.join(NON_CC)}).",
  "Tanpa query berpredikat bukan CC:")
if QGR:
    B(f" kontras utama dihitung per kelompok rumusan query (Tabel {tqg}).", "Kelompok query (pasca-review):")
if TR:
    B(f" seluruh konfigurasi utama dievaluasi ulang pada subkorpus {ribu(TR['n_kept'])} unit yang tidak melebihi "
      f"jendela encoder ({TR['n_relevant_kept']} unit relevan). Karena penghapusan fungsi panjang juga mengubah "
      "komposisi korpus dan distribusi relevansi, analisis ini tidak dapat memisahkan pengaruh pemotongan dari "
      "perubahan kandidat retrieval.", "Subkorpus tanpa pemotongan:")
if FFR:
    B(f" korpus, relevansi, query, dan encoder tetap sama, tetapi setiap fungsi dipecah menjadi potongan berurutan "
      f"sepanjang {CHUNK_TOKENS} token (batas mengikuti tokenizer encoder). Seluruh token fungsi dicakup melalui "
      "potongan, dengan representasi akhir bergantung pada metode agregasi; dependensi alur kendali antarpotongan tidak "
      "direpresentasikan secara eksplisit. Skor fungsi dihitung dengan dua cara: kosinus terhadap rerata ternormalisasi "
      "embedding potongan, dan skor kosinus maksimum di antara potongan. Kode hipotetis yang melebihi jendela "
      "diperlakukan sama (rerata embedding potongan), sehingga representasi korpus dan hipotesis berubah bersamaan dan "
      "kontribusi masing-masing sisi belum dipisahkan. Kandidat retrieval tidak berubah.",
      "Cakupan seluruh token (pasca-review):")
    B(" skor maksimum memberi fungsi dengan lebih banyak potongan lebih banyak kesempatan memperoleh skor tinggi. "
      f"Kami karena itu (i) menambahkan baseline yang mengurutkan fungsi menurut panjang (jumlah token encoder dan SLOC "
      "radon) dan (ii) mengulang perbandingan HyDE–Dense di dalam strata jumlah potongan (satu potongan vs dua atau "
      "lebih), dengan kandidat retrieval dibatasi pada strata tersebut.", "Kontrol panjang (pasca-review):")
    _extra = [k for k in FFR if k not in ("first", "mean", "max")]
    if _extra:
        B(" " + "; ".join(FFR[k]["encoder"] for k in _extra) + f" {c('gunther2023')}, encoder kode berkonteks "
          "panjang, dievaluasi pada korpus yang sama tanpa pemotongan. Karena arsitektur, pooling, dan data latihnya juga "
          "berbeda, hasilnya dibaca sebagai pembanding, bukan sebagai uji efek pemotongan.",
          "Encoder pembanding (pasca-review):")
    if FFM:
        def _tl(m_, k_):
            t_ = (m_ or {}).get(k_) or {}
            return f"{ribu(t_['max'])} / {t_['n_over_window']}" if t_ else "—"
        man_rows = [[(m_.get("model") or k).split("/")[-1], (m_.get("revision") or "—")[:10], m_.get("pooling") or "—",
                     "L2", ("ya" if m_.get("prefix_query") else "tidak"), ribu(m_.get("max_seq_length") or 0),
                     _tl(m_, "tokens_documents"), _tl(m_, "tokens_hypotheses")]
                    for k, m_ in FFM.items() if "error" not in m_]
        if man_rows:
            T(["Encoder", "Revisi", "Pooling", "Normalisasi", "Prefiks query", "Jendela", "Dokumen: maks / > jendela",
               "Hipotesis: maks / > jendela"], man_rows,
              "Manifest encoder (direkam otomatis saat eksperimen). Prefiks hanya untuk query (instruksi retrieval BGE); "
              "dokumen dan hipotesis tanpa prefiks. Panjang dalam token tokenizer masing-masing encoder.",
              widths=[1.5, 0.8, 0.6, 0.7, 0.6, 0.6, 0.9, 0.9], font=7)

# =============================================================================== 5 hasil
H("5. Hasil", 1)
H("5.1 Embedding langsung dan HyDE (RQ1)", 2)
P(f"Dense mencapai P@{K} = {n(SM['dense'][M])}, dibanding ekspektasi acak {n(SM['random'][M], 4)} ({stat(r_rand)}). "
  f"IK 90% selisihnya adalah {ci(*EQ['ci90'])}: {rand_phrase(r_rand)}"
  + ((f". Dengan SESOI ±{n(SESOI_P10, 2)} (pasca-review), IK 90% tersebut "
      + ("berada seluruhnya di dalam batas, sehingga pada tingkat presisi ini Dense dapat dinyatakan setara dengan "
         "acak. " if EQS["equivalent"] else "tidak berada seluruhnya di dalam batas. ")) if EQS and not r_rand["significant"] else ". ")
  + f"Dibanding BM25 (P@{K} = {n(SM['bm25'][M])}), {status(r_bm25)} ({stat(r_bm25)}).")


def pm(key, met):
    if key.startswith("pooled:"):
        return sum(SM[f"{L}:{key[7:]}"][met] for L in LID) / len(LID)
    return SM[key][met]


mrows = []
for key, lab in (("random", "Acak (ekspektasi analitik)"), ("bm25", "BM25"), ("dense", "Dense"),
                 ("pooled:hyde_fs", "HyDE few-shot, tunggal"), ("pooled:hyde_fs_avg", "HyDE few-shot, multi-sampel"),
                 ("pooled:hyde_zs", "HyDE zero-shot, tunggal"), ("pooled:hyde_zs_avg", "HyDE zero-shot, multi-sampel"),
                 ("random_f", "Acak + filter"), ("dense_f", "Dense + filter"),
                 ("pooled:hyde_fs_avg_f", "HyDE few-shot, multi-sampel + filter"), ("oracle", "Pengurutan CC")):
    mrows.append([lab] + [n(pm(key, met)) for met in (M, "ndcg_at_k", "rr")])
tmain = T(["Konfigurasi", f"P@{K}", f"nDCG@{K}", "MRR"], mrows,
          f"Metrik utama. Baris HyDE adalah rerata atas {len(LID)} generator; HyDE tunggal dirata-ratakan atas {NS} "
          "hipotesis.", widths=[3.0, 0.9, 0.9, 0.9], font=8)
_mv = [(lab, pm(key, M), pm(key, "ndcg_at_k"), pm(key, "rr")) for key, lab in (
    ("random", "acak"), ("bm25", "BM25"), ("dense", "Dense"), ("pooled:hyde_fs", "HyDE few-shot tunggal"),
    ("pooled:hyde_fs_avg", "HyDE few-shot multi-sampel"), ("pooled:hyde_zs", "HyDE zero-shot tunggal"),
    ("pooled:hyde_zs_avg", "HyDE zero-shot multi-sampel"))]
_flip = []
for i in range(len(_mv)):
    for j in range(i + 1, len(_mv)):
        a_, b_ = _mv[i], _mv[j]
        for mi, mlab in ((2, f"nDCG@{K}"), (3, "MRR")):
            if (a_[1] - b_[1]) * (a_[mi] - b_[mi]) < 0:
                _flip.append((a_, b_, mi, mlab))
_ex_f = sorted(_flip, key=lambda x: ("HyDE" not in x[0][0] + x[1][0]))[:2]
P(f"Tabel {tmain} melaporkan ketiga metrik. "
  + (("Beberapa konfigurasi berubah urutan antar-metrik, misalnya " + "; ".join(
      f"{a_[0]} vs {b_[0]} (P@{K} {n(a_[1])} vs {n(b_[1])}; {ml} {n(a_[mi])} vs {n(b_[mi])})" for a_, b_, mi, ml in _ex_f)
      + ". ") if _flip else f"Urutan konfigurasi menurut nDCG@{K} dan MRR sejalan dengan P@{K}. ")
  + f"P@{K} dipertahankan sebagai metrik utama sesuai tujuan mengukur jumlah fungsi relevan dalam sepuluh hasil teratas; "
    f"uji untuk nDCG@{K} dan MRR tersedia dalam paket replikasi.")
prow = []
for key, lab in (("hyde_fs", "few-shot, tunggal"), ("hyde_fs_avg", "few-shot, multi-sampel (+query)"),
                 ("hyde_fs_t0", "few-shot, T = 0"), ("hyde_zs", "zero-shot, tunggal"),
                 ("hyde_zs_avg", "zero-shot, multi-sampel (+query)"), ("hyde_zs_t0", "zero-shot, T = 0")):
    r = pool[key]
    prow.append(["HyDE " + lab, n(r["mean_a"]), f"{n(r['mean_diff'], 3, True)} {ci(*r['ci95'])}",
                 f"{r['wins']}/{r['ties']}/{r['losses']}",
                 ("< 0,001" if r["p_holm"] < .001 else pnum(r["p_holm"])) + dag(r), n(r["r_rb"], 2, True)])
T(["Konfigurasi (rerata atas generator)", f"P@{K}", "Δ terhadap Dense [IK 95%]", "M/S/K", "p_Holm", "r_rb"],
  prow, f"Efek HyDE gabungan pada P@{K} (Dense: {n(SM['dense'][M])}). M/S/K = menang/seri/kalah per query; "
        "† p_Holm < 0,05.", widths=[2.3, 0.6, 1.6, 0.7, 0.7, 0.5], font=8)
P(f"Dirata-ratakan atas generator, HyDE few-shot tunggal: {status(pool['hyde_fs'])} ({stat(pool['hyde_fs'])}); "
  f"HyDE zero-shot tunggal: {status(pool['hyde_zs'])} ({stat(pool['hyde_zs'])}); HyDE few-shot multi-sampel memberi "
  f"selisih terbesar pada konfigurasi dasar ({stat(pool['hyde_fs_avg'])}). Presisi absolut tetap rendah: rata-rata "
  f"{n(pool['hyde_fs_avg']['mean_a'] * K, 1)} unit relevan di antara {K} teratas.")
srows = []
for a, lab in (("pooled:hyde_fs", "HyDE few-shot, tunggal"), ("pooled:hyde_fs_avg", "HyDE few-shot, multi-sampel"),
               ("pooled:hyde_zs", "HyDE zero-shot, tunggal"), ("pooled:hyde_zs_avg", "HyDE zero-shot, multi-sampel"),
               ("dense", "Dense vs acak")):
    r1, r2 = SENS.get(a), SENU.get(a)
    srows.append([lab,
                  f"{n(r1['mean_diff'], 3, True)} {ci(*r1['ci95'])}{dag(r1)}" if r1 else "—",
                  f"{n(r2['mean_diff'], 3, True)} {ci(*r2['ci95'])}{dag(r2)}" if r2 else "—"])
tsens = T(["Kontras", f"Tanpa query {', '.join(NON_CC)} (n = {NQ - len(NON_CC)})",
   f"Subkorpus tanpa pemotongan ({ribu(TR.get('n_kept', 0))} unit)"], srows,
  f"Analisis sensitivitas: ΔP@{K} terhadap Dense (baris terakhir: Dense terhadap acak). † p_Holm < 0,05.",
  widths=[2.0, 2.2, 2.2], font=8)
P("Untuk HyDE few-shot multi-sampel, peningkatannya "
  + ("bertahan" if sens_q else "tidak terdeteksi") + " ketika kelompok query berpredikat bukan CC dikeluarkan"
  + (f" ({stat(SENS['pooled:hyde_fs_avg'])}; Tabel {tsens}). " if SENS.get("pooled:hyde_fs_avg") else f" (Tabel {tsens}). ")
  + ((f"Pada subkorpus tanpa fungsi yang terpotong ({ribu(TR['n_kept'])} unit, hanya {TR['n_relevant_kept']} "
      f"fungsi relevan), selisihnya {stat(r_u)}. " + TRUNC_SENT) if SU and sens_u is False else
     (f"Peningkatan juga bertahan pada subkorpus tanpa pemotongan ({stat(r_u)}). " if sens_u else ""))
  + (f"Analisis cakupan seluruh token di bawah ini menjawab keterbatasan tersebut tanpa mengubah korpus." if FFR else ""))
if FFR:
    frows = []
    for rep, v in FFR.items():
        t_ = {r["a"]: r for r in v["tests"]}
        def cell_(a):
            r = t_.get(a)
            return f"{n(r['mean_diff'], 3, True)} {ci(*r['ci95'])}{dag(r)}" if r else "—"
        frows.append([REP_LAB.get(rep, v["encoder"].split("/")[-1] + " (8.192 token, tanpa pemotongan; pembanding)"), n(v["summary_dense"]), n(v["summary_fs_avg_pooled"]),
                      cell_("pooled:hyde_fs_avg"), cell_("pooled:hyde_zs_avg"), cell_("dense")])
    tff = T(["Representasi", "Dense", "HyDE fs multi", "Δ fs multi [IK 95%]", "Δ zs multi [IK 95%]", "Dense − acak"],
            frows, f"Representasi fungsi dan encoder pada korpus yang sama ({ribu(MD['n_corpus'])} unit, {MD['n_relevant']} relevan; "
                   f"pasca-review). Nilai P@{K}; Δ terhadap Dense pada representasi yang sama; HyDE rerata atas "
                   "generator. † p_Holm < 0,05 di dalam keluarga representasi.",
            widths=[1.9, 0.55, 0.65, 1.3, 1.3, 1.3], font=7.5)
    cov = FF.get("coverage") or {}
    con = {r["desc"]: r for r in FF.get("contrast_vs_first", [])}
    txt = (f"Pada Tabel {tff}, "
           + ff_sentence()[0].lower() + ff_sentence()[1:]
           + (f"Sebanyak {cov['n_relevant_multi_chunk']} dari {MD['n_relevant']} fungsi relevan dan "
              f"{ribu(cov['n_hyp_multi_chunk'])} dari {ribu(cov['n_hyp'])} hipotesis stokastik memerlukan lebih dari "
              "satu potongan. " if cov else ""))
    _cs = [f"{lab_}: {stat(con[f'efek HyDE few-shot multi-sampel: {k_} - first'])}"
           for k_, lab_ in (("mean", "rerata potongan"), ("max", "maksimum potongan"))
           if f"efek HyDE few-shot multi-sampel: {k_} - first" in con]
    if _cs:
        txt += ("Dibanding representasi awal fungsi, perubahan efek HyDE few-shot multi-sampel adalah ("
                + "; ".join(_cs) + "). ")
    _ex = [k for k in FFR if k not in ("first", "mean", "max")]
    jtxt = ""
    if EX:
        _rd, _rz = ff_row(EX, "dense"), ff_row(EX, "pooled:hyde_zs_avg")
        _sm = FFE.get(EX, {}).get("summary", {})
        _ex_fs = {L: _sm[f"{L}:hyde_fs_avg"][M] for L in LID if f"{L}:hyde_fs_avg" in _sm}
        jtxt = (f"Encoder kode berkonteks panjang {FFR[EX]['encoder']} (jendela 8.192 token; tidak ada fungsi yang "
                f"terpotong) memberi pola yang sejalan tetapi lebih tajam (Tabel {{tff}}). Dense mencapai P@{K} = "
                f"{n(FFR[EX]['summary_dense'])}"
                + (f" (dibanding acak: {status(_rd)}; {stat(_rd)})" if _rd else "") + f", sedangkan HyDE few-shot multi-sampel mencapai "
                f"{n(FFR[EX]['summary_fs_avg_pooled'])} ({stat(ex_fs)})"
                + (f"; untuk HyDE zero-shot multi-sampel, {status(_rz)} ({stat(_rz)})" if _rz else "") + ". ")
        if _ex_fs:
            _lo, _hi = min(_ex_fs.values()), max(_ex_fs.values())
            jtxt += (f"Urutan generator juga berubah: HyDE few-shot multi-sampel berkisar dari {n(_lo)} "
                     f"({names([L for L in _ex_fs if abs(_ex_fs[L] - _lo) < 1e-9])}) hingga {n(_hi)} "
                     f"({names([L for L in _ex_fs if abs(_ex_fs[L] - _hi) < 1e-9])}), padahal dengan encoder utama "
                     f"rentangnya {n(low_v)} ({names(worst_list)}) hingga {n(top_v)} ({names(best_list)}). ")
        jtxt += ("Karena arsitektur, data latih, dan panjang konteks berubah sekaligus, hasil ini tidak menguji efek "
                 "pemotongan secara terpisah; hasil ini menunjukkan bahwa manfaat HyDE tidak terbatas pada satu "
                 "encoder, tetapi besarnya dan urutan generator bergantung pada encoder.")
    _dd = [(k, ff_row(k, "dense")) for k in ("mean", "max") if ff_row(k, "dense")]
    if _dd:
        txt += ("Cakupan seluruh token juga menaikkan Dense sendiri (P@" + str(K) + " " + n(FFR["first"]["summary_dense"])
                + " pada awal fungsi menjadi " + " dan ".join(f"{n(FFR[k]['summary_dense'])} ({'rerata' if k == 'mean' else 'maksimum'})"
                                                               for k, _ in _dd) + "), dan "
                + ("pada kedua cara agregasi Dense kini lebih baik dari acak. " if all(r["significant"] and r["mean_diff"] > 0 for _, r in _dd)
                   else "perbedaannya dengan acak berubah. "))
    _zz = [ff_row(k, "pooled:hyde_zs_avg") for k in ("mean", "max")]
    if all(z and not z["significant"] for z in _zz):
        txt += "Efek HyDE zero-shot multi-sampel tidak terdeteksi pada kedua representasi seluruh token. "
    if ff_fail:
        txt += ("Dengan demikian, peningkatan yang terlihat pada representasi utama tidak bertahan ketika seluruh isi "
                "fungsi dibaca; temuan RQ1 karena itu bergantung pada konfigurasi representasi.")
    elif ff_hold:
        txt += ("Dengan demikian, peningkatan pada representasi utama tidak semata-mata merupakan artefak pembacaan "
                "awal fungsi, meskipun besarnya bergantung pada cara potongan digabungkan.")
    P(txt)
    if jtxt:
        P(jtxt.replace("{tff}", str(tff)))
        jtxt = ""
    # ---- kontrol panjang dan strata jumlah potongan (review 3)
    if LENB or STRA.get("tests"):
        lsum = LENB.get("summary", {})
        lt = {x["b"] + "|" + x["a"]: x for x in LENB.get("tests", [])}
        ltxt = ("Skor maksimum potongan memberi fungsi yang lebih panjang lebih banyak kesempatan memperoleh skor tinggi, "
                f"dan {cov.get('n_relevant_multi_chunk', '—')} dari {MD['n_relevant']} fungsi relevan memerlukan lebih dari "
                "satu potongan. ")
        if lsum.get("length"):
            ltxt += (f"Mengurutkan fungsi menurut jumlah token menghasilkan P@{K} = {n(lsum['length'][M])}"
                     + (f" (SLOC: {n(lsum['sloc'][M])})" if lsum.get("sloc") else "")
                     + f", dibanding ekspektasi acak {n(SM['random'][M])}"
                     + ("; fungsi terpanjang di korpus ini umumnya ber-CC rendah, misalnya method dengan anotasi "
                        "dokumentasi parameter yang panjang. " if lsum["length"][M] < SM["random"][M] else ". "))
            _lh = lt.get("length|pooled:hyde_fs_avg")
            if _lh:
                ltxt += f"HyDE few-shot multi-sampel melampaui baseline panjang ({stat(_lh)}). "
        P(ltxt)
        _st = STRA.get("tests", [])
        if _st:
            ssum = STRA.get("summary", {})
            srows_ = []
            for x in _st:
                sm_ = ssum.get(x["rep"], {}).get(x["stratum"], {})
                srows_.append([rep_name(x["rep"]), x["stratum"], f"{sm_.get('n_docs', '—')} / {sm_.get('n_relevant', '—')}",
                               n(sm_["random_p10"]) if sm_.get("random_p10") is not None else "—",
                               n(sm_["dense"]) if sm_.get("dense") is not None else "—",
                               n(sm_["fs_avg_pooled"]) if sm_.get("fs_avg_pooled") is not None else "—",
                               f"{n(x['mean_diff'], 3, True)} {ci(*x['ci95'])}{dag(x)}"])
            tst = T(["Representasi", "Strata", "Unit / relevan", "Acak", "Dense", "HyDE fs multi", "Δ [IK 95%]"], srows_,
                    f"Perbandingan HyDE few-shot multi-sampel dan Dense di dalam strata jumlah potongan ({CHUNK_TOKENS} "
                    "token); kandidat retrieval dibatasi pada strata (pasca-review). Untuk unit satu potongan, rerata "
                    "dan maksimum potongan identik menurut konstruksi. † p_Holm < 0,05 di dalam keluarga strata.",
                    widths=[1.3, 0.8, 0.8, 0.5, 0.5, 0.7, 1.7], font=7.5)
            def _cls(x):
                return "pos" if x["significant"] and x["mean_diff"] > 0 else ("neg" if x["significant"] else "ns")
            _pos = [f"{rep_name(x['rep'])}, {x['stratum']} ({stat(x)})" for x in _st if _cls(x) == "pos"]
            _neg = [f"{rep_name(x['rep'])}, {x['stratum']} ({stat(x)})" for x in _st if _cls(x) == "neg"]
            stx = (f"Di dalam strata jumlah potongan yang sama (Tabel {tst}), keunggulan HyDE few-shot multi-sampel atas "
                   f"Dense hanya terdeteksi pada {len(_pos)} dari {len(_st)} perbandingan"
                   + (": " + "; ".join(_pos) if _pos else "") + ". "
                   + (("HyDE justru lebih rendah dari Dense pada " + "; ".join(_neg) + ". ") if _neg else "")
                   + "Perbandingan lain tidak menunjukkan perbedaan yang terdeteksi. ")
            if comp_shift:
                stx += (comp_text() + " Dengan kata lain, pada encoder utama sebagian besar peningkatan HyDE berasal dari "
                        "pergeseran hasil teratas ke fungsi yang lebih panjang, bukan dari pemeringkatan yang lebih baik di "
                        "antara fungsi dengan panjang serupa, dan perbedaan antar-generator sebagian besar mengikuti "
                        "seberapa jauh hipotesisnya menggeser hasil ke fungsi panjang. ")
                if lsum.get("length") and lsum["length"][M] < SM["random"][M]:
                    stx += ("Karena pengurutan menurut panjang saja gagal, HyDE bukan sekadar pengurut panjang: hipotesis "
                            "menggeser hasil ke fungsi panjang yang juga mirip secara isi. ")
            stx += "Peningkatan HyDE karena itu belum dapat ditafsirkan sebagai pembacaan struktur."
            P(stx)
    if jtxt:
        P(jtxt.replace("{tff}", str(tff)))
if QGR:
    qrows = []
    for g in QUERY_GROUPS:
        rr_ = {r["a"]: r for r in QGR if r["group"] == g}
        def qc(a):
            r = rr_.get(a)
            return f"{n(r['mean_diff'], 3, True)} {ci(*r['ci95'])}{dag(r)}" if r else "—"
        r0 = rr_.get("pooled:hyde_fs_avg")
        qrows.append([g, str(r0["n_queries"]) if r0 else "—", n(r0["mean_b"]) if r0 else "—",
                      qc("pooled:hyde_fs_avg"), qc("pooled:hyde_zs_avg")])
    tqr = T(["Kelompok", "n", "Dense", "Δ fs multi [IK 95%]", "Δ zs multi [IK 95%]"], qrows,
            f"Efek HyDE per kelompok query (P@{K}; pasca-review). † p_Holm < 0,05 di dalam keluarga kelompok "
            f"({FAMS.get('F_grp', len(QGR))} uji).", widths=[1.2, 0.4, 0.7, 1.9, 1.9], font=8)
    _fs = [r for r in QGR if r["a"] == "pooled:hyde_fs_avg"]
    _pos = [r for r in _fs if r["mean_diff"] > 0]
    _sig = [r["group"] for r in _fs if r["significant"] and r["mean_diff"] > 0]
    P(f"Selisih HyDE few-shot multi-sampel terhadap Dense positif pada {len(_pos)} dari {len(_fs)} kelompok query "
      f"(Tabel {tqr}; rentang {n(min(r['mean_diff'] for r in _fs), 3, True)} hingga "
      f"{n(max(r['mean_diff'] for r in _fs), 3, True)}); "
      + (f"setelah koreksi Holm, perbedaan terdeteksi pada kelompok {', '.join(_sig)}. " if _sig
         else "setelah koreksi Holm, tidak ada kelompok yang perbedaannya terdeteksi. ")
      + "Karena setiap kelompok hanya berisi 3–10 query, tabel ini kami baca sebagai gambaran heterogenitas antar-rumusan, "
        "bukan sebagai uji tersendiri per kelompok.")

H("5.2 Faktor generasi (RQ2)", 2)
grow = []
for L in LID:
    cells = [LAB[L]]
    for suf in ("fs", "fs_avg", "fs_t0", "zs", "zs_avg", "zs_t0"):
        cells.append(n(SM[f"{L}:hyde_{suf}"][M]) + dag(per(L, suf, "dense")))
    grow.append(cells)
grow.append(["Dense (tanpa LLM)"] + [n(SM["dense"][M])] * 6)
tg = T(["Generator", "fs tunggal", "fs multi", "fs T=0", "zs tunggal", "zs multi", "zs T=0"], grow,
       f"P@{K} per konfigurasi generator (multi = multi-sampel dengan query; † berbeda dari Dense, Holm di dalam "
       "generator).", widths=[1.7, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75], bold_rows=(len(LID),), font=8)
f1 = F("fig1_llm_dotplot.png", f"P@{K} dan nDCG@{K} per konfigurasi generator; garis abu-abu = baseline tanpa LLM.")
fr_fs, fr_zs = cross.get("hyde_fs_avg"), cross.get("hyde_zs_avg")
txt = (f"Secara eksploratif, hasil bervariasi antar-konfigurasi generator (Tabel {tg}, Gambar {f1}): HyDE few-shot "
       f"multi-sampel berkisar dari {n(low_v)} ({names(worst_list)}) hingga {n(top_v)} ({names(best_list)}). "
       f"Setidaknya satu varian berbeda dari Dense untuk {names(sig_any)}"
       + (f"; tidak ada varian yang berbeda untuk {names(nonsig)}" if nonsig else "") + ". ")
if fr_fs and "chi2" in fr_fs:
    txt += (f"Uji Friedman: few-shot multi-sampel χ² = {n(fr_fs['chi2'], 1)}, W = {n(fr_fs['kendalls_w'], 2)}, "
            f"p_Holm {pval(fr_fs['p_holm'])}")
    if fr_zs and "kendalls_w" in fr_zs:
        txt += f"; zero-shot multi-sampel W = {n(fr_zs['kendalls_w'], 2)}, p_Holm {pval(fr_zs['p_holm'])}."
if EX:
    txt += " Urutan ini tidak stabil terhadap pilihan encoder (Bagian 5.1)."
txt += (" Karena label generator tidak dapat diverifikasi (Bagian 7) dan karakteristik generator yang menyebabkan "
        "perbedaan belum diukur, temuan ini kami perlakukan sebagai perbedaan antar-konfigurasi layanan, bukan "
        "atribut model tertentu.")
P(txt)
arows = []
for p in ("fs", "zs"):
    for key, lab in (("avg_vs_single", "multi-sampel (+query) vs tunggal"),
                     ("nq_vs_single", "multi-sampel (tanpa query) vs tunggal: efek jumlah sampel"),
                     ("avg_vs_nq", "multi-sampel (+query) vs (tanpa query): efek penyertaan query")):
        r = agg.get((p, key))
        if r:
            arows.append([f"{PN[p]}: {lab}", f"{n(r['mean_diff'], 3, True)} {ci(*r['ci95'])}",
                          f"{r['wins']}/{r['ties']}/{r['losses']}",
                          ("< 0,001" if r["p_holm"] < .001 else pnum(r["p_holm"])) + dag(r)])
ta = T(["Kontras (rerata atas generator)", f"ΔP@{K} [IK 95%]", "M/S/K", "p_Holm"], arows,
       "Dekomposisi efek agregasi. † p_Holm < 0,05.", widths=[3.2, 1.6, 0.8, 0.8], font=8)
f2_ = F("fig2_dose_response.png", f"P@{K} HyDE multi-sampel (dengan query) sebagai fungsi jumlah hipotesis n "
                                    "(setiap titik dirata-ratakan atas semua kombinasi n hipotesis).", width=6.2)
P(f"Tabel {ta} memisahkan dua komponen yang tercampur dalam perbandingan multi-sampel vs tunggal. "
  + agg_sentence()
  + (f"(Penyertaan query: {stat(agg[('fs', 'avg_vs_nq')])}; perataan tanpa query: "
     f"{stat(agg[('fs', 'nq_vs_single')])}.) " if ("fs", "nq_vs_single") in agg and agg[("fs", "avg_vs_nq")]["significant"]
     else "")
  + f"Tren monoton terhadap n (Gambar {f2_}) "
  + ("terdeteksi untuk " + ", ".join(f"{LAB[d['llm']]} ({PN[d['prompt']]})" for d in dose_sig) if dose_sig
     else "tidak terdeteksi untuk konfigurasi mana pun") + " pada formulasi dengan query, dan "
  + ("terdeteksi untuk " + ", ".join(f"{LAB[d['llm']]} ({PN[d['prompt']]})" for d in dose_nq_sig) if dose_nq_sig
     else "tidak terdeteksi") + " pada formulasi tanpa query. Karena prompt few-shot sekaligus memuat contoh dan "
  "batasan struktur, efek prompt tidak dapat diatribusikan pada salah satunya.")
if ABL:
    def _abl_lab(d):
        return d.replace("fs:", "few-shot:").replace("zs:", "zero-shot:").replace(" vs ", " vs ")
    brows = [[_abl_lab(r["desc"]), f"{n(r['mean_diff'], 3, True)} {ci(*r['ci95'])}", f"{r['wins']}/{r['ties']}/{r['losses']}",
              ("< 0,001" if r["p_holm"] < .001 else pnum(r["p_holm"])) + dag(r)] for r in ABL.values()]
    tab_ = T(["Kontras", f"ΔP@{K} [IK 95%]", "M/S/K", "p_Holm"], brows,
             f"Ablasi pasca-review: query + satu hipotesis dan bobot query tetap (rerata atas generator dan atas semua "
             f"kombinasi n hipotesis; Holm di dalam keluarga ablasi, {len(ABL)} uji). IK adalah IK 95% pointwise, "
             "sedangkan p telah dikoreksi Holm. † p_Holm < 0,05.", widths=[3.4, 1.5, 0.7, 0.8], font=8)
    def _a(p, kind):
        key = {"q1": f"{p}: query + 1 hipotesis vs 1 hipotesis",
               "n_q": f"{p}: query + {NS} vs query + 1 (bobot query berubah)",
               "fw": f"{p}: bobot query tetap, n = {MD['dose_n'][-1]} vs n = {MD['dose_n'][0]}"}[kind]
        return ABL.get(key)
    q1, nq_, fw_ = _a("fs", "q1"), _a("fs", "n_q"), _a("fs", "fw")
    txt = f"Tabel {tab_} menguji dekomposisi ini secara langsung. "
    if q1:
        txt += f"Menambahkan vektor query pada satu hipotesis (few-shot): {status(q1)} ({stat(q1)}). "
    if nq_:
        txt += (f"Menambah hipotesis dari satu menjadi {NS} sambil tetap menyertakan query, dengan bobot query yang turun "
                f"dari 1/2 menjadi 1/{NS + 1}: {status(nq_)} ({stat(nq_)}). ")
    if fw_:
        txt += f"Dengan bobot query tetap 0,5, menambah hipotesis dari satu menjadi {NS}: {status(fw_)} ({stat(fw_)}). "
    if DMP.get("fs_fw"):
        txt += ("Rerata gabungan few-shot dengan bobot query tetap untuk n = " + ", ".join(str(x) for x in MD["dose_n"])
                + " adalah " + "; ".join(n(x) for x in DMP["fs_fw"]) + ". ")
    fwsig = [d for d in SUPP.get("dose_fixed_weight", []) if d["significant"]]
    txt += ("Uji L Page dengan bobot query tetap " + ("mendeteksi tren untuk " + ", ".join(
        f"{LAB[d['llm']]} ({PN[d['prompt']]})" for d in fwsig) if fwsig else "tidak mendeteksi tren untuk konfigurasi mana pun")
        + ". ")
    if q1 and fw_:
        _aq = agg.get(("fs", "avg_vs_nq"))
        if q1["significant"] and q1["mean_diff"] > 0 and not fw_["significant"]:
            txt += ("Pola ini konsisten dengan dekomposisi pada Tabel " + str(ta) + ": manfaat formulasi multi-sampel "
                    "berasal dari penyertaan query, sedangkan manfaat jumlah hipotesis belum terdeteksi.")
        elif not q1["significant"] and q1["ci95"][0] > 0 and not fw_["significant"]:
            txt += ("Besarnya efek penyertaan query pada satu hipotesis sebanding dengan efek penyertaan query pada "
                    f"Tabel {ta}" + (f" ({n(_aq['mean_diff'], 3, True)})" if _aq else "") + " dan IK 95%-nya tidak "
                    "memuat nol, tetapi setelah koreksi Holm di dalam keluarga ablasi perbedaannya belum terdeteksi. "
                    "Secara gabungan, menambah hipotesis, baik dengan bobot query berubah maupun tetap, tidak memberi "
                    "tambahan yang terdeteksi. Ablasi ini konsisten dengan dekomposisi pada Tabel " + str(ta) + ", tetapi belum "
                    "menguatkannya secara independen.")
        elif not q1["significant"]:
            txt += ("Penyertaan query pada satu hipotesis tidak memberi tambahan yang terdeteksi, sehingga ablasi ini "
                    "belum mendukung dekomposisi pada Tabel " + str(ta) + ".")
    P(txt)
_mm = mixed.get("params", {}) if isinstance(mixed, dict) else {}
if _mm:
    def _term(t):
        if t == "Intercept":
            return f"Intersep (zero-shot, tunggal, {LAB[sorted(LID)[0]]})"
        if ":" in t:
            return "Few-shot × multi-sampel"
        if "prompt" in t:
            return "Prompt few-shot"
        if "agg" in t:
            return "Multi-sampel (+query)"
        if t.startswith("C(llm)"):
            return "Generator " + LAB.get(t.split("[T.")[1].rstrip("]"), t)
        return t
    mrows_ = [[_term(t), n(v["coef"], 3, True), ci(*v["ci95"]), pval(v["p"]).replace("= ", "")]
              for t, v in _mm.items()]
    tmm = T(["Suku model", "Koefisien", "IK 95%", "p"], mrows_,
            f"Model linear campuran untuk ΔP@{K} HyDE terhadap Dense (n = {mixed.get('n_obs')} pengamatan "
            f"generator × prompt × formulasi × query; intersep acak per query"
            + (f", varians {n(mixed['group_var'], 4)}" if mixed.get("group_var") is not None else "") + ").",
            widths=[3.0, 1.0, 1.6, 0.8], font=8)
    _sigt = [_term(t) for t, v in _mm.items() if t != "Intercept" and v["p"] < .05]
    _nst = [_term(t) for t, v in _mm.items() if t != "Intercept" and v["p"] >= .05]
    P(f"Dalam model campuran (Tabel {tmm}), "
      + ("suku yang signifikan adalah " + ", ".join(_sigt) if _sigt else "tidak ada suku selain intersep yang signifikan")
      + ("; " + "suku " + ", ".join(_nst) + " tidak signifikan setelah generator dan "
         "query diperhitungkan" if _nst else "")
      + ". Karena prompt zero-shot dan formulasi tunggal menjadi referensi, efek multi-sampel pada prompt few-shot "
        "adalah jumlah suku multi-sampel dan suku interaksi. Model ini bersifat deskriptif dan tidak dikoreksi untuk "
        "perbandingan berganda.")

H("5.3 Kompleksitas kode hipotetis (RQ3)", 2)
f3_ = F("fig3_cc_hypothetical.png", "CC unit korpus dan kode hipotetis per konfigurasi generator dan prompt (T = 0,7).")
hc = "; ".join(f"{LAB[L]}: {hyp_cc[(L, 'fs')]:.0f} (few-shot) dan {hyp_cc[(L, 'zs')]:.0f} (zero-shot)"
               for L in LID if hyp_cc[(L, "fs")] is not None and hyp_cc[(L, "zs")] is not None)
if any(comp.values()):
    crows = [[LAB[L], PN[p], n(v["cc_median"], 0) if v.get("cc_median") is not None else "—",
              n(v["pct_comply_all"], 1), n(v["pct_cc_none"], 1), n(v["pct_truncated_any"], 1),
              n(v["pct_trimmed"], 1), ribu(int(v["median_chars"])) if v.get("median_chars") is not None else "—"]
             for (L, p), v in comp.items() if v]
    tcp = T(["Generator", "Prompt", "Median CC", f"% CC ≥ {MD['relevant_cc']}", "% CC tak terukur",
             "% terpotong", "% dipangkas", "Median karakter"], crows,
            f"Kepatuhan generasi terhadap predikat CC (T = 0,7; {NQ * NS} keluaran per baris). % CC ≥ "
            f"{MD['relevant_cc']} dihitung atas semua keluaran, dengan CC tak terukur dihitung tidak patuh; % terpotong = "
            f"berhenti pada batas {ribu(MAX_TOKENS)} token; CC keluaran yang dipangkas adalah batas bawah.",
            widths=[1.6, 0.75, 0.6, 0.7, 0.75, 0.7, 0.7, 0.8], font=7.5)
    _fsz = [L for L in LID if comp.get((L, "fs")) and comp.get((L, "zs"))
            and comp[(L, "fs")]["pct_comply_all"] > comp[(L, "zs")]["pct_comply_all"]]
    P(f"Kepatuhan generasi bervariasi (Tabel {tcp}): {n(min(comp_vals), 0)}–{n(max(comp_vals), 0)}% keluaran "
      f"memenuhi CC ≥ {MD['relevant_cc']}. "
      + (("Median CC berada di bawah ambang relevansi pada " + ", ".join(f"{LAB[L]} {PN[p]}" for L, p in below)
          + ", sehingga sebagian besar keluaran konfigurasi tersebut tidak memenuhi predikat. ") if below else "")
      + f"Prompt few-shot menghasilkan kepatuhan lebih tinggi daripada zero-shot pada {len(_fsz)} dari {len(LID)} "
        "generator.")
P(f"Median CC kode hipotetis adalah {hc}, dibanding median korpus {MD['corpus_median_cc']:.0f} (Gambar {f3_}). "
  f"RQ3 kami uji sebagai pemeriksaan apakah CC hipotesis dapat dipakai sebagai proksi keberhasilan retrieval. "
  f"Di dalam sel generator–query–prompt, "
  f"korelasi CC hipotesis dengan P@{K} secara gabungan mendekati nol (ρ = {n(pc['rho'], 3, True)}, {rci_lab} "
  f"{ci(*rci)}, n = {ribu(pc['n'])} hipotesis yang CC-nya terukur dari {NQ} query); panjang hipotesis juga tidak berasosiasi "
  f"(ρ = {n(pl['rho'], 3, True)}, p {pval(pl['p'])}). Per konfigurasi generator, korelasinya adalah "
  + "; ".join(f"{LAB[L]} ρ = {n(gl[L]['cc']['rho'], 2, True)} (p {pval(gl[L]['cc']['p'])})" for L in LID if "cc" in gl.get(L, {}))
  + ". "
  + (f"Asosiasi negatif yang signifikan pada {names(neg_sig)} berarti CC hipotesis bersifat prediktif di sana, tetapi "
     "ke arah yang berlawanan dengan yang diharapkan. " if neg_sig else "")
  + ("Tidak ada asosiasi positif yang signifikan. " if not pos_sig else f"Asosiasi positif muncul pada {names(pos_sig)}. ")
  + f"Pada tingkat sel generator–prompt (n = {cell['n']}), median CC hipotesis tidak berasosiasi dengan rerata "
  f"peningkatan (ρ = {n(cell['rho'], 2, True)}, p {pval(cell['p'])}). Korelasi ini bersifat deskriptif: hasil ini "
  "tidak membuktikan bahwa informasi CC hilang dari embedding. Informasi tersebut dapat saja terenkode tetapi tidak "
  "dimanfaatkan oleh pemeringkatan kosinus, atau terkalahkan oleh petunjuk gaya dan topik.")
if CBT:
    _other = [k for k in CBT if k != "first"]
    P("Korelasi di atas memakai CC kontinu, sedangkan predikat relevansi bersifat biner. Kami karena itu memisahkan dua "
      f"pertanyaan (pasca-review). Pada encoder utama, {comp_sentence('first')}. "
      + (("Pada representasi lain: " + "; ".join(f"{rep_name(k)}: {comp_sentence(k)}" for k in _other) + ". ")
         if _other else "")
      + f"Dengan demikian, pada konfigurasi yang diuji, CC hipotesis tidak dapat dipakai sebagai proksi keberhasilan "
        "retrieval; kesimpulan ini terikat pada encoder dan representasi yang dianalisis.")
else:
    P("Dengan demikian, pada encoder utama, kenaikan CC hipotesis tidak menunjukkan hubungan positif yang konsisten "
      f"dengan P@{K}.")

H("5.4 Metadata CC dan interaksi dengan HyDE (RQ4)", 2)
f4 = F("fig4_delta_heatmap.png", f"Perubahan P@{K} per query dari HyDE multi-sampel relatif terhadap Dense.")
ip = inter_pool.get("fs_avg")
P(f"Membatasi peringkat acak pada unit dengan CC ≥ {MD['filter_cc']} saja sudah menghasilkan P@{K} = "
  f"{n(SM['random_f'][M], 3)}. Dense + filter mencapai {n(SM['dense_f'][M])}, yaitu {status(r_df_rf)} dibanding acak + "
  f"filter ({stat(r_df_rf)}). Di dalam kumpulan yang difilter, HyDE few-shot multi-sampel memberi tambahan "
  f"{stat(pool['hyde_fs_avg_f'])}; konfigurasi tertinggi adalah " + " serta ".join(cfg_label(k) for k in best_cfgs)
  + f" (P@{K} = {n(best_val)}), masih jauh di bawah pengurutan CC ({n(SM['oracle'][M])}). "
  + (f"Kontras interaksi HyDE × filter gabungan untuk HyDE few-shot multi-sampel (rerata atas generator, bukan "
     f"konfigurasi tertinggi di atas) adalah {n(ip['mean_diff'], 3, True)} "
     f"(IK 95% {ci(*ip['ci95'])}, p_Holm {pval(ip['p_holm'])}): "
     + ("HyDE memberi manfaat di dalam filter, tetapi " if in_f else "manfaat HyDE di dalam filter belum terdeteksi, dan ")
     + ("sinergi antara keduanya belum terdeteksi. " if not ip["significant"] else "interaksinya terdeteksi. ") if ip else "")
  + f"Gambar {f4} menunjukkan bahwa peningkatan terkonsentrasi pada sebagian query dan konfigurasi.")

H("5.5 Ringkasan temuan", 2)
rq_rows = [
    ["RQ1", "Dense vs acak", f"{n(EQ['mean_diff'], 3, True)}; IK 90% {ci(*EQ['ci90'])}",
     (rand_phrase(r_rand)[0].upper() + rand_phrase(r_rand)[1:])],
    ["RQ1", "HyDE few-shot multi-sampel vs Dense", f"{n(pool['hyde_fs_avg']['mean_diff'], 3, True)} {ci(*pool['hyde_fs_avg']['ci95'])}",
     status(pool["hyde_fs_avg"]).capitalize()
     + ("; bertahan tanpa query berpredikat bukan CC" if sens_q else "; tidak terdeteksi tanpa query berpredikat bukan CC")
     + ("; sensitif terhadap penghapusan fungsi panjang" if sens_u is False else
        ("; bertahan pada subkorpus tanpa pemotongan" if sens_u else ""))],
    ["RQ2", "Variasi antar-konfigurasi generator", f"{n(low_v)}–{n(top_v)}",
     "Eksploratif" + ("; Friedman signifikan (few-shot)" if fr_fs and fr_fs.get("significant") else "")],
]
if ("fs", "nq_vs_single") in agg:
    rq_rows.append(["RQ2", "Efek jumlah sampel (few-shot, tanpa query)",
                    f"{n(agg[('fs', 'nq_vs_single')]['mean_diff'], 3, True)} {ci(*agg[('fs', 'nq_vs_single')]['ci95'])}",
                    status(agg[("fs", "nq_vs_single")]).capitalize()])
    rq_rows.append(["RQ2", "Efek penyertaan query (few-shot)",
                    f"{n(agg[('fs', 'avg_vs_nq')]['mean_diff'], 3, True)} {ci(*agg[('fs', 'avg_vs_nq')]['ci95'])}",
                    status(agg[("fs", "avg_vs_nq")]).capitalize()])
rq_rows += [
    ["RQ3", f"Kepatuhan CC ≥ {MD['relevant_cc']} (keluaran T = 0,7)",
     f"{n(min(comp_vals), 0)}–{n(max(comp_vals), 0)}%" if comp_vals else "—", "Bervariasi antar-konfigurasi"],
    ["RQ3", "CC hipotesis vs P@K (gabungan)", f"ρ = {n(pc['rho'], 3, True)} {ci(*rci)}",
     "Tidak ada hubungan positif konsisten (encoder utama)" + (f"; negatif pada {names(neg_sig)}" if neg_sig else "")],
    ["RQ4", "Dense + filter vs acak + filter", f"{n(r_df_rf['mean_diff'], 3, True)} {ci(*r_df_rf['ci95'])}",
     status(r_df_rf).capitalize()],
]
if ip:
    rq_rows.append(["RQ4", "Interaksi HyDE × filter", f"{n(ip['mean_diff'], 3, True)} {ci(*ip['ci95'])}",
                    "Sinergi belum terdeteksi" if not ip["significant"] else status(ip).capitalize()])
for k_, v_ in ff_whole.items():
    rq_rows.insert(2, ["RQ1", f"HyDE few-shot multi-sampel vs Dense, seluruh token ({'rerata' if k_ == 'mean' else 'maksimum'} "
                       "potongan)", f"{n(v_['mean_diff'], 3, True)} {ci(*v_['ci95'])}", status(v_).capitalize() + " (pasca-review)"])
_q1 = ABL.get("fs: query + 1 hipotesis vs 1 hipotesis")
_fw = ABL.get(f"fs: bobot query tetap, n = {MD['dose_n'][-1]} vs n = {MD['dose_n'][0]}")
for lab_, r_ in (("Query + satu hipotesis vs satu hipotesis (few-shot)", _q1),
                 (f"Bobot query tetap, n = {MD['dose_n'][-1]} vs {MD['dose_n'][0]} (few-shot)", _fw)):
    if r_:
        _i3 = next(i for i, r in enumerate(rq_rows) if r[0] == "RQ3")
        rq_rows.insert(_i3, ["RQ2", lab_, f"{n(r_['mean_diff'], 3, True)} {ci(*r_['ci95'])}",
                             status(r_).capitalize() + " (pasca-review)"])
if ex_fs:
    rq_rows.insert(2 + len(ff_whole), ["RQ1", f"HyDE few-shot multi-sampel vs Dense, {EXN}",
                                       f"{n(ex_fs['mean_diff'], 3, True)} {ci(*ex_fs['ci95'])}",
                                       status(ex_fs).capitalize() + " (pasca-review); urutan generator berubah"])
if _ST:
    rq_rows.insert(2 + len(ff_whole) + (1 if ex_fs else 0),
                   ["RQ1", "HyDE vs Dense di dalam strata jumlah potongan", f"{len(st_pos)} dari {len(_ST)} perbandingan "
                    "lebih tinggi" + (f"; {len(st_neg)} lebih rendah" if st_neg else ""),
                    "Peningkatan agregat sebagian besar efek komposisi (pasca-review)" if comp_shift else "Pasca-review"])
if CBT.get("first"):
    _b = CBT["first"]
    _i4 = next(i for i, x in enumerate(rq_rows) if x[0] == "RQ4")
    rq_rows.insert(_i4, ["RQ3", f"Patuh vs tidak patuh CC ≥ {MD['relevant_cc']} (dalam sel, encoder utama)",
                         f"{n(_b['mean_diff'], 3, True)} {ci(*_b['ci95'])}", status(_b).capitalize() + " (pasca-review)"])
T(["RQ", "Temuan", "Estimasi", "Status bukti"], rq_rows, f"Ringkasan temuan (P@{K}).",
  widths=[0.5, 2.2, 1.8, 2.0], font=7.5)

# =============================================================================== 6 pembahasan
H("6. Pembahasan", 1)
H("6.1 Interpretasi", 2)
P("Hasil ini menunjukkan bahwa HyDE dapat menaikkan retrieval fungsi ber-CC tinggi pada benchmark ini, tetapi tidak "
  "menunjukkan bahwa kenaikan tersebut terjadi karena sifat struktural kode hipotetis. Setidaknya empat penjelasan "
  "masih terbuka dan belum dapat dibedakan oleh desain ini: (1) representasi kurang peka terhadap struktur alur "
  "kendali; (2) informasi struktural terenkode tetapi geometri kemiripan kosinus tidak menerjemahkannya menjadi "
  f"peringkat yang tepat; (3) generator yang lebih berhasil menulis kode yang gayanya lebih selaras dengan kode "
  f"pustaka, sesuai temuan ReCo {c('li2024')}; dan (4) pemotongan oleh encoder menyembunyikan bagian kompleks dari "
  "fungsi yang relevan. Penjelasan (1) dan (2) memerlukan eksperimen pasangan kode terkendali dan probing; penjelasan "
  "(3) memerlukan pengukuran keselarasan gaya; penjelasan (4) sebagian ditangani oleh analisis sensitivitas dan "
  "memerlukan encoder berkonteks panjang.")
if EX and EX_GEN:
    _lo, _hi = min(EX_GEN.values()), max(EX_GEN.values())
    P("Temuan yang paling konsisten adalah ketergantungan pada pasangan generator–encoder dan cara fungsi "
      f"direpresentasikan. Konfigurasi {names(worst_list)} memberi P@{K} {n(low_v)} dengan BGE pada awal fungsi tetapi "
      f"{n(EX_GEN[worst_list[0]])} dengan {EXN}, sedangkan rentang antar-generator dengan {EXN} adalah {n(_lo)}–{n(_hi)}. "
      "Karena itu, perbandingan antar-generator atau antar-metode HyDE yang hanya memakai satu encoder dapat "
      "menghasilkan urutan yang tidak berlaku untuk encoder lain. Desain ini belum mengidentifikasi penyebabnya; "
      "kandidatnya antara lain keselarasan gaya kode hipotetis dengan data latih encoder, panjang konteks, dan cara "
      "pooling.")
if comp_shift:
    P("Analisis strata mempersempit tafsiran tersebut. Pada encoder utama, HyDE lebih sering menempatkan fungsi panjang "
      "di sepuluh hasil teratas, dan karena fungsi relevan terkonsentrasi pada fungsi panjang, pergeseran ini sudah cukup "
      "untuk menaikkan P@" + str(K) + ". Di antara fungsi dengan jumlah potongan yang sama, HyDE umumnya tidak lebih baik "
      "dari Dense. Pola ini konsisten dengan dugaan bahwa hipotesis yang panjang lebih mirip dengan fungsi panjang, "
      "terlepas dari strukturnya; dugaan ini belum diuji. Pengujian langsung "
      "memerlukan pasangan kode dengan panjang yang disamakan.")
if sens_u is False or q_drives or ff_whole:
    P("Beberapa hasil tambahan membatasi interpretasi. "
      + ((f"Pada subkorpus tanpa fungsi yang terpotong, peningkatan HyDE tidak terdeteksi; {TR['n_relevant_truncated']} "
          f"dari {TR['n_relevant']} fungsi relevan melebihi {TR['max_seq_length']} token. " + TRUNC_SENT)
         if sens_u is False else "")
      + ((ff_sentence() + ("Penjelasan (4) karena itu tetap terbuka sebagai salah satu penjelasan yang mungkin. "
                           if ff_fail else "Peningkatan tersebut karena itu tidak dapat dijelaskan semata-mata oleh "
                           "pembacaan awal fungsi. ")) if ff_whole else "")
      + ((agg_sentence() + "Vektor query mungkin berfungsi sebagai jangkar topik, tetapi desain ini tidak "
          "mengujinya. ") if q_drives else "")
      + "Mekanisme di balik pola-pola ini belum diuji dan memerlukan pasangan kode terkendali serta probing "
        "(Bagian 8).")
P((f"Ketepatan generasi berbeda antar-konfigurasi: {n(min(comp_vals), 0)}–{n(max(comp_vals), 0)}% keluaran memenuhi "
   f"CC ≥ {MD['relevant_cc']}" + (", dan median CC beberapa konfigurasi zero-shot berada di bawah ambang. " if below
                                  else ". ") if comp_vals else "")
  + ((f"Pada encoder utama, {comp_sentence('first')}; kepatuhan generasi karena itu tidak tampak sebagai penentu "
       "utama hasil pada konfigurasi tersebut.") if CBT.get("first") else
     "Pada encoder utama, kenaikan CC hipotesis tidak menunjukkan hubungan positif yang konsisten dengan P@" + str(K) + ".")
  + (f" Hubungan negatif pada {names(neg_sig)} menunjukkan bahwa kode hipotetis yang lebih kompleks dapat "
     "menurunkan hasil, misalnya karena menjadi lebih panjang, lebih bertema bisnis, atau lebih jauh secara gaya dari "
     "kode pustaka; dugaan ini belum diuji." if neg_sig else ""))
H("6.2 Implikasi praktis yang terbatas", 2)
P("Pada benchmark yang relevansinya ditentukan oleh CC, metadata CC memberi manfaat langsung, dan temuan ini "
  "sebagian merupakan konsekuensi definisi tugas. Implikasi praktis yang dapat dipertahankan adalah: bila sebuah "
  "kebutuhan didefinisikan oleh metrik yang dapat dihitung, metrik tersebut sebaiknya dihitung dan dipakai sebagai "
  "predikat eksplisit; HyDE dapat membantu bila metadata tidak tersedia, tetapi besarnya manfaat bergantung pada "
  f"konfigurasi: rata-rata {n(pool['hyde_fs_avg']['mean_a'] * K, 1)} dari {K} hasil teratas relevan pada konfigurasi dasar"
  + (f", {n(FFR['max']['summary_fs_avg_pooled'] * K, 1)} dengan skor maksimum potongan" if 'max' in FFR else "")
  + (f", dan {n(FFR[EX]['summary_fs_avg_pooled'] * K, 1)} dengan {EXN}" if EX else "")
  + ", dengan urutan generator yang berubah antar-encoder. Pemilihan generator untuk HyDE karena itu perlu dievaluasi "
    "bersama encoder yang dipakai"
  + ("; karena sebagian besar manfaatnya mengikuti preferensi terhadap fungsi panjang, manfaat tersebut dapat hilang "
     "pada korpus yang hubungan panjang dan kompleksitasnya berbeda. " if comp_shift else ". ")
  + 
  "Rekomendasi untuk kebutuhan campuran memerlukan evaluasi tersendiri (Bagian 8).")

# =============================================================================== 7 ancaman
H("7. Ancaman terhadap Validitas", 1)
PR([("Identitas generator. ", "b"),
    ("Semua generator diakses melalui gateway pihak ketiga yang kompatibel dengan OpenAI, bukan melalui endpoint "
     "resmi penyedia. ID model yang dikembalikan, seed, dan parameter sampling dicatat, tetapi identitas, versi, dan "
     "konfigurasi model di baliknya tidak dapat diverifikasi, demikian pula apakah seed dipatuhi. Akibatnya, "
     "perbandingan antar-generator mencerminkan perbedaan antar-konfigurasi layanan berlabel tertentu. Retrieval "
     f"dapat direproduksi persis dari {ribu(N_HYP)} teks hipotesis yang dirilis; reproduksi proses generasinya tidak "
     "dijamin.", "")])
_orig = V3_HYPO_CACHE.replace(".json", ".orig.json")
_excl = [m for m in CFG_LLMS if not m["enabled"] and "gateway" not in m["label"].lower() and "pilot" not in m["label"].lower()]
if _excl and os.path.exists(_orig):
    _oc = json.load(open(_orig))
    _parts = []
    for m in _excl:
        es = [e for e in _oc.values() if e["llm"] == m["id"]]
        if es:
            emp = sum(1 for e in es if not (e.get("code") or "").strip())
            lim = sum(1 for e in es if e.get("finish_reason") in ("length", "max_tokens"))
            _parts.append(f"{m['label']} ({emp} dari {len(es)} keluaran kosong; {lim} berhenti pada batas token)")
    if _parts:
        PR([("Konfigurasi yang dikeluarkan. ", "b"),
            ("Satu konfigurasi generator lain dicoba tetapi dikeluarkan sebelum analisis: " + "; ".join(_parts)
             + ". Pola ini konsisten dengan model penalaran yang menghabiskan anggaran token untuk penalaran internal "
               "sebelum menulis kode. Konfigurasi ini tidak disertakan dalam satu pun hasil.", "")])
PR([("Desain query dan ground truth. ", "b"),
    (f"Ke-{NQ} query adalah parafrasa dari satu kebutuhan, sehingga generalisasi ke kebutuhan lain terbatas. Satu "
     "kelompok query memuat predikat yang tidak setara dengan CC; hasil utama diuji ulang tanpa kelompok ini. "
     "Pengelompokan query, ablasi tambahan, analisis cakupan seluruh token, kontrol panjang, encoder pembanding, dan SESOI "
     "ditetapkan setelah review dan "
     "dilaporkan sebagai analisis pasca-review.", "")])
PR([("Pemotongan. ", "b"),
    ((f"{TR['n_relevant_truncated']} dari {TR['n_relevant']} fungsi relevan melebihi jendela {TR['max_seq_length']} "
      "token encoder. Subkorpus tanpa pemotongan mengubah komposisi korpus; analisis cakupan seluruh token "
      "mempertahankan korpus tetapi bergantung pada cara potongan digabungkan (rerata atau maksimum), tidak "
      "merepresentasikan dependensi antarpotongan, dan mengubah representasi korpus serta hipotesis sekaligus. "
      if TR else "")
     + f"Keluaran generator yang terpotong batas {ribu(MAX_TOKENS)} token diukur CC-nya setelah pemangkasan "
       "(batas bawah).", "")])
PR([("Validitas konstruk dan eksternal. ", "b"),
    (f"CC adalah proksi tunggal untuk kompleksitas struktural; ambang relevansi (CC ≥ {MD['relevant_cc']}) dan "
     f"filter (CC ≥ {MD['filter_cc']}) bersifat konvensi. Hanya satu encoder teks berukuran kecil"
     + (" dan satu encoder kode pembanding" if EX else "") + " serta tiga pustaka "
     "Python yang diuji; prompt few-shot mencampurkan contoh dan batasan struktur. Kedua encoder berbeda dalam "
     "pooling, data latih, dan panjang konteks (Tabel manifest encoder), sehingga perbedaan hasilnya tidak dapat "
     "diatribusikan pada satu faktor.", "")])
PR([("Validitas kesimpulan. ", "b"),
    (f"Dengan n = {NQ} query berpasangan, efek kecil dapat tidak terdeteksi; karena itu hasil non-signifikan "
     "dilaporkan sebagai belum terdeteksi, disertai ukuran efek dan interval kepercayaan.", "")])

# =============================================================================== 8 kesimpulan
H("8. Kesimpulan dan Agenda Penelitian", 1)
P(f"Pada benchmark tiga pustaka Python ini, beberapa konfigurasi HyDE meningkatkan P@{K}, dengan "
  "hasil yang bervariasi antar-konfigurasi generasi"
  + ((" dan sensitif terhadap cara fungsi panjang direpresentasikan. " if ff_fail else
      " dan tetap terdeteksi ketika seluruh token fungsi dicakup melalui potongan"
      + (" maupun dengan encoder kode berkonteks panjang, meskipun besarnya dan urutan generator bergantung pada "
         "encoder. " if ex_hold else ". ")) if ff_whole else ". ")
  + (("Namun, di dalam strata jumlah potongan yang sama keunggulan HyDE umumnya tidak terdeteksi; pada encoder utama "
      "peningkatan agregat sebagian besar berasal dari pergeseran hasil teratas ke fungsi panjang. ") if comp_shift else "")
  + agg_sentence()
  + f"Pada encoder utama dan konfigurasi yang diuji, kenaikan CC kode hipotetis tidak menunjukkan hubungan positif "
    f"yang konsisten dengan P@{K}"
  + (", dan hipotesis yang memenuhi predikat tidak menghasilkan retrieval yang lebih baik. " if CBT.get("first")
     and not CBT["first"]["significant"] else ". ")
  + "Temuan ini belum mengidentifikasi mekanismenya. Metadata CC memberikan manfaat langsung pada benchmark yang "
    "relevansinya ditentukan oleh CC.")
P("Status saat ini: "
  + (("peningkatan HyDE bertahan ketika seluruh token fungsi dicakup"
      + (" dan dengan encoder kode pembanding" if ex_hold else "") + ", sehingga agenda dilanjutkan dengan studi "
      "mekanisme"
      + ("; karena analisis strata menunjukkan efek komposisi panjang, studi mekanisme memprioritaskan desain yang "
         "menyamakan panjang fungsi dan hipotesis. " if comp_shift else ". ")) if (ff_whole and not ff_fail) else
     "peningkatan HyDE tidak bertahan ketika seluruh token fungsi dicakup, sehingga agenda difokuskan ulang sebagai "
     "studi kepekaan konfigurasi. ")
  + f"Agenda berikut (Tabel {_tab[0] + 1}; waktu indikatif) memakai aturan keputusan prospektif yang ditetapkan sebelum "
    "pengumpulan data studi mekanisme: bila efek HyDE tidak bertahan pada himpunan encoder dan representasi yang "
    "didaftarkan untuk studi tersebut, studi difokuskan ulang sebagai studi kepekaan konfigurasi (encoder, "
    "representasi fungsi panjang, generator, dan prompt). Analisis pasca-review dalam naskah ini tetap berstatus "
    "eksploratif.")
T(["Tahap", "Waktu", "Keluaran dan syarat lanjut"],
  [["Audit fondasi", "Bulan 1", "Sebagian selesai dalam naskah ini (pipeline, manifest encoder, analisis pasca-review). "
                                "Tersisa: generasi ulang melalui endpoint dengan identitas model terdokumentasi"],
   ["Studi mekanisme", "Bulan 2–3", f"Pasangan kode terkendali dengan panjang disamakan (struktur diubah, identifier/komentar/gaya dijaga, dan "
                                          f"sebaliknya); perubahan embedding dan peringkat; probing dengan control task "
                                          f"{c('hewitt2019')}; beberapa encoder (satu encoder kode berkonteks panjang sudah diuji di naskah ini) "
                                          "dengan pooling dan konteks yang divariasikan secara terpisah"],
   ["Evaluasi retrieval", "Bulan 4–5", "Beberapa kebutuhan independen (semantik, struktural, campuran) dengan "
                                            "parafrasa di dalamnya; label relevansi ahli; baseline kuat; repositori terpisah"],
   ["Evaluasi artefak", "Bulan 6–7", "Router yang memisahkan predikat struktural dan kebutuhan semantik; kesalahan "
                                          "parsing, relevansi, latensi, dan biaya"],
   ["Evaluasi penggunaan", "Bulan 8–9", f"Bila mengklaim manfaat kerja: evaluasi dengan pengembang mengikuti design "
                                             f"science {c('hevner2004')}, dengan ukuran sampel beralasan"]],
  "Agenda penelitian (tahap setelah audit fondasi mengikuti aturan keputusan prospektif di atas).",
  widths=[1.3, 0.8, 4.4], font=8)
H("Ketersediaan Data", 2)
P("Skrip, korpus beserta metrik dan hash commit, seluruh teks hipotesis beserta metadata generasinya, manifest "
  "encoder, hasil per query, dan keluaran statistik tersedia dalam paket replikasi"
  + (f" yang dapat diakses selama review di {REPLICATION_URL}" if REPLICATION_URL else
     " [TAUTAN PAKET REPLIKASI BELUM DIISI: atur REPLICATION_URL di config_v3.py]")
  + (f"; arsip permanen: {REPLICATION_DOI}." if REPLICATION_DOI else "; DOI arsip permanen menyusul."))

H("Lampiran A. Daftar Query", 1)
T(["ID", "Query"], [[q, t + ("  [predikat bukan CC]" if q in NON_CC else "")] for q, t in QUERIES_V2],
  "Query bahasa alami (dalam bahasa Inggris sebagaimana digunakan dalam eksperimen; Q01–Q05 dari studi pilot).",
  widths=[0.5, 5.6], font=8)

H("Daftar Pustaka", 1)
for i, k in enumerate(_order, 1):
    p = doc.add_paragraph(f"[{i}] {REFS[k]}")
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)
    for r in p.runs:
        r.font.size = Pt(9)
unused = [k for k in REFS if k not in _order]
doc.save(OUT)
print(f"[✓] {OUT} ditulis | konfigurasi generator: {LID} | pustaka: {len(_order)}"
      + (f" | tidak dipakai: {unused}" if unused else ""))
