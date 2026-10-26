"""
v3 Langkah 6: Figure multi-LLM untuk paper (PNG 300 dpi), dibaca dari JSON hasil v3.
Warna = identitas LLM (urutan tetap, maks. 4); garis referensi abu-abu = baseline tanpa LLM.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from common_v2 import load_corpus
from config_v3 import V3_RESULTS_FILE, V3_HYPO_CACHE, V3_FIG_DIR, K, RELEVANT_CC, RANDOM_SEED, PROMPTS

INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#8a8985"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # warna kategorikal tetap
MARKERS = ["o", "s", "D", "^"]
LANG = os.environ.get("MAHYDE_LANG", "en")
if LANG == "id":
    V3_FIG_DIR = os.path.join(V3_FIG_DIR, "id")
TR = {
    "en": {"single": "single", "avg": "avg (Gao)", "t0": "T = 0", "single_f": "single + filter",
           "avg_f": "avg + filter", "dense_nollm": "Dense (no LLM)", "dense_f": "Dense + filter",
           "n_axis": "Hypotheses averaged with the query (n)", "corpus": "Corpus",
           "cc_axis": "Cyclomatic complexity (log)", "relevance": "relevance",
           "delta": "Δ P@{K} vs Dense (HyDE avg)"},
    "id": {"single": "tunggal", "avg": "rata-rata (Gao)", "t0": "T = 0", "single_f": "tunggal + filter",
           "avg_f": "rata-rata + filter", "dense_nollm": "Dense (tanpa LLM)", "dense_f": "Dense + filter",
           "n_axis": "Jumlah hipotesis yang dirata-rata dengan query (n)", "corpus": "Korpus",
           "cc_axis": "Cyclomatic complexity (log)", "relevance": "relevansi",
           "delta": "Δ P@{K} terhadap Dense (HyDE rata-rata)"},
}[LANG]
PNAME = {"fs": "few-shot", "zs": "zero-shot"}
VARIANTS = [("", TR["single"]), ("_avg", TR["avg"]), ("_t0", TR["t0"])]
FVARIANTS = [("_f", TR["single_f"]), ("_avg_f", TR["avg_f"])]


def dec(ax, axis="x"):
    """Desimal koma untuk versi Indonesia."""
    if LANG != "id":
        return
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: (f"{v:.2f}".rstrip("0").rstrip(".") if v != 0 else "0").replace(".", ","))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def llm_style(res):
    llms = res["metadata"]["llms"]
    if len(llms) > len(SLOTS):
        print(f"[!] {len(llms)} LLM > {len(SLOTS)} warna; hanya {len(SLOTS)} pertama yang diplot berwarna berbeda")
    return [(m["id"], m["label"], SLOTS[i % len(SLOTS)], MARKERS[i % len(MARKERS)]) for i, m in enumerate(llms)]


def fig_dotplot(res, path):
    from matplotlib.lines import Line2D
    S = res["summary"]
    rows = [(p, suf, f"{PNAME[p]} · {lab}") for p in PROMPTS for suf, lab in VARIANTS + FVARIANTS]
    styles = llm_style(res)
    refs = (("dense", "--", TR["dense_nollm"]), ("bm25", "-.", "BM25"), ("dense_f", ":", TR["dense_f"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.3 * len(rows) + 1.6), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    y = np.arange(len(rows))[::-1]
    for ax, (metric, title) in zip(axes, [("p_at_k", f"P@{K}"), ("ndcg_at_k", f"nDCG@{K}")]):
        for ref, ls, _ in refs:
            ax.axvline(S[ref][metric], color=REF, ls=ls, lw=1.1)
        for i, (L, label, col, mk) in enumerate(styles):
            off = (i - (len(styles) - 1) / 2) * 0.17
            xs = [S[f"{L}:hyde_{p}{suf}"][metric] for p, suf, _ in rows]
            ax.scatter(xs, y + off, s=26, color=col, marker=mk, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, max(0.05, max(S[m][metric] for m in S if m != "oracle")) * 1.12)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        dec(ax, "x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[2] for r in rows])
    handles = [Line2D([], [], ls="", marker=mk, color=col, mec="white", ms=7, label=label)
               for _, label, col, mk in styles] + \
              [Line2D([], [], color=REF, ls=ls, lw=1.1, label=lab) for _, ls, lab in refs]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.55, -0.02))
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(path); plt.close(fig)


def fig_dose(res, path):
    S, ns = res["summary"], res["metadata"]["dose_n"]
    styles = llm_style(res)
    fig, axes = plt.subplots(1, len(PROMPTS), figsize=(7.0, 3.2), sharey=True)
    for ax, p in zip(axes, PROMPTS):
        ax.axhline(S["dense"]["p_at_k"], color=REF, ls="--", lw=1)
        ax.text(ns[-1], S["dense"]["p_at_k"], " Dense", va="center", fontsize=7, color=INK2)
        for L, label, col, mk in styles:
            ys = [S[f"{L}:hyde_{p}_avg@{n}"]["p_at_k"] for n in ns]
            ax.plot(ns, ys, color=col, lw=2, marker=mk, ms=6, mec="white", mew=0.8, label=label)
        ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
        ax.minorticks_off()
        ax.set_title(f"HyDE {PNAME[p]}", loc="left", fontsize=10, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        dec(ax, "y")
    axes[0].set_ylabel(f"P@{K}")
    fig.supxlabel(TR["n_axis"], fontsize=9, y=0.13)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(path); plt.close(fig)


def fig_cc(res, cache, path):
    styles = llm_style(res)
    corpus_cc = [f["cc"] for f in load_corpus()["functions"]]
    groups = [(TR["corpus"], corpus_cc, REF)]
    for L, label, col, _ in styles:
        for p in PROMPTS:
            v = [e["cc"] for e in cache.values() if e["llm"] == L and e["prompt"] == p
                 and e["sampling"] == "t07" and e["cc"] is not None]
            short = label.split(" (")[0].replace("-Instruct", "")
            groups.append((f"{short}\n{PNAME[p]}", v, col))
    fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * len(groups) + 1), 3.3))
    rng = np.random.default_rng(RANDOM_SEED)
    for i, (lab, vals, col) in enumerate(groups):
        if not vals:
            continue
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(vals)), np.maximum(vals, 1), s=7, color=col,
                   alpha=0.3 if i == 0 else 0.6, linewidths=0)
        med = float(np.median(vals))
        ax.plot([i - 0.3, i + 0.3], [med, med], color=INK, lw=1.6)
        ax.text(i, med * 1.15, f"{med:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.axhline(RELEVANT_CC, color=INK2, lw=0.8, ls="--")
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.text(len(groups) - 0.35, RELEVANT_CC, f"CC = {RELEVANT_CC}\n({TR['relevance']})", ha="left", va="center",
            fontsize=7, color=INK2, clip_on=False)
    ax.set_yscale("log")
    ticks = [1, 2, 5, 10, 20, 50, 100, 200]
    ax.set_yticks(ticks); ax.set_yticklabels([str(t) for t in ticks]); ax.minorticks_off()
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=7, rotation=30, ha="right")
    ax.set_ylabel(TR["cc_axis"])
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_delta_heatmap(res, path):
    pq = res["per_query"]
    styles = llm_style(res)
    rows, labels = [], []
    for L, label, _, _ in styles:
        for p in PROMPTS:
            rows.append([r["modes"][f"{L}:hyde_{p}_avg"]["p_at_k"] - r["modes"]["dense"]["p_at_k"] for r in pq])
            labels.append(f"{label.split(' (')[0].replace('-Instruct', '')} · {PNAME[p]}")
    M = np.array(rows)
    lim = max(0.1, float(np.abs(M).max()))
    fig, ax = plt.subplots(figsize=(7.4, 0.32 * len(rows) + 1.3))
    im = ax.imshow(M, cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xticks(range(len(pq))); ax.set_xticklabels([r["qid"] for r in pq], rotation=90)
    for (i, j), v in np.ndenumerate(M):
        if abs(v) >= 0.05:
            ax.text(j, i, f"{v:+.1f}".replace("0.", ".").replace(".", "," if LANG == "id" else "."),
                    ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > lim * 0.6 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(TR["delta"].format(K=K)); cb.outline.set_visible(False)
    if LANG == "id":
        from matplotlib.ticker import FuncFormatter
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    os.makedirs(V3_FIG_DIR, exist_ok=True)
    res = json.load(open(V3_RESULTS_FILE))
    cache = json.load(open(V3_HYPO_CACHE))
    fig_dotplot(res, os.path.join(V3_FIG_DIR, "fig1_llm_dotplot.png"))
    fig_dose(res, os.path.join(V3_FIG_DIR, "fig2_dose_response.png"))
    fig_cc(res, cache, os.path.join(V3_FIG_DIR, "fig3_cc_hypothetical.png"))
    fig_delta_heatmap(res, os.path.join(V3_FIG_DIR, "fig4_delta_heatmap.png"))
    print(f"[✓] Figures -> {V3_FIG_DIR}")


if __name__ == "__main__":
    main()
