# HyDE for Cyclomatic-Complexity Code Retrieval — Replication Package

Replication package for the paper *"Manfaat dan Batas HyDE pada Pencarian Fungsi Python Berdasarkan
Cyclomatic Complexity"* (Benefits and Limits of HyDE for Retrieving Python Functions by Cyclomatic Complexity).

The study measures how much Hypothetical Document Embeddings (HyDE) help dense retrieval on a diagnostic
benchmark whose relevance is defined exactly by cyclomatic complexity (CC ≥ 11), across four generator
configurations, two prompts, two encoders and several representations of long functions.

## Contents

| Path | Description |
|---|---|
| `config.py`, `config_v3.py` | Corpus, queries, prompts, generators, thresholds, query groups, statistics settings |
| `01_extract_metrics_v2.py` | Clones Requests, Flask and FastAPI at fixed commits and extracts production functions with radon CC |
| `02_build_index_v2.py` | Embeds the corpus with `BAAI/bge-small-en-v1.5` |
| `03_generate_hypothetical_v3.py` | Generates hypothetical functions (8 samples at T = 0.7 + 1 at T = 0 per generator × query × prompt); needs an API key |
| `03b_repair_cache_v3.py` | Re-cleans cached generations (code fences, truncated outputs); idempotent |
| `04_evaluate_v3.py` | Main evaluation: random (analytic), BM25, Dense, HyDE variants, filters, CC sorting, length baselines, ablations |
| `04c_fullfunc_v3.py` | Same corpus with all tokens covered via 480-token chunks (mean / max), long-context code encoder (Jina v2 code), chunk-count strata, encoder manifest |
| `05_stats_v3.py` | Pre-specified tests (Wilcoxon + Holm, bootstrap CIs, Friedman, Page, mixed model, mechanism correlations) |
| `05b_supp_stats_v3.py` | Post-review analyses (ablations, query groups, SESOI equivalence, cluster bootstrap, compliance with query-level aggregation, representations, length baselines incl. binary ≥ 2-chunk baseline, chunk-count strata, top-10 composition and midpoint decomposition of the global-ranking P@10 difference) |
| `06_visualize_v3.py` | Figures (`MAHYDE_LANG=id` for Indonesian labels) |
| `make_paper_v3_id.py` | Builds the manuscript (.docx) directly from the result files, so every number in the text is traceable |
| `run_v3.sh` | Full pipeline including generation |
| `run_v3_eval.sh` | Re-evaluation from the released generations (no LLM calls) |
| `data/v2/` | Corpus (1,005 functions with code, CC, SLOC, commit hashes) and its BGE embeddings |
| `data/v3/hypothetical_cache_v3.json` | All 1,440 generation responses with metadata (requested/returned model, seed, sampling, finish reason, usage, CC) |
| `data/v3/hypothetical_cache_v3.orig.json` | The same responses before re-cleaning by `03b` |
| `data/v3/eval_*.json`, `stats*.json`, `*_log.txt` | Per-query results, encoder manifest, statistics and console logs used in the paper |
| `figures/v3/` | Figures (English; `id/` Indonesian) |

## Versions

Release `v1.1-review4` corresponds to the manuscript revised after the fourth review round. All numbers in that
manuscript are produced by `bash run_v3_eval.sh` from the files in this release.

## Reproducing the results

Retrieval and statistics can be reproduced exactly from the released generations; no API key is needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_v3_eval.sh          # 03b -> 04 -> 04c -> 05 -> 05b -> 06 -> manuscript
```

Notes:

- The encoders are downloaded from Hugging Face on first use. Revisions used in the paper (recorded in
  `data/v3/eval_fullfunc_v3.json` → `manifest`): `BAAI/bge-small-en-v1.5` @ `5c38ec7c405e`, CLS pooling,
  512-token window, BGE query instruction as query prefix; `jinaai/jina-embeddings-v2-base-code` @ `516f4baf13de`,
  mean pooling, 8,192-token window, no prefix (requires `trust_remote_code`). Both outputs are L2-normalised.
- `04c_fullfunc_v3.py` with the Jina encoder takes a while on a laptop; skip it with `MAHYDE_SKIP_EXTRA=1`.
- `data/v2/corpus_emb.npy` is reused if present; `01`/`02` rebuild the corpus and embeddings from the pinned commits.
- Smoke test without models: `MAHYDE_FAKE_EMB=1 MAHYDE_V3_DIR=/tmp/v3t python 04_evaluate_v3.py` (hashing embedder).

Regenerating hypotheses (`bash run_v3.sh`) requires `OPENAI_API_KEY` in `.env` (see `.env.example`) and an
OpenAI-compatible endpoint configured in `config_v3.py`.

## Important limitations

- **Generator identity.** All generators were accessed through a third-party OpenAI-compatible gateway. Requested and
  returned model IDs, seeds and sampling parameters are recorded, but the identity and version of the underlying
  models cannot be verified independently, and generation is not guaranteed to be reproducible. Labels such as
  "GPT-6-astra" are service labels. Retrieval results are exactly reproducible from the released texts.
- One further configuration (MiniMax-M3.1-Flash) was excluded before analysis because most outputs were empty;
  its raw outputs remain in `hypothetical_cache_v3.orig.json`.
- Analyses added after peer review are labelled *post-review* in the outputs and the manuscript and are exploratory.

## Corpus licences

The corpus contains source code from [psf/requests](https://github.com/psf/requests) (Apache-2.0),
[pallets/flask](https://github.com/pallets/flask) (BSD-3-Clause) and
[fastapi/fastapi](https://github.com/fastapi/fastapi) (MIT) at the commits listed in `data/v2/corpus.json`.
That code remains under its original licences.

## Contact

Rahmat — Program Studi Ilmu Komputer, Universitas Pancasakti Makassar.
