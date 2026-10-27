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
| `05b_supp_stats_v3.py` | Post-review analyses (ablations, query groups, SESOI equivalence, cluster bootstrap, compliance with query-level aggregation, representations, length baselines incl. binary ≥ 2-chunk baseline, chunk-count strata, top-10 composition, midpoint decomposition of the global-ranking P@10 difference with sensitivity to the empty-stratum convention, cluster-bootstrap correlations per generator) |
| `06_visualize_v3.py` | Figures (`MAHYDE_LANG=id` for Indonesian labels) |
| `make_paper_v3_id.py` | Builds the manuscript (.docx) directly from the result files, so every number in the text is traceable |
| `run_v3.sh` | Full pipeline including generation |
| `run_v3_eval.sh` | Re-evaluation from the released generations (no LLM calls; re-embeds with locked checkpoints) |
| `run_stats_only.sh` | Statistics, figures and manuscript from stored rankings (no models) |
| `verify_embeddings.py` | Checks stored embeddings against the reference manifest and the locked checkpoint; exits 1 on failure |
| `pin_encoders.py` | Records/checks the commit of the Jina external modelling code (`trust_remote_code`) |
| `compare_fullfunc.py`, `run_review6_checks.sh` | Shows that re-running `04c` with the pinned code reproduces the stored results |
| `data/v2/` | Corpus (1,005 functions with code, CC, SLOC, commit hashes) and its BGE embeddings |
| `data/v3/hypothetical_cache_v3.json` | All 1,440 generation responses with metadata (requested/returned model, seed, sampling, finish reason, usage, CC) |
| `data/v3/hypothetical_cache_v3.orig.json` | The same responses before re-cleaning by `03b` |
| `data/v3/eval_*.json`, `stats*.json`, `*_log.txt` | Per-query results, encoder manifest, statistics and console logs used in the paper |
| `figures/v3/` | Figures (English; `id/` Indonesian) |

## Versions

Release `v1.3-review6` corresponds to the manuscript revised after the sixth review round (`v1.2-review5`, `v1.1-review4`: earlier rounds). All numbers in that
manuscript are produced from the files in this release (`bash run_stats_only.sh`; `bash run_v3_eval.sh` re-embeds).

## Reproducing the results

Three levels of reproduction are supported:

| Level | What is reproduced | Command | Needs |
|---|---|---|---|
| (a) Statistics | All tests, tables, figures and the manuscript from the stored rankings and per-query results | `bash run_stats_only.sh` | Python only, no models |
| (b) Retrieval | Re-embedding and re-ranking with the locked encoder checkpoints and external code, after verifying the stored embeddings | `bash run_v3_eval.sh` (runs `pin_encoders.py --check` and `verify_embeddings.py --all` first; stops on failure) | Hugging Face downloads |
| (c) Generation | Re-generating the hypotheses | `bash run_v3.sh` | API key; not guaranteed (see limitations) |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_stats_only.sh       # level (a)
bash run_v3_eval.sh          # level (b): verify -> 03b -> 04 -> 04c -> 05 -> 05b -> 06 -> manuscript
```

Notes:

- Encoder checkpoints are locked in the loaders: `BAAI/bge-small-en-v1.5` @ `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
  (`config.EMBEDDING_REVISION`; CLS pooling, 512-token window, BGE query instruction as query prefix) and
  `jinaai/jina-embeddings-v2-base-code` @ `516f4baf13dec4ddddda8631e019b5737c8bc250` (`config_v3.EXTRA_ENCODERS`;
  mean pooling, 8,192-token window, no prefix). Both outputs are L2-normalised.
- The Jina checkpoint loads its modelling code through `trust_remote_code` from a *separate* repository,
  `jinaai/jina-bert-v2-qk-post-norm`; `revision=` on the checkpoint does not lock that code. The code commit used in
  the experiments (`3baf9e3ac750e76e8edd3019170176884695fb94`) is recorded in `data/v3/encoder_pins.json`
  (`pin_encoders.py`), passed as `code_revision` by `04c_fullfunc_v3.py`, and checked against the module that is
  actually imported; a mismatch stops the run. `data/v3/pin_comparison.txt` shows that re-running `04c` with the
  pinned code reproduces all 228 stored metrics exactly.
- `verify_embeddings.py` compares the corpus SHA-256, the order of function IDs, the embedding file SHA-256 and the
  encoder revision with the released reference manifest `data/v2/embedding_reference.json`, then re-embeds the
  corpus (`--all`, as in `run_v3_eval.sh`; default is a 64-function sample, labelled `sample`). Any failed check exits
  with status 1 and stops the pipeline. The full audit for this release is `data/v2/embedding_check.json`
  (`scope: full`, 1,005 functions, minimum cosine 0.9999999).
- `04c_fullfunc_v3.py` with the Jina encoder takes a while on a laptop; skip it with `MAHYDE_SKIP_EXTRA=1`.
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
