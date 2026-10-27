#!/usr/bin/env bash
# MaHyDE v3 — HyDE multi-LLM berversi.  Jalankan dari folder RISET:
#   source .venv/bin/activate && bash run_v3.sh
#
# Persiapan sekali:
#   1. Isi OPENAI_API_KEY_OFFICIAL di .env (key OpenAI resmi, BUKAN gateway lama)
#   2. Model lokal (open-weight):  brew install ollama && ollama serve &
#                                  ollama pull qwen2.5-coder:7b && ollama pull llama3.1:8b
#   3. Atur model yang dipakai di config_v3.py -> LLMS[...]["enabled"]
#
# Langkah 03 bisa dihentikan & dilanjutkan kapan saja (cache per generasi).
# Satu LLM saja:  python 03_generate_hypothetical_v3.py --llm qwen25coder7b
set -euo pipefail
cd "$(dirname "$0")"
pip install -q rank_bm25==0.2.2 scipy "statsmodels>=0.14" pandas
mkdir -p data/v3

# korpus & embedding v2 dipakai ulang (buat bila belum ada)
[ -f data/v2/corpus.json ]    || python 01_extract_metrics_v2.py
[ -f data/v2/corpus_emb.npy ] || python 02_build_index_v2.py

echo "== Cek koneksi semua LLM =="
python 03_generate_hypothetical_v3.py --check
python 03_generate_hypothetical_v3.py
python 03b_repair_cache_v3.py
python 04_evaluate_v3.py | tee data/v3/eval_log.txt
python 05_stats_v3.py    | tee data/v3/stats_log.txt
python 06_visualize_v3.py
echo "== Selesai: data/v3/*.json, figures/v3/*.png =="
