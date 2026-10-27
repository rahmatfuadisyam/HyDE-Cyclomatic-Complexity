#!/usr/bin/env bash
# Reproduksi tingkat (a): seluruh statistik, gambar, dan naskah dari ranking/hasil per query yang tersimpan.
# Tidak memuat encoder maupun LLM.   bash run_stats_only.sh
set -euo pipefail
cd "$(dirname "$0")"
python 05_stats_v3.py      | tee data/v3/stats_log.txt
python 05b_supp_stats_v3.py | tee data/v3/stats_supp_log.txt
python 06_visualize_v3.py
MAHYDE_LANG=id python 06_visualize_v3.py
python make_paper_v3_id.py paper_mahyde_v3_id.docx
