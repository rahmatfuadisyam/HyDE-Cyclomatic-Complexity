#!/usr/bin/env bash
# MaHyDE v3 — evaluasi ulang setelah perbaikan cache (TIDAK memanggil LLM).
#   source .venv/bin/activate && bash run_v3_eval.sh
#
# Opsional SEBELUMNYA — regenerasi output MiniMax yang kosong (max_tokens 4000, lewat gateway):
#   python 03_generate_hypothetical_v3.py --llm minimaxm31flash --redo-bad
#   python 03b_repair_cache_v3.py
# Bila tidak dijalankan, MiniMax otomatis dikeluarkan oleh gerbang kualitas (output kosong > 5%).
set -euo pipefail
cd "$(dirname "$0")"
python verify_embeddings.py                         # review 5: embedding tersimpan cocok dengan korpus & checkpoint
python 03b_repair_cache_v3.py                       # idempoten; aman diulang
python 04_evaluate_v3.py | tee data/v3/eval_log.txt  # butuh model BGE (lokal di Mac)
python 04c_fullfunc_v3.py | tee data/v3/fullfunc_log.txt  # fungsi utuh (potongan 480 token); encoder pembanding: lewati dengan MAHYDE_SKIP_EXTRA=1
python 05_stats_v3.py    | tee data/v3/stats_log.txt
python 05b_supp_stats_v3.py | tee data/v3/stats_supp_log.txt  # analisis pasca-review (ablasi, kelompok query, SESOI, fungsi utuh)
python 06_visualize_v3.py
# python make_paper_v3.py paper_mahyde_v3.docx   # versi EN belum disinkronkan dengan revisi pasca-review
MAHYDE_LANG=id python 06_visualize_v3.py
python make_paper_v3_id.py paper_mahyde_v3_id.docx
echo "== Selesai: paper_mahyde_v3_id.docx (revisi pasca-review) =="
