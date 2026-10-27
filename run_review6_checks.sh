#!/usr/bin/env bash
# Review 6: kunci kode eksternal Jina, audit embedding penuh, buktikan hasil 04c tidak berubah, perbarui naskah.
#   source .venv/bin/activate && bash run_review6_checks.sh
set -euo pipefail
cd "$(dirname "$0")"
python pin_encoders.py
python verify_embeddings.py --all
cp data/v3/eval_fullfunc_v3.json data/v3/eval_fullfunc_v3.before_pin.json
python 04c_fullfunc_v3.py | tee data/v3/fullfunc_log.txt
python compare_fullfunc.py data/v3/eval_fullfunc_v3.before_pin.json data/v3/eval_fullfunc_v3.json | tee data/v3/pin_comparison.txt
python 05b_supp_stats_v3.py | tee data/v3/stats_supp_log.txt
python make_paper_v3_id.py paper_mahyde_v3_id.docx
echo "== Selesai: pin, audit embedding, perbandingan 04c, statistik, naskah =="
