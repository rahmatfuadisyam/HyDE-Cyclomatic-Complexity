"""
Konfigurasi eksperimen v3: HyDE multi-LLM yang dapat direplikasi.

Prinsip replikasi (dijawab langsung ke kritik reviewer):
  * Setiap LLM diidentifikasi dengan ID model BERVERSI (snapshot bertanggal untuk API,
    tag + digest untuk model open-weight lokal). Tidak ada gateway pihak ketiga yang
    identitas modelnya tidak bisa diverifikasi (gateway lama dimatikan; hanya arsip).
  * Setiap panggilan mencatat: model yang diminta, model yang DIKEMBALIKAN server
    (response.model), system_fingerprint, seed, parameter sampling, token usage, waktu.
  * Minimal satu model open-weight -> siapa pun bisa menjalankan ulang secara lokal.

Korpus & embedding memakai data v2 (data/v2/corpus.json, corpus_emb.npy) — tidak berubah.

CARA MENGATUR MODEL
-------------------
Ubah `enabled` di daftar LLMS. API key dibaca dari environment / .env
(nama variabelnya di field `api_key_env`). Untuk model lokal via Ollama:
    ollama pull qwen2.5-coder:7b && ollama pull llama3.1:8b
"""
import os

import config  # memuat .env + konfigurasi v2 (korpus, query, prompt)
from config import (DATA_DIR, QUERIES_V2, HYDE_PROMPT_TEMPLATE, HYDE_ZS_PROMPT_TEMPLATE,  # noqa: F401
                    K, RELEVANT_CC, FILTER_CC, RANDOM_SEED, N_BOOTSTRAP,
                    BGE_QUERY_INSTRUCTION, EMBEDDING_MODEL)

V3_DIR = os.environ.get("MAHYDE_V3_DIR", os.path.join(DATA_DIR, "v3"))
V3_HYPO_CACHE = os.path.join(V3_DIR, "hypothetical_cache_v3.json")
V3_MANIFEST = os.path.join(V3_DIR, "llm_manifest.json")
V3_RESULTS_FILE = os.path.join(V3_DIR, "eval_results_v3.json")
V3_STATS_FILE = os.path.join(V3_DIR, "stats_v3.json")
V3_FIG_DIR = os.environ.get("MAHYDE_V3_FIG_DIR", os.path.join("figures", "v3"))

# ----------------------------------------------------------------------------- LLM
# Field:
#   id           : nama pendek (dipakai di file hasil & figure)
#   label        : nama untuk paper
#   model        : ID model BERVERSI yang dikirim ke API
#   base_url     : endpoint OpenAI-compatible
#   api_key_env  : nama env var berisi API key (None = tidak perlu)
#   open_weight  : True bila bobot model terbuka (dapat dijalankan ulang lokal)
#   concurrency  : jumlah request paralel (lokal: 1)
#   enabled      : ikut dijalankan atau tidak
LLMS = [
    {
        "id": "gpt6astra",
        "label": "GPT-6-astra",
        "model": "cx/gpt-6-astra",
        "base_url": "https://9ai.gudangapp.store/v1",
        "api_key_env": "OPENAI_API_KEY",
        "open_weight": False,
        "concurrency": 4,
        "enabled": True,
    },
    {
        "id": "qwen25coder32b",
        "label": "Qwen2.5-Coder-32B-Instruct",
        "model": "cf/@cf/qwen/qwen2.5-coder-32b-instruct",   # Cloudflare Workers AI via gateway
        "base_url": "https://9ai.gudangapp.store/v1",
        "api_key_env": "OPENAI_API_KEY",
        "open_weight": True,
        "concurrency": 1,
        "enabled": True,
    },
    {
        "id": "gemini38flash",
        "label": "Gemini-3.8-Flash",
        "model": "ag/gemini-3.8-flash",
        "base_url": "https://9ai.gudangapp.store/v1",
        "api_key_env": "OPENAI_API_KEY",
        "open_weight": False,   # dikoreksi: bobot tidak dirilis / tidak terverifikasi
        "concurrency": 1,
        "enabled": True,
    },
    {
        "id": "minimaxm31flash",
        "label": "MiniMax-M3.1-Flash",
        "model": "nvidia/minimaxai/minimax-m3.1-flash",
        "base_url": "https://9ai.gudangapp.store/v1",
        "api_key_env": "OPENAI_API_KEY",
        "open_weight": False,   # dikoreksi: bobot tidak dirilis / tidak terverifikasi
        "concurrency": 4,
        "max_tokens": 4000,   # model reasoning: token habis untuk penalaran -> output kosong pada 1000
        "enabled": False,   # dikeluarkan dari analisis (keputusan peneliti)
    },
    {
        "id": "claudesonnet5",
        "label": "Claude-Sonnet-5",
        "model": "cc/claude-sonnet-5",
        "base_url": "https://9ai.gudangapp.store/v1",
        "api_key_env": "OPENAI_API_KEY",
        "open_weight": False,   # dikoreksi: bobot tidak dirilis / tidak terverifikasi
        "concurrency": 2,
        "enabled": True,
    },
]


def enabled_llms():
    return [m for m in LLMS if m["enabled"]]


# ----------------------------------------------------------------------------- generasi
PROMPTS = ("fs", "zs")          # fs = few-shot bisnis (pilot), zs = zero-shot ala Gao et al.
N_SAMPLES = 8                   # generasi stokastik per (LLM, query, prompt); Gao et al. memakai N = 8
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P = 1.0
DETERMINISTIC_RUN = True        # tambahan 1 generasi temperature 0 (seed tetap) per (LLM, query, prompt)
MAX_TOKENS = 1000               # default; bisa di-override per LLM dengan field "max_tokens"

# Gerbang kualitas data: LLM dikeluarkan dari analisis bila proporsi output kosong melebihi ambang ini
MAX_EMPTY_RATE = 0.05
# Semua LLM v3 diakses lewat gateway pihak ketiga (bukan endpoint resmi penyedia) -> dicatat di paper
ACCESS_NOTE = "third-party OpenAI-compatible gateway"
DOSE_N = (1, 2, 4, 8)           # jumlah hipotesis untuk analisis dose-response (subset berurutan)


# =====================================================================
# === REVISI REVIEW KEDUA ==============================================
# =====================================================================
# Pengelompokan query menurut cara predikat dinyatakan (bukan "murni-CC").
QUERY_GROUPS = {
    "metrik": {
        "label": "Menyebut metrik atau definisinya secara eksplisit",
        "qids": ["Q01", "Q06", "Q07", "Q14"],
    },
    "struktural": {
        "label": "Menyebut ciri struktural yang terkait (percabangan, nesting, perulangan, kondisi)",
        "qids": ["Q02", "Q04", "Q05", "Q09", "Q10", "Q11", "Q13", "Q15", "Q16", "Q17"],
    },
    "pemeliharaan": {
        "label": "Menyebut hasil pemeliharaan atau kompleksitas umum (refaktor, utang teknis, paling kompleks)",
        "qids": ["Q08", "Q12", "Q20"],
    },
    "bukan_cc": {
        "label": "Predikat bukan CC (parameter, ukuran fungsi, indentasi)",
        "qids": ["Q03", "Q18", "Q19"],
    },
}

# Representasi fungsi utuh: potong per jendela token encoder, lalu agregasi (model encoder SAMA).
CHUNK_TOKENS = 480            # < 512 untuk memberi ruang token khusus
V3_FULLFUNC_FILE = os.path.join(V3_DIR, "eval_fullfunc_v3.json")
V3_SUPP_STATS_FILE = os.path.join(V3_DIR, "stats_supp_v3.json")

# Encoder pembanding opsional (model berbeda = faktor arsitektur ikut berubah; laporkan terpisah).
# Kosongkan list untuk melewati. trust_remote_code diperlukan untuk sebagian model kode.
EXTRA_ENCODERS = [
    {"id": "jina_code_8k", "name": "jinaai/jina-embeddings-v2-base-code", "max_seq_length": 8192,
     "trust_remote_code": True, "query_prefix": "",
     "revision": "516f4baf13dec4ddddda8631e019b5737c8bc250",    # review 5: checkpoint dikunci
     "code_repo": "jinaai/jina-bert-v2-qk-post-norm", "pooling": "mean"},  # review 6: kode eksternal (trust_remote_code)
]

# Batas efek bermakna untuk P@10, dengan alasan substantif: 0,05 = setengah dokumen relevan
# di antara 10 hasil teratas. Ditetapkan pada revisi pasca-review (bukan pra-registrasi).
# review 6: revisi kode eksternal encoder (ditulis oleh pin_encoders.py)
_PINS = os.path.join(V3_DIR, "encoder_pins.json")
if os.path.exists(_PINS):
    import json as _json
    _pins = _json.load(open(_PINS))
    for _e in EXTRA_ENCODERS:
        _e["code_revision"] = (_pins.get(_e["id"]) or {}).get("code_revision")
SESOI_P10 = 0.05
N_BOOT = 10000                # resampling bootstrap per query (unit = query), interval persentil
# DOI paket replikasi (isi setelah unggah ke Zenodo/OSF); kosong = ditulis sebagai "akan ditambahkan"
REPLICATION_DOI = ""
# Review 3: tautan paket replikasi yang dapat diakses selama review (mis. repositori GitHub/OSF anonim)
REPLICATION_URL = "https://github.com/rahmatfuadisyam/HyDE-Cyclomatic-Complexity.git"
# Review 4: tag rilis repositori yang sesuai dengan versi naskah
REPLICATION_REF = "v1.3-review6"
