"""
Review 6: bekukan revisi KODE EKSTERNAL encoder pembanding (Jina v2 code memuat modeling/configuration dari
repositori jinaai/jina-bert-v2-qk-post-norm melalui trust_remote_code). revision= pada checkpoint tidak
otomatis mengunci repositori kode itu, jadi commit kodenya dicatat di data/v3/encoder_pins.json dan diteruskan
sebagai code_revision oleh 04c_fullfunc_v3.py; 04c memeriksa bahwa modul yang benar-benar dimuat cocok.

    python pin_encoders.py                         # ambil commit kode dari cache modul Hugging Face lokal
    python pin_encoders.py --code-revision <sha>   # tentukan manual (mis. bila cache memuat >1 versi)
    python pin_encoders.py --check                 # dipakai run_v3_eval.sh; gagal -> exit 1
"""
import argparse
import json
import os
import sys

from config_v3 import EXTRA_ENCODERS, V3_DIR

PINS = os.path.join(V3_DIR, "encoder_pins.json")


def modules_cache():
    if os.environ.get("HF_MODULES_CACHE"):
        return os.environ["HF_MODULES_CACHE"]
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    return os.path.join(hf_home, "modules")


def cached_code_revisions(code_repo):
    """transformers menyimpan modul jarak jauh di transformers_modules/<org>/<repo>/<commit>; versi baru mengganti
    '-' pada nama repositori dengan '_hyphen_'. Keduanya diperiksa."""
    org, repo = code_repo.split("/")
    cands = [os.path.join(modules_cache(), "transformers_modules", org, r)
             for r in (repo, repo.replace("-", "_hyphen_"))]
    d = next((c for c in cands if os.path.isdir(c)), cands[-1])
    if not os.path.isdir(d):
        return d, []
    revs = [x for x in os.listdir(d) if len(x) == 40 and all(c in "0123456789abcdef" for c in x)]
    return d, sorted(revs, key=lambda x: os.path.getmtime(os.path.join(d, x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-revision")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    extra = [e for e in EXTRA_ENCODERS if e.get("code_repo")]
    if a.check:
        if os.environ.get("MAHYDE_SKIP_EXTRA") == "1" or os.environ.get("MAHYDE_FAKE_EMB") == "1":
            print("[i] encoder pembanding dilewati; pemeriksaan pin tidak diperlukan"); return
        pins = json.load(open(PINS)) if os.path.exists(PINS) else {}
        missing = [e["id"] for e in extra if not (pins.get(e["id"]) or {}).get("code_revision")]
        if missing:
            print(f"[✗] code_revision belum dikunci untuk {missing}: jalankan python pin_encoders.py", file=sys.stderr)
            raise SystemExit(1)
        print("[✓] pin encoder:", {k: v["code_revision"][:12] for k, v in pins.items()}); return
    pins = json.load(open(PINS)) if os.path.exists(PINS) else {}
    for e in extra:
        if a.code_revision:
            rev, src = a.code_revision, "manual"
        else:
            d, revs = cached_code_revisions(e["code_repo"])
            print(f"[i] cache {d}: {revs}")
            if len(revs) != 1:
                print("[✗] cache memuat 0 atau >1 versi kode; tentukan dengan --code-revision <sha> "
                      "(versi yang dipakai eksperimen: direktori tertua bila belum pernah diperbarui)", file=sys.stderr)
                raise SystemExit(1)
            rev, src = revs[0], "local HF modules cache"
        pins[e["id"]] = {"model": e["name"], "revision": e.get("revision"), "code_repo": e["code_repo"],
                         "code_revision": rev, "source": src}
    json.dump(pins, open(PINS, "w"), indent=2)
    print(json.dumps(pins, indent=2))
    print(f"[✓] ditulis: {PINS}")


if __name__ == "__main__":
    main()
