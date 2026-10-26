"""
v3 Langkah 3: Generasi kode hipotetis untuk BEBERAPA LLM berversi.

Untuk setiap LLM aktif x query x prompt:
  * N_SAMPLES generasi stokastik (temperature SAMPLE_TEMPERATURE, seed = SEED + run)
  * 1 generasi deterministik (temperature 0, seed tetap) bila DETERMINISTIC_RUN
Kunci cache: "<llm>|<qid>|<prompt>|<t07|t0>|<run>"

Yang dicatat per panggilan (untuk replikasi): model yang diminta, model yang
dikembalikan server, system_fingerprint, seed (dan apakah didukung), temperature,
top_p, max_tokens, finish_reason, token usage, latency, timestamp, hash prompt.
Manifest LLM (digest Ollama, host endpoint, versi library) disimpan di llm_manifest.json.

Pemakaian:
  python 03_generate_hypothetical_v3.py                 # semua LLM aktif
  python 03_generate_hypothetical_v3.py --llm qwen25coder7b
  python 03_generate_hypothetical_v3.py --check         # hanya cek koneksi tiap LLM
"""
import argparse
import hashlib
import json
import os
import platform
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import urlparse

from radon.complexity import cc_visit, cc_rank
from radon.visitors import ComplexityVisitor

from hypo_utils_v3 import clean_code_v3 as clean_code, code_cc_v3 as code_cc
from config_v3 import (LLMS, enabled_llms, QUERIES_V2, HYDE_PROMPT_TEMPLATE, HYDE_ZS_PROMPT_TEMPLATE,
                       PROMPTS, N_SAMPLES, SAMPLE_TEMPERATURE, SAMPLE_TOP_P, DETERMINISTIC_RUN,
                       MAX_TOKENS, RANDOM_SEED, V3_DIR, V3_HYPO_CACHE, V3_MANIFEST)

TEMPLATES = {"fs": HYDE_PROMPT_TEMPLATE, "zs": HYDE_ZS_PROMPT_TEMPLATE}
SYSTEM = "You are a Python code generator. Output ONLY code."
FAKE = os.environ.get("MAHYDE_FAKE_LLM") == "1"
_lock = threading.Lock()


# ----------------------------------------------------------------------------- util
def jobs_for(llm):
    out = []
    for qid, q in QUERIES_V2:
        for p in PROMPTS:
            for r in range(N_SAMPLES):
                out.append((f"{llm['id']}|{qid}|{p}|t07|{r}", qid, q, p, SAMPLE_TEMPERATURE, RANDOM_SEED + r, r))
            if DETERMINISTIC_RUN:
                out.append((f"{llm['id']}|{qid}|{p}|t0|0", qid, q, p, 0.0, RANDOM_SEED, 0))
    return out


def api_key(llm):
    if llm["api_key_env"] is None:
        return "not-needed"
    key = os.environ.get(llm["api_key_env"], "")
    if not key and not FAKE:
        raise SystemExit(f"[{llm['id']}] env {llm['api_key_env']} belum di-set (isi di .env)")
    return key


def ollama_digest(llm):
    """Digest bobot model untuk endpoint Ollama (None bila bukan Ollama)."""
    u = urlparse(llm["base_url"])
    if u.port != 11434:
        return None
    try:
        with urllib.request.urlopen(f"{u.scheme}://{u.netloc}/api/tags", timeout=5) as r:
            tags = json.load(r)
        for m in tags.get("models", []):
            if m.get("name") == llm["model"] or m.get("model") == llm["model"]:
                return {"digest": m.get("digest"), "size": m.get("size"),
                        "modified_at": m.get("modified_at"), "details": m.get("details")}
        return {"digest": None, "note": "model belum di-pull di Ollama"}
    except Exception as e:
        return {"digest": None, "error": str(e)[:200]}


# ----------------------------------------------------------------------------- LLM call
class Caller:
    def __init__(self, llm):
        self.llm = llm
        self.seed_supported = True
        self.client = None
        if not FAKE:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key(llm), base_url=llm["base_url"], timeout=180)

    def __call__(self, prompt, temperature, seed):
        if FAKE:
            return self._fake(prompt, temperature, seed)
        kw = dict(model=self.llm["model"],
                  messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                  temperature=temperature, top_p=SAMPLE_TOP_P,
                  max_tokens=self.llm.get("max_tokens", MAX_TOKENS))
        last = None
        for attempt in range(6):
            try:
                t0 = time.time()
                if self.seed_supported:
                    try:
                        resp = self.client.chat.completions.create(seed=seed, **kw)
                    except Exception as e:  # provider menolak parameter seed
                        if "seed" in str(e).lower():
                            self.seed_supported = False
                            resp = self.client.chat.completions.create(**kw)
                        else:
                            raise
                else:
                    resp = self.client.chat.completions.create(**kw)
                ch = resp.choices[0]
                if not clean_code(ch.message.content or "") and attempt < 2:
                    raise RuntimeError(f"output kosong (finish_reason={ch.finish_reason})")
                u = getattr(resp, "usage", None)
                return {
                    "raw": ch.message.content or "",
                    "returned_model": getattr(resp, "model", None),
                    "system_fingerprint": getattr(resp, "system_fingerprint", None),
                    "finish_reason": ch.finish_reason,
                    "usage": {"prompt": getattr(u, "prompt_tokens", None),
                              "completion": getattr(u, "completion_tokens", None)} if u else None,
                    "latency_s": round(time.time() - t0, 2),
                    "seed_sent": seed if self.seed_supported else None,
                }
            except Exception as e:
                last = e
                wait = min(60, 2 ** attempt) + random.random()
                print(f"    [!] {self.llm['id']}: {str(e)[:120]} -> retry {wait:.0f}s")
                time.sleep(wait)
        raise RuntimeError(f"{self.llm['id']} gagal 6x: {last}")

    def _fake(self, prompt, temperature, seed):
        """Smoke test: kode palsu dengan kompleksitas bervariasi per LLM/seed."""
        rnd = random.Random(hash((self.llm["id"], prompt, seed, temperature)))
        depth = rnd.randint(1, 6)
        body, ind = [], "    "
        for d in range(depth):
            body.append(f"{ind * (d + 1)}for x{d} in items{d}:")
            body.append(f"{ind * (d + 2)}if x{d} > {rnd.randint(0, 9)}:")
            body.append(f"{ind * (d + 3)}total += x{d}")
            body.append(f"{ind * (d + 2)}elif x{d} < 0:")
            body.append(f"{ind * (d + 3)}continue")
        code = "def fake_fn(items0, items1, items2, items3, items4, items5):\n    total = 0\n" + \
               "\n".join(body) + "\n    return total"
        return {"raw": code, "returned_model": f"FAKE-{self.llm['id']}", "system_fingerprint": None,
                "finish_reason": "stop", "usage": None, "latency_s": 0.0, "seed_sent": seed}


# ----------------------------------------------------------------------------- main
def load_json(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def check(llms):
    for llm in llms:
        try:
            out = Caller(llm)("Write a Python function that returns 1. Output only code.", 0.0, RANDOM_SEED)
            print(f"[OK] {llm['id']:<16} -> returned_model={out['returned_model']} "
                  f"fingerprint={out['system_fingerprint']} seed_supported={out['seed_sent'] is not None}")
        except BaseException as e:
            print(f"[GAGAL] {llm['id']:<16} {str(e)[:200]}")


def run_llm(llm, cache, manifest, redo_bad=False):
    try:
        import openai as _openai_mod
    except ImportError:
        _openai_mod = None
    if redo_bad:   # generasi ulang output kosong / tidak dapat diukur
        bad = [j[0] for j in jobs_for(llm) if j[0] in cache
               and (cache[j[0]].get("empty") or cache[j[0]].get("cc") is None)]
        print(f"[+] --redo-bad: {len(bad)} entri {llm['id']} akan dibuat ulang")
        for k in bad:
            cache.pop(k)
    jobs = [j for j in jobs_for(llm) if j[0] not in cache]
    total = len(jobs_for(llm))
    print(f"\n[+] {llm['id']} ({llm['model']}): {len(jobs)} dari {total} generasi perlu dibuat")
    manifest[llm["id"]] = {
        **{k: llm[k] for k in ("label", "model", "open_weight")},
        "endpoint_host": urlparse(llm["base_url"]).netloc,
        "ollama": ollama_digest(llm) if not FAKE else None,
        "openai_python": getattr(_openai_mod, "__version__", None),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "host_platform": platform.platform(),
    }
    if not jobs:
        summarize_versions(llm, cache, manifest)
        return
    caller = Caller(llm)
    done = [0]

    def work(job):
        key, qid, q, p, temp, seed, r = job
        prompt = TEMPLATES[p].format(query=q)
        res = caller(prompt, temp, seed)
        code = clean_code(res["raw"])
        entry = {"llm": llm["id"], "requested_model": llm["model"], "qid": qid, "query": q,
                 "prompt": p, "temperature": temp, "top_p": SAMPLE_TOP_P,
                 "max_tokens": llm.get("max_tokens", MAX_TOKENS),
                 "sampling": "t0" if temp == 0 else "t07", "run": r,
                 "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest()[:12],
                 "timestamp": datetime.now(timezone.utc).isoformat(),
                 **res, "code": code, **code_cc(code)}
        with _lock:
            cache[key] = entry
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(jobs):
                save_json(cache, V3_HYPO_CACHE)
            print(f"  [{done[0]}/{len(jobs)}] {key} CC={entry['cc']} model={entry['returned_model']}")

    with ThreadPoolExecutor(max_workers=max(1, llm.get("concurrency", 1))) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for f in as_completed(futs):
            f.result()
    save_json(cache, V3_HYPO_CACHE)
    manifest[llm["id"]]["seed_supported"] = caller.seed_supported
    summarize_versions(llm, cache, manifest)


def summarize_versions(llm, cache, manifest):
    rets = sorted({cache[j[0]]["returned_model"] for j in jobs_for(llm) if j[0] in cache})
    fps = sorted({str(cache[j[0]]["system_fingerprint"]) for j in jobs_for(llm) if j[0] in cache})
    manifest[llm["id"]]["returned_models"] = rets
    manifest[llm["id"]]["system_fingerprints"] = fps
    if len(rets) > 1:
        print(f"  [!] PERINGATAN: server mengembalikan >1 versi model: {rets}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="append", help="id LLM (boleh berulang); default: semua yang enabled")
    ap.add_argument("--check", action="store_true", help="hanya uji koneksi")
    ap.add_argument("--redo-bad", action="store_true",
                    help="buat ulang entri kosong / CC tidak terukur (pakai max_tokens per LLM)")
    args = ap.parse_args()
    by_id = {m["id"]: m for m in LLMS}
    llms = [by_id[i] for i in args.llm] if args.llm else enabled_llms()
    if args.check:
        return check(llms)
    os.makedirs(V3_DIR, exist_ok=True)
    cache = load_json(V3_HYPO_CACHE, {})
    manifest = load_json(V3_MANIFEST, {})
    print(f"[+] LLM: {[m['id'] for m in llms]} | N={N_SAMPLES} (T={SAMPLE_TEMPERATURE}) "
          f"+ {'1 (T=0)' if DETERMINISTIC_RUN else '0'} | {'FAKE' if FAKE else 'REAL'}")
    for llm in llms:
        run_llm(llm, cache, manifest, redo_bad=args.redo_bad)
        save_json(manifest, V3_MANIFEST)
    print(f"\n[✓] Cache {V3_HYPO_CACHE} ({len(cache)} entri) | manifest {V3_MANIFEST}")


if __name__ == "__main__":
    main()
