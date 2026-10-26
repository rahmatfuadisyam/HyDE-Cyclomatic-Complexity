"""
v3 Langkah 3b: Perbaiki cache hipotesis TANPA memanggil LLM lagi.
  * membersihkan ulang kode dari field `raw` (pagar markdown tak tertutup, <think>)
  * mengukur ulang CC dengan trim-repair untuk output terpotong
  * menandai output kosong (mis. model reasoning yang kehabisan max_tokens)
Cache asli disimpan sekali sebagai hypothetical_cache_v3.orig.json.
"""
import collections
import json
import os
import shutil

from config_v3 import V3_HYPO_CACHE
from hypo_utils_v3 import clean_code_v3, code_cc_v3


def main():
    orig = V3_HYPO_CACHE.replace(".json", ".orig.json")
    if not os.path.exists(orig):
        shutil.copy2(V3_HYPO_CACHE, orig)
        print(f"[+] Backup cache asli -> {orig}")
    cache = json.load(open(V3_HYPO_CACHE))
    changed = collections.Counter()
    for k, e in cache.items():
        code = clean_code_v3(e.get("raw", ""))
        if code != e.get("code"):
            changed[e["llm"]] += 1
        for f in ("parse_ok", "cc", "rank", "n_blocks", "cc_method", "trimmed_lines", "error", "empty"):
            e.pop(f, None)
        e["code"] = code
        e.update(code_cc_v3(code))
    tmp = V3_HYPO_CACHE + ".tmp"
    json.dump(cache, open(tmp, "w"), indent=1)
    os.replace(tmp, V3_HYPO_CACHE)

    by = collections.defaultdict(list)
    for e in cache.values():
        by[e["llm"]].append(e)
    print(f"\n{'LLM':<18}{'n':>5}{'kode berubah':>14}{'kosong':>8}{'parsed':>8}{'trimmed':>9}{'CC=None':>9}")
    for L, es in by.items():
        c = collections.Counter(e["cc_method"] for e in es)
        print(f"{L:<18}{len(es):>5}{changed[L]:>14}{sum(e['empty'] for e in es):>8}"
              f"{c['parsed']:>8}{c['trimmed']:>9}{c[None]:>9}")
    print(f"[✓] Cache diperbarui: {V3_HYPO_CACHE}")


if __name__ == "__main__":
    main()
