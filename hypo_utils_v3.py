"""
Utilitas v3 untuk output LLM: pembersihan kode & pengukuran CC yang tahan output terpotong.

Masalah yang ditemukan di run v3 pertama:
  * Claude membuka pagar ```python tetapi sering tidak menutupnya (terutama bila
    terpotong max_tokens) -> regex v2 gagal -> pagar ikut di-embed & radon gagal parse.
  * Output terpotong menghasilkan Python tidak valid -> CC = None.
Perbaikan: buang pagar pembuka/penutup secara independen; bila parse gagal, potong
baris dari belakang sampai valid ("trim repair") dan catat metodenya.
"""
import re

from radon.complexity import cc_visit, cc_rank
from radon.visitors import ComplexityVisitor

_OPEN = re.compile(r"^\s*```[A-Za-z0-9_+-]*[ \t]*\n")
_CLOSE = re.compile(r"\n?```\s*$")
_FENCED = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n(.*?)\n```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def clean_code_v3(raw):
    text = _THINK.sub("", raw or "").strip()
    m = _FENCED.search(text)
    if m:                                   # blok kode lengkap di mana pun dalam teks
        return m.group(1).strip()
    i = text.find("```")
    if i >= 0:                              # pagar pembuka tanpa penutup
        text = text[i:]
        text = _OPEN.sub("", text, count=1)
    text = _CLOSE.sub("", text)
    return text.strip()


def _cc_of(code):
    blocks = [b for b in cc_visit(code) if b.letter != "C"]
    cc = max(b.complexity for b in blocks) if blocks else ComplexityVisitor.from_code(code).complexity
    return cc, len(blocks)


def code_cc_v3(code, max_trim=80):
    """CC kode hipotetis. cc_method: parsed | trimmed | None (gagal/kosong)."""
    if not code or not code.strip():
        return {"parse_ok": False, "cc": None, "rank": None, "cc_method": None, "empty": True}
    try:
        cc, nb = _cc_of(code)
        return {"parse_ok": True, "cc": cc, "rank": cc_rank(cc), "n_blocks": nb,
                "cc_method": "parsed", "empty": False}
    except Exception as e:
        err = str(e)[:200]
    lines = code.rstrip().split("\n")
    for k in range(1, min(max_trim, len(lines) - 1) + 1):
        try:
            cc, nb = _cc_of("\n".join(lines[:-k]))
            if nb == 0 and cc <= 1:
                continue
            return {"parse_ok": False, "cc": cc, "rank": cc_rank(cc), "n_blocks": nb,
                    "cc_method": "trimmed", "trimmed_lines": k, "error": err, "empty": False}
        except Exception:
            continue
    return {"parse_ok": False, "cc": None, "rank": None, "cc_method": None, "error": err, "empty": False}
