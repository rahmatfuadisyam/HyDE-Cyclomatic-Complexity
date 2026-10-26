"""
v2 Langkah 1: Ekstraksi metrik per fungsi/method (kode produksi saja).

Perbaikan terhadap v1:
  * v1 memfilter `block.letter in ("A","B")`, padahal di radon `letter` adalah
    TIPE blok (F=function, M=method, C=class), bukan rank. Akibatnya tidak ada
    yang tersaring. v2 memakai `cc_rank(cc)` untuk rank A-F.
  * v1 memotong kode dengan `lines[lineno-1 : lineno-1+endline]` (endline adalah
    nomor baris absolut) sehingga snippet meluber ke kode setelah fungsi.
    v2: `lines[lineno-1 : endline]`.
  * Blok class dibuang (method-nya sudah tercatat sendiri -> tidak double count).
  * Direktori docs_src/docs/examples/scripts/tests dibuang (bukan kode produksi).
  * Duplikat kode identik dibuang; commit hash repo dicatat (reproducibility).
"""
import ast
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from radon.complexity import cc_visit, cc_rank
from radon.raw import analyze

from config import REPOS, REPO_DIR, V2_DIR, V2_CORPUS_FILE, EXCLUDE_PATH_PARTS


def repo_names():
    return [u.rstrip("/").split("/")[-1].replace(".git", "") for u in REPOS]


def clone_repos():
    os.makedirs(REPO_DIR, exist_ok=True)
    for url, name in zip(REPOS, repo_names()):
        path = os.path.join(REPO_DIR, name)
        if not os.path.exists(path):
            subprocess.run(["git", "clone", "--depth", "1", url, path], check=True)


def commit_of(path):
    try:
        out = subprocess.run(["git", "-C", path, "log", "-1", "--format=%H %cs"],
                             capture_output=True, text=True, check=True).stdout.split()
        return {"commit": out[0], "date": out[1]}
    except Exception:
        return {"commit": "unknown", "date": "unknown"}


def is_excluded(rel_parts):
    *dirs, fname = rel_parts
    if any(p in EXCLUDE_PATH_PARTS for p in dirs):
        return True
    return fname.startswith("test_") or fname.endswith("_test.py") or fname == "conftest.py"


def max_nesting(node):
    """Kedalaman nesting maksimum struktur kontrol di dalam fungsi."""
    ctrl = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)
    if hasattr(ast, "TryStar"):
        ctrl = ctrl + (ast.TryStar,)
    if hasattr(ast, "Match"):
        ctrl = ctrl + (ast.Match,)

    def walk(n, depth):
        best = depth
        for ch in ast.iter_child_nodes(n):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            d = depth + 1 if isinstance(ch, ctrl) else depth
            best = max(best, walk(ch, d))
        return best
    return walk(node, 0)


def func_nodes(tree):
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[(n.name, n.lineno)] = n
    return out


def extract_file(path: Path, repo: str, root: Path):
    src = path.read_text(encoding="utf-8", errors="ignore")
    try:
        blocks = cc_visit(src)
        tree = ast.parse(src)
    except Exception as e:
        print(f"[!] skip {path}: {e}")
        return []
    lines = src.splitlines()
    nodes = func_nodes(tree)
    rows = []
    for b in blocks:
        if b.letter == "C":          # class: dibuang, method dicatat terpisah
            continue
        code = "\n".join(lines[b.lineno - 1: b.endline])
        node = nodes.get((b.name, b.lineno))
        n_params = 0
        if node is not None:
            a = node.args
            n_params = len(a.posonlyargs) + len(a.args) + len(a.kwonlyargs) \
                + (1 if a.vararg else 0) + (1 if a.kwarg else 0)
            if b.is_method and a.args and a.args[0].arg in ("self", "cls"):
                n_params -= 1
        try:
            sloc = analyze(code).sloc
        except Exception:
            sloc = b.endline - b.lineno + 1
        rows.append({
            "repo": repo,
            "file": str(path.relative_to(root)),
            "function": b.fullname if hasattr(b, "fullname") else b.name,
            "kind": "method" if b.is_method else "function",
            "line": b.lineno,
            "endline": b.endline,
            "cc": b.complexity,
            "rank": cc_rank(b.complexity),
            "sloc": sloc,
            "n_params": n_params,
            "max_nesting": max_nesting(node) if node is not None else None,
            "code": code,
        })
    return rows


def main():
    clone_repos()
    root = Path(REPO_DIR)
    corpus, seen, repos_meta = [], set(), {}
    dropped_dup = 0
    for repo in repo_names():
        rpath = root / repo
        repos_meta[repo] = commit_of(str(rpath))
        for py in sorted(rpath.rglob("*.py")):
            rel = py.relative_to(rpath).parts
            if is_excluded(rel):
                continue
            for row in extract_file(py, repo, root):
                h = hashlib.sha1(row["code"].strip().encode()).hexdigest()
                if h in seen:
                    dropped_dup += 1
                    continue
                seen.add(h)
                row["id"] = f"f{len(corpus):05d}"
                corpus.append(row)

    os.makedirs(V2_DIR, exist_ok=True)
    with open(V2_CORPUS_FILE, "w") as f:
        json.dump({"repos": repos_meta, "n": len(corpus), "functions": corpus}, f, indent=1)

    print(f"[✓] {len(corpus)} fungsi/method (duplikat dibuang: {dropped_dup})")
    print("    per repo :", dict(Counter(r["repo"] for r in corpus)))
    print("    per rank :", dict(sorted(Counter(r["rank"] for r in corpus).items())))
    print(f"[✓] Saved to {V2_CORPUS_FILE}")


if __name__ == "__main__":
    main()
