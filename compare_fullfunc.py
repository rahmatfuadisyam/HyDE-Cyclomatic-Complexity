"""Review 6: bandingkan dua keluaran 04c (sebelum/sesudah pin kode Jina). Beda metrik > toleransi -> exit 1."""
import json
import sys

TOL = 1e-9


def main(a, b):
    A, B = json.load(open(a))["representations"], json.load(open(b))["representations"]
    worst, n = 0.0, 0
    for rep in sorted(set(A) & set(B)):
        sa, sb = A[rep].get("summary", {}), B[rep].get("summary", {})
        for mode in sorted(set(sa) & set(sb)):
            for k in ("p_at_k", "ndcg_at_k", "rr"):
                if k in sa[mode] and k in sb[mode]:
                    worst = max(worst, abs(sa[mode][k] - sb[mode][k])); n += 1
    missing = sorted(set(A) ^ set(B))
    print(f"{n} nilai dibandingkan; selisih maksimum {worst:.3g}; representasi hanya di satu berkas: {missing}")
    if worst > TOL or missing:
        print("[✗] HASIL BERBEDA", file=sys.stderr)
        raise SystemExit(1)
    print("[✓] Identik: hasil 04c sama sebelum dan sesudah penguncian kode")


if __name__ == "__main__":
    main(*sys.argv[1:3])
