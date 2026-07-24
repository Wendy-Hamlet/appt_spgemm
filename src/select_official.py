#!/usr/bin/env python3
"""Build the official-proxy benchmark set: the first 100 SuiteSparse *square*
matrices ordered by size (Yu Feng: the real test set is 'the first 100 by size').

"size" is ambiguous (matrix dimension n vs nnz); we default to dimension n
ascending and also print the nnz-ordered boundary so the interpretation can be
confirmed. Emits data/manifest_official.csv.
"""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

def load():
    lines = open(DATA / "ssstats.csv").read().splitlines()
    rows = []
    for ln in lines[2:]:
        p = ln.split(",")
        if len(p) < 12:
            continue
        try:
            nr, nc, nnz = int(p[2]), int(p[3]), int(p[4])
            isreal = int(p[5]); psym = float(p[9])
        except ValueError:
            continue
        if nr == nc and isreal == 1:
            rows.append(dict(group=p[0], name=p[1], n=nr, nnz=nnz,
                             psym=psym, kind=p[11]))
    return rows

def main():
    sq = load()
    by_n = sorted(sq, key=lambda m: (m["n"], m["nnz"]))
    by_nnz = sorted(sq, key=lambda m: (m["nnz"], m["n"]))
    print(f"square real matrices total: {len(sq)}")
    print(f"[by n]   first 100: n in [{by_n[0]['n']}, {by_n[99]['n']}], "
          f"nnz in [{min(m['nnz'] for m in by_n[:100])}, {max(m['nnz'] for m in by_n[:100])}]")
    print(f"[by nnz] first 100: nnz in [{by_nnz[0]['nnz']}, {by_nnz[99]['nnz']}], "
          f"n in [{min(m['n'] for m in by_nnz[:100])}, {max(m['n'] for m in by_nnz[:100])}]")

    sel = by_n[:100]
    man = DATA / "manifest_official.csv"
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "name", "n", "nnz", "nnz_per_row", "pattern_sym", "kind"])
        for m in sel:
            w.writerow([m["group"], m["name"], m["n"], m["nnz"],
                        f'{m["nnz"]/max(1,m["n"]):.1f}', f'{m["psym"]:.3f}', m["kind"]])
    print(f"wrote {len(sel)} -> {man}")
    print("\nboundary of by-n set (rows 95..100):")
    for m in by_n[95:100]:
        print(f'  {m["group"]+"/"+m["name"]:28s} n={m["n"]:>6} nnz={m["nnz"]:>8}')

if __name__ == "__main__":
    main()
