#!/usr/bin/env python3
"""Build the official-proxy benchmark set (Group 1).

Interpretation (Yu Feng, user, 2026-07-24): the official test set matches the
SIZE regime of the first 100 matrices in SuiteSparse's DEFAULT ordering (matrix
ID order = ssstats.csv file order, which begins with the Harwell-Boeing set);
the actual matrices may differ in content. The faithful, reproducible proxy is
therefore the first 100 SQUARE real matrices in default order -- they occupy
exactly that size regime (n in [24, ~45k], median ~800; small-to-medium, where
kernel-launch / symbolic overhead dominates).

Emits data/manifest_official.csv.
"""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

def main():
    lines = open(DATA / "ssstats.csv").read().splitlines()
    recs = []
    for ln in lines[2:]:                     # default (ID) order preserved
        p = ln.split(",")
        if len(p) < 12:
            continue
        try:
            nr, nc, nnz, isreal = int(p[2]), int(p[3]), int(p[4]), int(p[5])
            psym = float(p[9])
        except ValueError:
            continue
        recs.append(dict(group=p[0], name=p[1], n=nr, nc=nc, nnz=nnz,
                         isreal=isreal, psym=psym, kind=p[11]))
    sel = [r for r in recs if r["n"] == r["nc"] and r["isreal"] == 1][:100]
    man = DATA / "manifest_official.csv"
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "name", "n", "nnz", "nnz_per_row", "pattern_sym", "kind"])
        for m in sel:
            w.writerow([m["group"], m["name"], m["n"], m["nnz"],
                        f'{m["nnz"]/max(1,m["n"]):.1f}', f'{m["psym"]:.3f}', m["kind"]])
    ns = [m["n"] for m in sel]; zs = [m["nnz"] for m in sel]
    print(f"wrote {len(sel)} square real matrices (default-order first 100) -> {man}")
    print(f"  n   range [{min(ns)}, {max(ns)}]  nnz range [{min(zs)}, {max(zs)}]")
    print(f"  kinds: {sorted(set(m['kind'][:20] for m in sel))[:8]} ...")

if __name__ == "__main__":
    main()
