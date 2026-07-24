#!/usr/bin/env python3
"""Select a structurally-diverse subset of SuiteSparse square matrices for the
APPT 2026 SpGEMM challenge (A x A and A x A^T).

The official 100-matrix set is not public to us, so we build our own diverse
benchmark spanning size / symmetry / sparsity-structure classes, which is
exactly what stresses a *general* SpGEMM solver. Reads ssstats.csv, filters to
real square matrices in a tractable nnz band, buckets by structural class, and
emits a manifest.
"""
import csv, sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SS = DATA / "ssstats.csv"

# tractable band: big enough to be non-trivial on GPU, small enough that A*A
# fill-in stays in A100-80GB range for a dev sweep. (User re-runs on A100.)
NNZ_MIN, NNZ_MAX = 20_000, 6_000_000

def load():
    rows = []
    with open(SS) as f:
        lines = f.read().splitlines()
    # line0 = count, line1 = date, rest = data
    for ln in lines[2:]:
        parts = ln.split(",")
        if len(parts) < 12:
            continue
        group, name = parts[0], parts[1]
        try:
            nr, nc, nnz = int(parts[2]), int(parts[3]), int(parts[4])
            isreal = int(parts[5])
            psym = float(parts[9]); nsym = float(parts[10])
        except ValueError:
            continue
        kind = parts[11]
        rows.append(dict(group=group, name=name, nrows=nr, ncols=nc, nnz=nnz,
                         isreal=isreal, psym=psym, nsym=nsym, kind=kind))
    return rows

def classify(m):
    """Coarse structural class from cheap metadata (nnz/row, symmetry, kind)."""
    deg = m["nnz"] / max(1, m["nrows"])
    sym = "sym" if m["psym"] >= 0.95 else ("nsym" if m["psym"] <= 0.6 else "psym")
    k = m["kind"].lower()
    if any(w in k for w in ["circuit", "power", "semiconductor"]):
        struct = "circuit"
    elif any(w in k for w in ["graph", "web", "network", "social"]):
        struct = "graph"       # often power-law / scale-free row degrees
    elif any(w in k for w in ["structural", "fluid", "thermal", "electromag",
                              "cfd", "mechanics", "2d", "3d", "acoustic"]):
        struct = "pde"         # usually banded / bounded degree
    elif "optimization" in k:
        struct = "opt"
    else:
        struct = "other"
    size = "S" if m["nnz"] < 100_000 else ("M" if m["nnz"] < 1_000_000 else "L")
    return struct, sym, size, deg

def main():
    rows = load()
    sq = [m for m in rows
          if m["nrows"] == m["ncols"] and m["isreal"] == 1
          and NNZ_MIN <= m["nnz"] <= NNZ_MAX]
    # bucket
    buckets = {}
    for m in sq:
        st, sy, sz, deg = classify(m)
        m["_deg"] = deg
        buckets.setdefault((st, sy), []).append(m)
    # pick: for each (struct, sym) bucket, take a small/med/large spread
    picks = []
    for key, ms in sorted(buckets.items()):
        ms.sort(key=lambda x: x["nnz"])
        if not ms:
            continue
        # small, median, large representative of this class
        idxs = sorted(set([0, len(ms)//2, len(ms)-1]))
        for i in idxs:
            picks.append(ms[i])
    # dedup + print
    seen = set(); out = []
    for m in picks:
        key = (m["group"], m["name"])
        if key in seen:
            continue
        seen.add(key); out.append(m)
    out.sort(key=lambda x: x["nnz"])

    man = DATA / "manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "name", "nrows", "nnz", "nnz_per_row",
                    "pattern_sym", "num_sym", "kind"])
        for m in out:
            w.writerow([m["group"], m["name"], m["nrows"], m["nnz"],
                        f'{m["_deg"]:.1f}', f'{m["psym"]:.3f}',
                        f'{m["nsym"]:.3f}', m["kind"]])
    print(f"square real matrices in band: {len(sq)}")
    print(f"structural buckets: {len(buckets)}")
    print(f"selected: {len(out)}  -> {man}")
    print(f'{"group/name":32s} {"n":>8s} {"nnz":>10s} {"deg":>6s} {"psym":>5s}  kind')
    for m in out:
        print(f'{m["group"]+"/"+m["name"]:32s} {m["nrows"]:8d} {m["nnz"]:10d} '
              f'{m["_deg"]:6.1f} {m["psym"]:5.2f}  {m["kind"][:34]}')

if __name__ == "__main__":
    main()
