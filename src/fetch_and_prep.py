#!/usr/bin/env python3
"""Download SuiteSparse matrices from the manifest, convert to CSR (.npz), and
compute cheap structural descriptors used later by the dispatcher.

Descriptors (all cheap, single pass): n, nnz, deg_mean/max/cv (row-imbalance),
pattern symmetry estimate, and an A*A output-nnz *upper bound* (sum over rows of
sum of row-lengths of referenced rows) computed on a sample for large matrices.
"""
import io, sys, csv, tarfile, urllib.request, os
from pathlib import Path
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data"
MTX = DATA / "mtx"; MTX.mkdir(exist_ok=True)
CSR = DATA / "csr"; CSR.mkdir(exist_ok=True)
MIRROR = "https://suitesparse-collection-website.herokuapp.com/MM"
PROXY = "http://192.168.48.122:3128"

opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))

def fetch(group, name):
    dst = MTX / f"{group}__{name}.mtx"
    if dst.exists():
        return dst
    url = f"{MIRROR}/{group}/{name}.tar.gz"
    raw = opener.open(url, timeout=180).read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith(".mtx"))
        data = tf.extractfile(member).read()
    dst.write_bytes(data)
    return dst

def descriptors(A):
    A = A.tocsr()
    A.sum_duplicates()
    n = A.shape[0]
    deg = np.diff(A.indptr)
    dmean = deg.mean(); dmax = int(deg.max()); dcv = deg.std() / max(1e-9, dmean)
    # pattern symmetry estimate: nnz(pattern(A) & pattern(A^T)) / nnz
    P = (A != 0).astype(np.int8)
    inter = P.multiply(P.T)
    psym = inter.nnz / max(1, P.nnz)
    return dict(n=n, nnz=int(A.nnz), deg_mean=float(dmean), deg_max=dmax,
                deg_cv=float(dcv), psym_est=float(psym))

def main():
    man = DATA / "manifest.csv"
    rows = list(csv.DictReader(open(man)))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(rows)
    only_smallest = "--smallest" in sys.argv
    if only_smallest:
        rows = sorted(rows, key=lambda r: int(r["nnz"]))[:limit]
    else:
        rows = rows[:limit]
    out = DATA / "descriptors.csv"
    write_header = not out.exists()
    fout = open(out, "a", newline="")
    w = csv.writer(fout)
    if write_header:
        w.writerow(["group", "name", "n", "nnz", "deg_mean", "deg_max",
                    "deg_cv", "psym_est"])
    for r in rows:
        g, nm = r["group"], r["name"]
        tag = f"{g}/{nm}"
        npz = CSR / f"{g}__{nm}.npz"
        try:
            if not npz.exists():
                p = fetch(g, nm)
                A = sio.mmread(str(p)).tocsr().astype(np.float64)
                sp.save_npz(npz, A)
            else:
                A = sp.load_npz(npz)
            d = descriptors(A)
            w.writerow([g, nm, d["n"], d["nnz"], f'{d["deg_mean"]:.2f}',
                        d["deg_max"], f'{d["deg_cv"]:.3f}', f'{d["psym_est"]:.3f}'])
            fout.flush()
            print(f"OK  {tag:34s} n={d['n']:>8} nnz={d['nnz']:>9} "
                  f"deg_max={d['deg_max']:>6} cv={d['deg_cv']:.2f} psym={d['psym_est']:.2f}")
        except Exception as e:
            print(f"ERR {tag:34s} {type(e).__name__}: {e}")
    fout.close()

if __name__ == "__main__":
    main()
