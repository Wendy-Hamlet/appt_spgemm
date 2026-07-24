#!/usr/bin/env python3
"""Full cuSPARSE baseline sweep over a matrix group.

For each CSR matrix: ensure the binary export exists, cheaply estimate the
output nnz upper bound for both tasks (to skip matrices that would OOM an 80GB
GPU in this dev sweep), then run the compiled cusparse_baseline and collect
timings + C_nnz.

Output-nnz upper bounds (pre-dedup, exact ceilings):
  A*A  : sum over nonzeros (i,k) of deg_row(k)        (row expansion)
  A*A^T: sum over columns k of colcount(k)^2          (column outer products)
A*A^T can be far denser (Gram of a high-degree column -> dense block).
"""
import sys, csv, struct, subprocess, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp

PROJ = Path(__file__).resolve().parent.parent
CSR = PROJ / "data" / "csr"
BIN = PROJ / "data" / "bin"; BIN.mkdir(exist_ok=True)
RES = PROJ / "results"; RES.mkdir(exist_ok=True)
BINARY = PROJ / "bench" / "cusparse_baseline"

# skip if either task's estimated output nnz upper bound exceeds this
# (fp64 C ~ 12 B/nnz + cuSPARSE intermediate buffers ~ a few x). 300M -> ~4GB C.
CAP = 300_000_000

def write_bin(A, path):
    A = A.tocsr(); A.sort_indices()
    n, nnz = A.shape[0], A.nnz
    with open(path, "wb") as f:
        f.write(b"CSR1"); f.write(struct.pack("<ii", n, 0)); f.write(struct.pack("<q", nnz))
        f.write(A.indptr.astype("<i4").tobytes())
        f.write(A.indices.astype("<i4").tobytes())
        f.write(A.data.astype("<f8").tobytes())

def est_outputs(A):
    A = A.tocsr()
    deg = np.diff(A.indptr).astype(np.int64)          # row lengths
    est_aa = int(deg[A.indices].sum())                # sum_{(i,k)} deg(k)
    colcount = np.bincount(A.indices, minlength=A.shape[1]).astype(np.int64)
    est_aat = int((colcount * colcount).sum())        # sum_k colcount(k)^2
    return est_aa, est_aat

def main():
    group = sys.argv[1] if len(sys.argv) > 1 else "group2"
    npzs = sorted(CSR.glob("*.npz"))
    out = RES / f"baseline_{group}.csv"
    w = csv.writer(open(out, "w", newline=""))
    w.writerow(["tag", "n", "nnz", "est_aa_ub", "est_aat_ub",
                "task", "C_nnz", "ms", "status"])
    for p in npzs:
        tag = p.stem
        A = sp.load_npz(p).astype(np.float64).tocsr()
        n, nnz = A.shape[0], A.nnz
        est_aa, est_aat = est_outputs(A)
        binf = BIN / f"{tag}.csr"
        if not binf.exists():
            write_bin(A, binf)
        reps = 20 if nnz < 200_000 else (8 if nnz < 2_000_000 else 3)
        if max(est_aa, est_aat) > CAP:
            for task, est in [("AA", est_aa), ("AAt", est_aat)]:
                w.writerow([tag, n, nnz, est_aa, est_aat, task, "", "", "skip_oom"])
            print(f"SKIP {tag:34s} est_aa={est_aa:.2e} est_aat={est_aat:.2e} > cap")
            continue
        try:
            t0 = time.time()
            r = subprocess.run([str(BINARY), tag, str(binf), str(reps)],
                               capture_output=True, text=True, timeout=600)
            res = {}
            for ln in r.stdout.splitlines():
                if ln.startswith("RESULT,"):
                    f = ln.split(",")   # RESULT,tag,task,n,nnz,C_nnz,ms
                    res[f[2]] = (int(f[5]), float(f[6]))
            for task in ("AA", "AAt"):
                if task in res:
                    cnnz, ms = res[task]
                    w.writerow([tag, n, nnz, est_aa, est_aat, task, cnnz, f"{ms:.4f}", "ok"])
                    print(f"OK   {tag:34s} {task:4s} C_nnz={cnnz:>11} {ms:8.3f} ms")
                else:
                    w.writerow([tag, n, nnz, est_aa, est_aat, task, "", "", "fail"])
                    print(f"FAIL {tag:34s} {task:4s} (rc={r.returncode}) {r.stderr.strip()[:80]}")
        except subprocess.TimeoutExpired:
            for task in ("AA", "AAt"):
                w.writerow([tag, n, nnz, est_aa, est_aat, task, "", "", "timeout"])
            print(f"TIMEOUT {tag}")
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
