#!/usr/bin/env python3
"""Export CSR .npz matrices to a simple binary format the CUDA baseline reads,
and also dump the scipy reference results (C=A*A and C=A*A^T) nnz + checksum for
correctness cross-checking.

Binary CSR format (little-endian):
  char[4] "CSR1"; int32 n; int32 pad; int64 nnz;
  int32 indptr[n+1]; int32 indices[nnz]; double data[nnz]
"""
import sys, struct
from pathlib import Path
import numpy as np
import scipy.sparse as sp

PROJ = Path(__file__).resolve().parent.parent
CSR = PROJ / "data" / "csr"
BIN = PROJ / "data" / "bin"; BIN.mkdir(exist_ok=True)
REF = PROJ / "data" / "reference.csv"

def write_bin(A, path):
    A = A.tocsr(); A.sort_indices()
    n = A.shape[0]; nnz = A.nnz
    with open(path, "wb") as f:
        f.write(b"CSR1")
        f.write(struct.pack("<ii", n, 0))
        f.write(struct.pack("<q", nnz))
        f.write(A.indptr.astype("<i4").tobytes())
        f.write(A.indices.astype("<i4").tobytes())
        f.write(A.data.astype("<f8").tobytes())

def checksum(C):
    """order-independent numeric summary for cross-impl correctness."""
    C = C.tocsr(); C.sum_duplicates()
    v = C.data
    return int(C.nnz), float(v.sum()), float(np.abs(v).sum()), float((v * v).sum())

def main():
    npzs = sorted(CSR.glob("*.npz"))
    if len(sys.argv) > 1:
        npzs = npzs[:int(sys.argv[1])]
    rows = []
    for p in npzs:
        tag = p.stem
        A = sp.load_npz(p).astype(np.float64).tocsr()
        write_bin(A, BIN / f"{tag}.csr")
        AA = (A @ A)
        AAt = (A @ A.T)
        c1 = checksum(AA); c2 = checksum(AAt)
        rows.append((tag, A.shape[0], A.nnz, *c1, *c2))
        print(f"{tag:34s} n={A.shape[0]:>8} nnz={A.nnz:>9} "
              f"AA_nnz={c1[0]:>10} AAt_nnz={c2[0]:>10}")
    with open(REF, "w") as f:
        f.write("tag,n,nnz,aa_nnz,aa_sum,aa_abssum,aa_sq,"
                "aat_nnz,aat_sum,aat_abssum,aat_sq\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"wrote {len(rows)} matrices -> {BIN} and {REF}")

if __name__ == "__main__":
    main()
