#!/usr/bin/env python3
"""Head-to-head sweep: our custom SpGEMM vs cuSPARSE, tasks AA & AAt, over a
matrix group. Cross-checks correctness (our checksums vs cuSPARSE's -- both use
structural SpGEMM semantics, so they must match) and records speedups.

Usage: run_compare.py <group1|group2> [--limit N]
  group1 -> data/manifest_official.csv   group2 -> data/manifest.csv
"""
import sys, csv, struct, subprocess, re
from pathlib import Path
import numpy as np
import scipy.sparse as sp

PROJ = Path(__file__).resolve().parent.parent
CSR = PROJ / "data" / "csr"; BIN = PROJ / "data" / "bin"; RES = PROJ / "results"
OURS = PROJ / "bench" / "spgemm"; CUS = PROJ / "bench" / "cusparse_baseline"
CAP = 300_000_000

def write_bin(A, path):
    A = A.tocsr(); A.sort_indices()
    with open(path, "wb") as f:
        f.write(b"CSR1"); f.write(struct.pack("<ii", A.shape[0], 0)); f.write(struct.pack("<q", A.nnz))
        f.write(A.indptr.astype("<i4").tobytes()); f.write(A.indices.astype("<i4").tobytes()); f.write(A.data.astype("<f8").tobytes())

def est(A):
    A = A.tocsr(); deg = np.diff(A.indptr).astype(np.int64)
    cc = np.bincount(A.indices, minlength=A.shape[1]).astype(np.int64)
    return int(deg[A.indices].sum()), int((cc*cc).sum())

def parse(out):
    """return {task: (C_nnz, ms, (s,as,sq))} pairing __chk lines with RESULTs in order."""
    chks = re.findall(r"__chk s=(\S+) as=(\S+) sq=(\S+)", out)
    res = {}
    ci = 0
    for ln in out.splitlines():
        if ln.startswith("RESULT,"):
            f = ln.split(",")  # RESULT,tag,task,n,nnz,C_nnz,ms
            chk = tuple(float(x) for x in chks[ci]) if ci < len(chks) else (0, 0, 0)
            res[f[2]] = (int(f[5]), float(f[6]), chk); ci += 1
    return res

def rel_ok(a, b, rt=1e-6):
    return abs(a - b) <= rt * max(1.0, abs(b))

def main():
    group = sys.argv[1] if len(sys.argv) > 1 else "group1"
    man = PROJ / "data" / ("manifest_official.csv" if group == "group1" else "manifest.csv")
    rows = list(csv.DictReader(open(man)))
    if "--limit" in sys.argv:
        rows = rows[:int(sys.argv[sys.argv.index("--limit")+1])]
    out = RES / f"compare_{group}.csv"
    w = csv.writer(open(out, "w", newline=""));
    w.writerow(["tag","n","nnz","task","C_nnz_ours","C_nnz_cus","ms_ours","ms_cus",
                "speedup","chk_match","status"])
    n_ok=n_bad=n_skip=0
    for r in rows:
        tag = f'{r["group"]}__{r["name"]}'
        npz = CSR / f"{tag}.npz"
        if not npz.exists():
            print(f"MISS {tag} (no npz)"); continue
        A = sp.load_npz(npz).astype(np.float64).tocsr()
        n, nnz = A.shape[0], A.nnz
        e_aa, e_aat = est(A)
        binf = BIN / f"{tag}.csr"
        if not binf.exists(): write_bin(A, binf)
        if max(e_aa, e_aat) > CAP:
            for t in ("AA","AAt"): w.writerow([tag,n,nnz,t,"","","","","","","skip_oom"])
            n_skip+=1; print(f"SKIP {tag}"); continue
        reps = 20 if nnz < 200_000 else (8 if nnz < 2_000_000 else 3)
        try:
            cu = parse(subprocess.run([str(CUS),tag,str(binf),str(reps)],capture_output=True,text=True,timeout=600).stdout)
            our = {}
            for t in ("AA","AAt"):
                o = subprocess.run([str(OURS),tag,str(binf),t,str(reps)],capture_output=True,text=True,timeout=600)
                our.update(parse(o.stdout))
        except subprocess.TimeoutExpired:
            for t in ("AA","AAt"): w.writerow([tag,n,nnz,t,"","","","","","","timeout"]);
            print(f"TIMEOUT {tag}"); continue
        for t in ("AA","AAt"):
            if t in our and t in cu:
                on,om,oc = our[t]; cn,cm,cc = cu[t]
                match = (on==cn) and all(rel_ok(a,b) for a,b in zip(oc,cc))
                spd = cm/om if om>0 else 0
                w.writerow([tag,n,nnz,t,on,cn,f"{om:.4f}",f"{cm:.4f}",f"{spd:.2f}",int(match),"ok"])
                n_ok+=1;
                if not match: n_bad+=1; print(f"  MISMATCH {tag} {t}: ours nnz={on} chk={oc} vs cus nnz={cn} chk={cc}")
            else:
                w.writerow([tag,n,nnz,t,"","","","","","","fail"]); print(f"  FAIL {tag} {t}")
        print(f"OK {tag:28s} AA spd={our.get('AA',[0,0])[1] and cu.get('AA',[0,1])[1]/our['AA'][1]:.2f}x "
              f"AAt spd={(cu.get('AAt',[0,1])[1]/our['AAt'][1]) if our.get('AAt',[0,0])[1] else 0:.2f}x")
    print(f"\n== {group}: ok_tasks={n_ok} mismatches={n_bad} skipped={n_skip} -> {out}")

if __name__ == "__main__":
    main()
