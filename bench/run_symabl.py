#!/usr/bin/env python3
"""Symmetric A*A^T ablation: plain AAt (transpose + SpGEMM) vs AAtS (upper-tri +
mirror) across both groups. Verifies AAtS produces the same full result as AAt
(nnz + checksum) and reports the AAtS/AAt speedup."""
import sys, csv, subprocess, re, math
from pathlib import Path
import numpy as np, scipy.sparse as sp
PROJ=Path(__file__).resolve().parent.parent
CSR=PROJ/"data"/"csr"; BIN=PROJ/"data"/"bin"; OURS=PROJ/"bench"/"spgemm"
CAP=300_000_000
import struct
def write_bin(A,p):
    A=A.tocsr();A.sort_indices()
    open(p,"wb").write(b"CSR1"+struct.pack("<ii",A.shape[0],0)+struct.pack("<q",A.nnz)
        +A.indptr.astype("<i4").tobytes()+A.indices.astype("<i4").tobytes()+A.data.astype("<f8").tobytes())
def est_aat(A):
    cc=np.bincount(A.indices,minlength=A.shape[1]).astype(np.int64); return int((cc*cc).sum())
def parse(out):
    r=re.search(r"__chk s=(\S+) as=(\S+) sq=(\S+)",out); res=re.search(r"RESULT,[^,]+,[^,]+,\d+,\d+,(\d+),([\d.]+)",out)
    if not r or not res: return None
    return (int(res.group(1)),float(res.group(2)),tuple(float(x) for x in r.groups()))
def rok(a,b): return abs(a-b)<=1e-6*max(1.0,abs(b))
def main():
    tags=[]
    for man in ("manifest_official.csv","manifest.csv"):
        for r in csv.DictReader(open(PROJ/"data"/man)):
            tags.append(f'{r["group"]}__{r["name"]}')
    tags=list(dict.fromkeys(tags))
    out=open(PROJ/"results"/"symabl.csv","w",newline=""); w=csv.writer(out)
    w.writerow(["tag","n","nnz","C_nnz","ms_aat","ms_aats","speedup","match"])
    sp_list=[]; nbad=0
    for tag in tags:
        npz=CSR/f"{tag}.npz"
        if not npz.exists(): continue
        A=sp.load_npz(npz).astype(np.float64).tocsr(); n,nnz=A.shape[0],A.nnz
        if est_aat(A)>CAP: continue
        binf=BIN/f"{tag}.csr"
        if not binf.exists(): write_bin(A,binf)
        reps=20 if nnz<200000 else (8 if nnz<2_000_000 else 3)
        try:
            a=parse(subprocess.run([str(OURS),tag,str(binf),"AAt",str(reps)],capture_output=True,text=True,timeout=600).stdout)
            b=parse(subprocess.run([str(OURS),tag,str(binf),"AAtS",str(reps)],capture_output=True,text=True,timeout=600).stdout)
        except subprocess.TimeoutExpired: continue
        if not a or not b: continue
        match=(a[0]==b[0]) and all(rok(x,y) for x,y in zip(a[2],b[2]))
        spd=a[1]/b[1] if b[1]>0 else 0
        w.writerow([tag,n,nnz,a[0],f"{a[1]:.4f}",f"{b[1]:.4f}",f"{spd:.3f}",int(match)])
        if not match: nbad+=1; print(f"MISMATCH {tag}: AAt nnz={a[0]} chk={a[2]} vs AAtS nnz={b[0]} chk={b[2]}")
        else: sp_list.append((tag,n,nnz,spd))
    out.close()
    gm=math.exp(sum(math.log(s) for _,_,_,s in sp_list)/len(sp_list))
    wins=sum(1 for *_,s in sp_list if s>=1.0)
    print(f"\n== AAtS symmetric ablation: {len(sp_list)} matrices, mismatches={nbad} ==")
    print(f"AAtS/AAt speedup: geomean={gm:.3f}x  median={sorted(s for *_,s in sp_list)[len(sp_list)//2]:.3f}x  "
          f"wins(>=1x)={wins}/{len(sp_list)}  range=[{min(s for *_,s in sp_list):.2f},{max(s for *_,s in sp_list):.2f}]")
    # by size bucket
    for lo,hi,nm in [(0,1e5,"small nnz<1e5"),(1e5,1e6,"med 1e5-1e6"),(1e6,1e12,"large>1e6")]:
        sub=[s for _,_,z,s in sp_list if lo<=z<hi]
        if sub: print(f"  {nm:16s}: n={len(sub)} geomean={math.exp(sum(math.log(x) for x in sub)/len(sub)):.3f}x")
if __name__=="__main__": main()
