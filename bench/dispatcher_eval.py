#!/usr/bin/env python3
"""Evaluate a structure-only dispatcher that picks {ours, cuSPARSE} per (matrix,
task) from cheap structural descriptors -- no timing/profiling at decision time.
Compares against ours-always, cuSPARSE-always, and the oracle (min per task).
"""
import csv, math
from pathlib import Path
import numpy as np, scipy.sparse as sp

PROJ = Path(__file__).resolve().parent.parent
CSR = PROJ / "data" / "csr"

def feats(tag):
    A = sp.load_npz(CSR / f"{tag}.npz").astype(np.float64).tocsr()
    n = A.shape[0]; nnz = A.nnz
    deg = np.diff(A.indptr).astype(np.float64)
    dmean = deg.mean(); dcv = deg.std()/max(1e-9,dmean)
    cc = np.bincount(A.indices, minlength=n).astype(np.int64)
    est_aa = int((deg[A.indices]).sum()); est_aat = int((cc*cc).sum())
    P=(A!=0).astype(np.int8); inter=P.multiply(P.T); psym=inter.nnz/max(1,P.nnz)
    return dict(n=n, nnz=nnz, deg=dmean, dcv=dcv,
                fill_aa=est_aa/max(1,nnz), fill_aat=est_aat/max(1,nnz), psym=psym)

def load():
    rows=[]
    for g in ("group1","group2"):
        for r in csv.DictReader(open(PROJ/"results"/f"compare_{g}.csv")):
            if r["status"]!="ok": continue
            rows.append(dict(group=g, tag=r["tag"], task=r["task"],
                             mo=float(r["ms_ours"]), mc=float(r["ms_cus"])))
    return rows

def report(rows, name, pick):
    """pick(r,f)->'ours'|'cus'. return total ms."""
    fcache={}
    tot=0; noracle=0; correct=0
    for r in rows:
        f=fcache.setdefault(r["tag"], feats(r["tag"]))
        ch=pick(r,f)
        t=r["mo"] if ch=="ours" else r["mc"]
        tot+=t
        best="ours" if r["mo"]<=r["mc"] else "cus"
        if ch==best: correct+=1
    return tot, correct

def main():
    rows=load()
    fcache={tag:feats(tag) for tag in {r["tag"] for r in rows}}
    tc=sum(r["mc"] for r in rows); to=sum(r["mo"] for r in rows)
    tor=sum(min(r["mo"],r["mc"]) for r in rows)
    print(f"tasks={len(rows)}")
    print(f"cuSPARSE-always : {tc:8.1f}ms  (1.00x)")
    print(f"ours-always     : {to:8.1f}ms  ({tc/to:.2f}x)")
    print(f"ORACLE          : {tor:8.1f}ms  ({tc/tor:.2f}x)  [upper bound]")
    # candidate structure-only rules
    rules={
      "nnz<2e5 -> ours": lambda r,f: "ours" if f["nnz"]<2e5 else "cus",
      "fill<threshold": lambda r,f: "ours" if (f["fill_aa"] if r["task"]=="AA" else f["fill_aat"])<40 else "cus",
      "small OR low-fill": lambda r,f: "ours" if (f["nnz"]<1e5 or (f["fill_aa"] if r["task"]=="AA" else f["fill_aat"])<25) else "cus",
    }
    for nm,rule in rules.items():
        t,corr=report(rows,nm,rule)
        print(f"rule[{nm:22s}]: {t:8.1f}ms  ({tc/t:.2f}x)  acc={100*corr/len(rows):.0f}%  captures {100*(tc-t)/(tc-tor):.0f}% of oracle gain")
    # shallow decision tree (learned) as best-simple-rule ceiling
    try:
        from sklearn.tree import DecisionTreeClassifier
        X=[]; y=[]; keys=["n","nnz","deg","dcv","fill_aa","fill_aat","psym"]
        for r in rows:
            f=fcache[r["tag"]]
            X.append([f[k] for k in keys]); y.append(0 if r["mo"]<=r["mc"] else 1)
        X=np.array(X); y=np.array(y)
        clf=DecisionTreeClassifier(max_depth=3,random_state=0).fit(X,y)
        pred=clf.predict(X)
        t=sum((r["mo"] if p==0 else r["mc"]) for r,p in zip(rows,pred))
        print(f"tree(depth3)    : {t:8.1f}ms  ({tc/t:.2f}x)  acc={100*(pred==y).mean():.0f}%  captures {100*(tc-t)/(tc-tor):.0f}% of oracle")
        imp={k:round(v,2) for k,v in zip(keys,clf.feature_importances_) if v>0.05}
        print(f"  top features: {imp}")
    except ImportError:
        print("(sklearn not available; skipping learned tree)")

if __name__=="__main__":
    main()
