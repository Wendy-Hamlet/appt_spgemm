#!/usr/bin/env python3
"""Run the online dispatcher over the same (matrix,task) set as the compare
sweep, and compare realized total time (descriptor overhead + chosen run)
against ours-always / cuSPARSE-always / oracle from the compare CSVs."""
import csv, subprocess, re
from pathlib import Path
PROJ=Path(__file__).resolve().parent.parent
DISP=PROJ/"bench"/"dispatch"; BIN=PROJ/"data"/"bin"
rows=[]
for g in ("group1","group2"):
    for r in csv.DictReader(open(PROJ/"results"/f"compare_{g}.csv")):
        if r["status"]=="ok": rows.append(r)
tot_disp=tot_desc=0.0; tc=to=torc=0.0; n_ours=0
out=open(PROJ/"results"/"dispatch.csv","w",newline=""); w=csv.writer(out)
w.writerow(["tag","task","choice","desc_ms","run_ms","total_ms","ms_ours","ms_cus"])
for r in rows:
    tag,task=r["tag"],r["task"]; binf=BIN/f"{tag}.csr"
    reps=20 if int(r["nnz"])<200000 else (8 if int(r["nnz"])<2000000 else 3)
    p=subprocess.run([str(DISP),tag,str(binf),task,str(reps)],capture_output=True,text=True,timeout=600)
    m=re.search(r"DISPATCH,[^,]+,[^,]+,(\w+),dcv=([\d.]+),fill=([\d.]+),C_nnz=(\d+),desc_ms=([\d.]+),run_ms=([\d.]+),total_ms=([\d.]+)",p.stdout)
    if not m: print("FAIL",tag,task,p.stdout[:120],p.stderr[:120]); continue
    choice=m.group(1); desc=float(m.group(5)); run=float(m.group(6)); total=float(m.group(7))
    mo=float(r["ms_ours"]); mc=float(r["ms_cus"])
    tot_disp+=total; tot_desc+=desc; tc+=mc; to+=mo; torc+=min(mo,mc)
    if choice=="ours": n_ours+=1
    w.writerow([tag,task,choice,f"{desc:.4f}",f"{run:.4f}",f"{total:.4f}",f"{mo:.4f}",f"{mc:.4f}"])
out.close()
N=len(rows)
print(f"tasks={N}  dispatcher picked ours {n_ours}, cuSPARSE {N-n_ours}")
print(f"cuSPARSE-always : {tc:8.1f} ms  (1.00x)")
print(f"ours-always     : {to:8.1f} ms  ({tc/to:.2f}x)")
print(f"ONLINE dispatch : {tot_disp:8.1f} ms  ({tc/tot_disp:.2f}x)   [incl {tot_desc:.1f}ms descriptor overhead = {100*tot_desc/tot_disp:.1f}%]")
print(f"oracle          : {torc:8.1f} ms  ({tc/torc:.2f}x)")
print(f"-> online captures {100*(tc-tot_disp)/(tc-torc):.0f}% of oracle gain")
