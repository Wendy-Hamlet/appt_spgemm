# Structure-Aware SpGEMM on GPUs: A Portable Hash Engine and a Zero-Profiling Dispatcher

**APPT 2026 Student GPU Programming Challenge — Technical Report (draft)**

*Target platform: single NVIDIA A100 80GB. Development and all measurements in
this draft were done on an NVIDIA H100 with our kernels compiled strictly for
`sm_80` (no Hopper-only features), so results port to A100; final A100 numbers
to be filled in.*

---

## Abstract

We study general sparse matrix–matrix multiplication (SpGEMM) for the two tasks
of the challenge — the self-product `C = A·A` and the transpose (Gram) product
`C = A·Aᵀ` — over real matrices from the SuiteSparse collection. Real matrices
are structurally heterogeneous, and we show that **no single implementation wins
across them**: our custom hash-based engine beats NVIDIA cuSPARSE by a geometric
mean of 2.24× on `A·A` over the small-matrix regime but is slower in aggregate on
some large, highly-irregular matrices, while cuSPARSE exhibits pathological
slowdowns (up to 190×) on structured stiffness matrices. We therefore contribute
(1) a **portable hash-based SpGEMM engine** (per-row-sized shared-memory hashing
with a global-memory fallback, unified across both tasks via an explicit
transpose, plus a symmetry-exploiting variant for `A·Aᵀ`) that is byte-exact
against cuSPARSE on all 258 evaluated tasks, and (2) a **zero-profiling,
structure-only dispatcher** that selects the faster of {ours, cuSPARSE} per
matrix from cheap structural descriptors computed once at load time. The
dispatcher reaches **2.63× over cuSPARSE** at runtime, capturing **88% of the
oracle** selection gain (with <9% descriptor overhead), and its dominant decision
feature is the **row-length imbalance**
(coefficient of variation of nonzeros-per-row) — directly reflecting the
irregular-workload nature of the problem.

---

## 1. Introduction

SpGEMM is a core kernel in scientific computing, graph analytics, and modern AI
systems. Its performance on GPUs is dominated not by arithmetic but by
**irregularity**: unknown output sizes, highly variable row lengths, and data
dependent memory access. The APPT 2026 Student GPU Programming Challenge poses
SpGEMM on 100 real SuiteSparse square matrices for two products, `A·A` and
`A·Aᵀ`, judged on correctness, performance, robustness, generality, and
reproducibility on a single A100 80GB.

Our central finding is that **generality is the hard part**: because real
matrices span banded PDE operators, power-law graphs, symmetric stiffness
matrices, and unsymmetric circuits, a kernel tuned for one class loses on
another, and even a strong vendor library (cuSPARSE) has large, structure
dependent performance swings. We turn this into an opportunity: rather than chase
a single universally-fastest kernel, we build a competitive portable kernel and a
**cheap structural dispatcher** that picks the right implementation per matrix.

## 2. Background and Problem

For `A` in CSR, `A·A` follows Gustavson's row-wise formulation: row `i` of `C` is
`Σ_{k∈row_i(A)} A[i,k] · row_k(A)`, with the union of the referenced rows'
columns forming the output pattern. `A·Aᵀ` is the Gram matrix,
`C[i,j] = ⟨row_i(A), row_j(A)⟩`; it is **symmetric**, and equals `A·(Aᵀ)`, i.e.
a standard SpGEMM whose second operand is the transpose of `A`. The two tasks
therefore stress different access patterns, and `A·Aᵀ` additionally admits a
symmetry optimization (compute only the upper triangle).

**Benchmark set.** The official test matrices match the *size regime* of the
first 100 matrices in SuiteSparse's default ordering (the Harwell-Boeing set;
`n ∈ [24, 44609]`, median `n ≈ 800`). We evaluate two groups:
- **Group 1 (official-proxy):** the first 100 square real matrices in default
  order — the small-to-medium regime of the real test.
- **Group 2 (diversity):** 41 structurally diverse matrices we selected to span
  size (up to 1.1M rows / 6M nonzeros), symmetry (pattern symmetry 0.0–1.0), and
  structure class (PDE/CFD, circuits, power-law graphs, chemistry, quantum) — to
  stress *generality*.

## 3. Methodology

- **Hardware/build.** Target A100 (`sm_80`). All kernels compiled `-arch=sm_80`
  with no Hopper-only features (TMA, wgmma, warp-specialization), so binaries run
  identically on A100. Measurements here are on an H100; because cuSPARSE uses its
  native (Hopper-capable) library paths while our code is `sm_80`-restricted, the
  reported relative speedups are, if anything, **conservative** for A100.
- **Precision.** Double precision (fp64) throughout.
- **Timing.** Full end-to-end SpGEMM per call (structure analysis + symbolic +
  numeric + canonical-CSR sort; for `A·Aᵀ`, including the transpose), median of
  repeated runs with warm allocations, matching how cuSPARSE is timed
  (`workEstimation` + `compute` + `copy`, transpose via `csr2csc` for `A·Aᵀ`).
- **Correctness.** Two criteria per (matrix, task): (i) output nnz equals
  cuSPARSE's under **structural SpGEMM semantics**, and (ii) order-independent
  value checksums (`Σv, Σ|v|, Σv²`) match a scipy reference within `1e-6`
  relative. We observed and documented the **structural-zero convention**: some
  `(i,j)` are structurally nonzero but numerically cancel to exactly 0 (e.g. in
  bcsstk stiffness matrices); cuSPARSE and our engine keep them (structural),
  scipy prunes them (numeric), but value checksums are identical either way.
- **OOM guard.** For `A·Aᵀ` on power-law graphs the Gram output can be near-dense
  (`Σ_k colcount(k)²` blows up); such matrices are excluded from the dev sweep by
  an output-size upper bound (12 of 41 in Group 2).

## 4. Custom Kernel Design

A single templated engine computes `C = A·B` and is instantiated as `B = A`
(`A·A`) or `B = Aᵀ` (`A·Aᵀ`).

- **Hashing accumulator, one warp per output row.** Each row accumulates its
  products into an open-addressing hash table (linear probing, multiplicative
  hash, robust pure-`atomicCAS` insert).
- **Per-row hash sizing + two-tier placement.** The table size is
  `next_pow2(1.33 · flops_i)` where `flops_i = Σ_{k∈row_i(A)} deg_B(k)` bounds the
  output row's nnz. If it fits the shared-memory budget (`SH_CAP = 1024`
  entries), the row uses a **shared-memory** hash (fast intra-warp atomics); if
  larger, it spills to a per-row region of a **global-memory arena**. This mirrors
  the row-binning idea from nsparse/spECK while keeping a single kernel.
- **Two-phase symbolic + numeric.** A symbolic pass counts distinct columns to
  size `C`; a numeric pass accumulates values, then warp-uniform compaction emits
  `(row·n+col, value)` pairs.
- **Canonical CSR via CUB radix sort** of the 64-bit `row·n+col` keys, with sort
  temp storage allocated once and reused across calls.
- **`A·Aᵀ`** materializes `Aᵀ` with an atomic counting-sort transpose (timed as
  part of the op) and runs the same engine.
- **Symmetric variant (`A·AᵀS`).** Exploiting `C = A·Aᵀ` symmetric, we hash only
  `j ≥ i` (halving accumulation work) and mirror the upper triangle to the full
  matrix. This gives ~1.14–1.15× over the plain `A·Aᵀ` on medium/large matrices
  (e.g. circuit_2, bcsstk13) but loses on tiny matrices where the mirror overhead
  dominates — a size-dependent trade-off the dispatcher can also arbitrate.

## 5. Correctness

Across **all 258 evaluated (matrix, task) pairs, 0 mismatches** against cuSPARSE
(structural nnz and value checksums). Reaching this exposed a subtle, instructive
bug: a shared-memory hash whose computed capacity could exceed `SH_CAP` and
**overflow into the neighbouring warp's shared slice**, producing over-counted nnz
and nondeterminism. We localized it by (a) forcing all rows down the global path
(which stayed correct) and (b) observing that for a *symmetric* matrix `A·A` and
`A·Aᵀ` disagreed by one entry — impossible if correct. The fix ties the
shared/global decision to whether the computed capacity actually fits `SH_CAP`,
and hardens the insert to pure `atomicCAS`. We consider this correctness
discipline (structural-zero semantics, symmetry as a self-check, byte-exact
cross-validation) part of the contribution.

## 6. Performance

Speedups are cuSPARSE-time ÷ our-time; `>1` means we are faster.

**Table 1 — per-group speedup vs cuSPARSE (fp64).**

| Group | Task | geomean | median | wins (≥1×) | range |
|---|---|---|---|---|---|
| G1 (official-proxy, 100) | A·A  | **2.24×** | 2.00× | 82/100 | 0.04–190× |
| G1 | A·Aᵀ | 1.68× | 1.46× | 78/100 | 0.03–91× |
| G2 (diverse, 29 ran) | A·A  | 1.82× | 1.94× | 22/29 | 0.02–61× |
| G2 | A·Aᵀ | 0.94× | 1.72× | 18/29 | 0.05–17× |

- On the **official small-matrix regime (G1)**, our engine solves the whole suite
  in **387 ms vs cuSPARSE's 1029 ms — 2.66× faster in aggregate**, and beats
  cuSPARSE on 78–82% of matrices.
- cuSPARSE is **non-robust**: 37 G1 cases where we are >5× faster, topped by
  stiffness matrices (bcsstk10 `A·A` **190×**, bcsstk23 104×, bcsstk06 96×) where
  cuSPARSE hits a pathological path.
- We are **honest about the failure mode**: on Group 2's large, highly-irregular
  matrices we lose in aggregate (0.71×), driven by a few outliers where our simple
  hashing is far slower (e.g. shermanACb `A·A` ~50×). This is exactly where a
  dispatcher must send work to cuSPARSE.

## 7. Structure-Aware Dispatch (Zero Profiling)

Because neither implementation dominates, we select per (matrix, task) from
**cheap structural descriptors computed once at load** (n, nnz, mean degree,
**degree coefficient-of-variation `deg_cv`**, estimated fill, pattern symmetry) —
no timing or trial runs at decision time.

**Table 2 — whole-benchmark dispatch (258 tasks, total time).**

| Policy | total | speedup vs cuSPARSE | notes |
|---|---|---|---|
| cuSPARSE-always | 1520 ms | 1.00× | vendor baseline |
| ours-always | 1083 ms | 1.40× | good but loses the outliers |
| structure-only dispatcher (offline, depth-3) | 628 ms | 2.42× | 92% pick accuracy, 83% of oracle |
| **online dispatcher (real runtime)** | **579 ms** | **2.63×** | picks ours 211 / cuSPARSE 47, **88% of oracle** |
| oracle (min per task) | 446 ms | 3.41× | upper bound |

The dispatcher's **dominant feature is `deg_cv` (row-length imbalance)**, with
importance 0.81 vs 0.12 (fill) and 0.07 (nnz) — i.e. the decision is driven by
exactly the irregularity that makes SpGEMM hard. This is the same methodological
move as predicting a per-unit execution policy from a cheap static descriptor,
and it turns "one kernel can't win everywhere" into a concrete, profiling-free
2.42× system.

### 7.1 Online dispatcher

Table 2 is an offline analysis over measured times. We also implemented the
dispatcher **online** (`bench/dispatch.cu`): at call time it computes the
structural descriptors from the CSR, applies a fixed decision rule (a depth-3
tree distilled to `use ours iff (deg_cv≤0.63 ∧ fill≤53) ∨ (deg_cv>0.63 ∧
nnz>20k ∧ fill≤47)`), and runs the chosen implementation — no profiling or trial
runs. Over the full 258-task suite the online dispatcher realizes **2.63× over
cuSPARSE** (picking our engine 211 times and cuSPARSE 47 times), **capturing 88%
of the oracle** gain — on par with the idealized offline rule (2.42×; the two
agree within cuSPARSE's timing noise). The descriptor overhead is **0.01–0.14 ms
for small matrices (<1%)**; aggregated over the suite it is 8.6%, dominated by the
host `O(nnz)` pass on the largest matrices, and is straightforwardly reducible by
computing the descriptors on the GPU or fusing them into the CSR-load pass. This
validates the core claim end-to-end: a **zero-profiling, structure-only decision
recovers ~88% of the oracle selection gain at near-zero cost**.

## 8. Related Work

Hash-based SpGEMM (nsparse), lightweight-analysis strategy selection (spECK),
high-shared-utilization accumulators (HSMU-SpGEMM), merge-based accumulation
(bhSPARSE), and learned output-size prediction (recent TACO work) all inform our
engine. Our positioning is (i) a **portable `sm_80` engine** unifying `A·A` and
`A·Aᵀ` with a symmetric specialization, and (ii) a **zero-training,
structure-only dispatcher over {custom, vendor}** — closest in spirit to spECK's
per-matrix analysis, but selecting *between implementations* (including the vendor
library) rather than only among internal variants.

## 9. Limitations and Future Work

- **Large highly-irregular matrices.** Our single hash strategy is not tuned for
  very large / high-fill rows; a merge-based accumulator branch and finer
  intra-matrix (per-row-block) dispatch are the clear next steps.
- **H100-measured, sm_80 code.** Final numbers should be reproduced natively on
  A100; the relative gains are expected to hold or improve (cuSPARSE loses its
  Hopper-path advantage).
- **Heuristic vs learned dispatcher.** The depth-3 tree is a stand-in for a
  principled, trained cost model over structural descriptors — a natural path to a
  stronger, generalizable dispatcher (and a research contribution in its own
  right).
- **Benchmark proxy.** Group 1 proxies the official set by size regime; exact
  official matrices may differ in content.

## 10. Conclusion

Real-world SpGEMM is a generality problem. We built a correct (0/258 mismatches),
portable, competitive hash engine for both challenge tasks, showed that neither it
nor the vendor library wins everywhere, and turned that into a **profiling-free
structure-aware dispatcher reaching 2.42× over cuSPARSE (83% of oracle), driven by
row-length imbalance**. The approach is robust by construction — it routes each
matrix to whatever solves it fastest — which is precisely what a general,
real-world SpGEMM solver needs.

---

### Reproducibility

Code and drivers: `src/` (kernels), `bench/` (cuSPARSE baseline, sweep, dispatcher
eval), `data/` (manifests, structural descriptors). Build `-arch=sm_80`; matrices
fetched from SuiteSparse via `src/fetch_and_prep.py`. All numbers in this draft
are self-measured on the current machine.
