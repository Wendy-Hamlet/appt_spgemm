# APPT 2026 — Student GPU Programming Challenge (SpGEMM)

Structure-aware adaptive **SpGEMM** for the APPT 2026 Student GPU Programming
Challenge: for 100 real sparse square matrices, compute both `C = A·A`
(self-product) and `C = A·Aᵀ` (transpose product) on a single **NVIDIA A100
80GB**, judged on correctness, performance, robustness, generality, and
reproducibility. Implementation in CUDA (dev on H100 restricted to `sm_80`
features so results port to A100).

## Idea

Real SuiteSparse matrices are structurally heterogeneous (banded PDE vs
power-law graphs, symmetric vs unsymmetric, low vs high fill-in); no single
kernel wins across all of them. We compute cheap structural descriptors in a
light pre-pass and **dispatch each matrix to a matched (format, accumulator,
load-balance) strategy**, plus a symmetry-exploiting kernel for `A·Aᵀ` (Gram,
upper-triangle only). Baseline: cuSPARSE `cusparseSpGEMM`.

## Layout

| dir | contents |
|---|---|
| `src/` | matrix selection, download/prep, CUDA kernels, dispatcher |
| `data/` | `ssstats.csv`, `manifest.csv`, `descriptors.csv` (matrix bodies git-ignored) |
| `bench/` | benchmark drivers, correctness harness |
| `results/` | summarized CSVs, figures |
| `report/` | technical report |

## Reproduce the benchmark set

```bash
python src/select_matrices.py          # -> data/manifest.csv (diverse subset)
python src/fetch_and_prep.py           # download + CSR + structural descriptors
```
