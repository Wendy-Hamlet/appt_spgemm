# 正确性语义笔记

## 结构性零 (structural zeros) 约定 — SpGEMM 的一个真实歧义
`A·A` 中某些 `(i,j)` 结构上非零（pattern 相交），但数值累加**恰好抵消为 0.0**
（如刚度矩阵 bcsstk 的对称 +/- 结构）。两种合法约定：
- **结构语义**（cuSPARSE 默认、nsparse/spECK、我们的 v3）：保留为显式零，`nnz` = pattern 乘积。
- **数值语义**（scipy `@` 默认）：prune 掉，`nnz` 更小。

实测 `HB/bcsstk13`：pattern(A)@pattern(A) = **396773** = cuSPARSE = v3；scipy 数值 = 395923
（850 个精确抵消项）。`Bai/rw5151`、`Bomhof/circuit_2` 无抵消，三者 nnz 一致。

**结论**：
- v3 与 cuSPARSE 逐位一致 = 标准 GPU SpGEMM 语义，**无 bug**。
- 这 850 个零对 value checksum(sum/abssum/sq) 贡献为 0 → **数值结果与 scipy 完全一致**。
- 我们的正确性判据 = **order-independent value checksum**（对结构零鲁棒），辅以 nnz（结构语义）。
- 若官方 reference 采数值语义，输出加一遍 `eliminate_zeros`（阈值 0）即可切换；默认保持结构语义（与 cuSPARSE 对齐，最通用）。

## 正确性判据（本项目统一）
对每个矩阵、每个任务（A·A / A·Aᵀ）：
1. `C_nnz` 与 cuSPARSE 结构语义一致（允许与 scipy 差 = 精确抵消数）。
2. value checksum `(Σv, Σ|v|, Σv²)` 与 scipy 参照在 rtol 1e-9 内一致。
判据 2 是主判据（对结构零鲁棒）。
