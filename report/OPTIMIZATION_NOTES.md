# SpGEMM 优化技巧调研 → 自研核路线图

调研自高性能 SpGEMM 文献 + 高性能 sparse-attention kernel，映射到我们 V1
（warp-per-row 全局 hash）的具体弱点与改法。约束：**目标 A100 = sm_80**，
Hopper-only 特性（TMA multicast、wgmma warp-specialization）不可用；但 **Ampere
的 `cp.async` 异步拷贝可用**。

## 一、来自 SpGEMM 文献的核心技巧

| 技巧 | 出处 | 我们怎么用 |
|---|---|---|
| **两阶段 symbolic + numeric** | nsparse, spECK, opsparse | symbolic 只算每行 distinct 列数（定 C 结构+精确 hash 大小），numeric 再累加值。V1 单趟 hash 用 flops 上界定表 → 过大。symbolic 定精确大小后 numeric 表能塞进 shared memory |
| **行按 flops/输出大小分箱 (binning)，每箱专用 kernel** | nsparse, spECK | 我们最大的杠杆。按每行 flops 分 {thread-per-row 小行 / warp-per-row 中行 / block-per-row 大行 / 全局-hash 超大行}，各用专用配置。**这就是 intra-matrix dispatch**，直接对应 dispatcher 叙事 |
| **shared-memory hash 累加器**（快 atomics + 局部性） | nsparse, HSMU-SpGEMM | 小/中行的 hash 表放 shared memory（sm_80 可开到 164KB dynamic），atomicAdd 走 shared 比全局快一个量级。超出 shared 容量的行才落全局 hash |
| **hash 负载因子 / 乘子调优**（减冲突） | Balanced Hashing, opsparse | 选合适 load factor（~1.33–2×）和乘子（Knuth 2654435761）平衡冲突与占用。开放寻址线性探测 |
| **高 shared 利用率 + 细粒度 kernel 分组** | HSMU-SpGEMM, opsparse | 一个 block 塞多个小行的 hash 表，提高 occupancy；避免一行独占一个 block 浪费 |
| **merge-based 累加器**（对高 fill/长行更优） | bhSPARSE (Liu&Vinter) | 作为 dispatcher 的另一条策略分支：长稠密行用排序-归并而非 hash（hash 冲突严重时） |
| **ML/启发式预测输出大小 & 方法选择** | TACO 2025 (3774654), spECK 轻量分析 | 直接印证我们的 dispatcher。他们用 ML 预测输出 nnz；我们用**廉价结构描述子**（行不平衡 CoV、列度平方和）选策略。related work 锚点 |

## 二、来自 sparse-attention kernel 的可迁移思想（概念迁移，非 Hopper 指令）

| 思想 | 出处 | 迁移到 SpGEMM |
|---|---|---|
| **online / streaming 累加**（不落全量中间结果） | FlashAttention online-softmax | 我们的 hash 累加器本质就是 streaming 累加，避免 ESC 那样物化全部 flops 中间积 |
| **producer-consumer 解耦搬运与计算** | FA3 / tile-skipping 稀疏核 | sm_80 无 TMA，但可用 **`cp.async` 双缓冲**：一批 warp 预取下一段 A 的行块进 shared，计算 warp 并行累加，隐藏不规则 gather 延迟 |
| **按 max_selected_blocks 而非总量分区做负载均衡** | Native Sparse Attn / tile-skipping | 我们按**每行 flops 上界**分箱 + 均衡，避免长尾行拖垮整个 grid（对应 swang1 165ms 病态） |
| **同组共享稀疏结构的 query 一起载入 SRAM** | GQA sparse tiling | A×A 中同一 A[i,k] 的 row-k 会被多行复用 → 可缓存热门 row-k 于 shared 供 block 内多行共享 |

## 三、V1 具体弱点 → 优先级改法

1. **[基准假象+真开销] 每 rep 重新 `cudaMalloc`+`memset` 整个全局 hash 表并计时。**
   → 分配与计时解耦；buffer 预分配复用。**先做，立刻见效。**
2. **[最大真实杠杆] 全行走全局 hash。** → 行按 flops 分箱：小/中行 **shared-memory hash**，
   超大行才全局 hash。预计对小矩阵 regime（Group 1）提升最大。
3. **[表过大] 用 flops 上界定表。** → 加 symbolic 阶段定精确大小，缩小 numeric hash → 更易入 shared。
4. **[全局排序贵] thrust 全局 sort 得 canonical CSR。** → 每行在 shared 内**局部排序**（行内条目少），去掉全局 sort。
5. **[负载不均] 一 warp 一行不分大小。** → thread/warp/block 三档粒度按行大小指派。

## 四、定位（related work，报告用）
我们的自适应 dispatch 与 spECK 的"轻量分析选策略"、TACO'25 的"ML 预测"同源，
差异化 = **纯结构描述子（零训练、零 profiling）+ A×Aᵀ 对称特化 + 大规模真实套件**。
—— 与 DSV4 研究"从廉价静态特征预测每单元策略"是同一 meta-idea 的跨域平移。

### Sources
- Balanced Hashing GPU SpGEMM (Nagasaka/nsparse 系)
- spECK: lightweight analysis (PPoPP'20)
- opsparse (arXiv 2206.07244), HSMU-SpGEMM (IEEE 2024)
- Optimizing General SpGEMM on GPU, TACO 2025 (dl.acm.org/10.1145/3774654)
- FlashAttention 1/2/3；Native Sparse Attention (arXiv 2502.11089)
