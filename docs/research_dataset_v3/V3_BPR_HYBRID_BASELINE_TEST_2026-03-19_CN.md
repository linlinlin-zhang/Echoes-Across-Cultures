# V3 BPR 两阶段混合基线测试记录
更新日期：2026-03-19

## 1. 目标

在确认 `BPR-MF` 是当前最值得认真对待的经典协同过滤对手后，继续沿着“更像主流平台混合推荐”的方向推进：

- 第一阶段保留 `BPR` 作为候选召回主干
- 第二阶段增加 learned reranker
- 训练目标从点式分类改成 pairwise ranking

这条线的目的，是构建一个比前一版 `two_stage_hybrid_ranker` 更强、也比纯 `BPR` 更接近平台风格的代理基线。

## 2. 实现方式

代码位置：

- `E:/Desktop/Echo/dcas/embedding_recommenders.py`
- `E:/Desktop/Echo/dcas/scripts/run_recommender_benchmarks.py`

配置：

- `E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v3_culturemert_stage3_bprhybrid.run.json`

核心设计：

1. 召回主干
- 使用已有 `BPR-MF` checkpoint
- 对目标文化候选池计算 `BPR score`
- 再与 `cosine / knn / novelty / minority / target-affinity / source preference` 做加权融合，形成候选召回分数

2. 第二阶段 learned reranker
- 沿用 `MLP` 型 ranker 结构
- 输入不再只是手工分数，而是：
  - `10` 维候选标量特征
  - `BPR score`
  - 融合后的 recall score
  - `user/item/abs-diff/product` 交互特征

3. 训练目标
- 不再用点式 `BCE`
- 改为 pairwise `-logsigmoid(pos - neg)`
- hard negatives 优先从高召回候选中采样

## 3. 结果

benchmark 产物：

- `E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert_stage3_bprhybrid/benchmark_summary.json`
- `E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert_stage3_bprhybrid/benchmark_table.md`

`bpr_two_stage_hybrid` 最终指标：

- `serendipity = 0.5067`
- `KL = 2.0586`
- `minority = 0.2023`
- `target_prob = 0.1879`

## 4. 与 BPR / 旧 hybrid / DCAS 对比

相对 `BPR-MF`：

- `serendipity +3.08%`
- `KL -1.78%`
- `minority exposure +35.65%`
- `target_prob -4.01%`

这里 `KL -1.78%` 表示新方法更差，因为 `KL` 越低越好。

这说明：

- 新方法确实把 `BPR` 往“更有惊喜度、更有长尾曝光”的方向推了一步
- 但代价是 `cultural calibration` 和 `target culture affinity` 回落

相对仓库原有 heuristic hybrid：

- `serendipity -5.61%`
- `KL +4.86%`
- `minority exposure +188.93%`
- `target_prob +12.97%`

这说明：

- 它在 `serendipity` 上还没超过 heuristic hybrid
- 但在 `KL / minority / target_prob` 三项上已经更强

相对 `DCAS stage3 / dcas_full_ot`：

- `serendipity -40.05%`
- `KL -0.76%`
- `minority exposure -15.63%`
- `target_prob -2.18%`

也就是说：

- 这条新基线已经比之前几条强得多
- 但仍没有追上 `DCAS stage3`
- 真正接近 `DCAS` 的，是 `KL` 和 `target_prob`
- 差距最大的仍然是 `serendipity`

## 5. 判断

这轮结果比 `LightFM-like` 和第一版 `strong hybrid` 更有价值，原因是它终于展现出一种“更像平台混合推荐”的 trade-off：

- 相对 `BPR`，它能显著换来更多 `serendipity` 和 `minority exposure`
- 相对 heuristic hybrid，它又保住了更好的 `KL` 和 `target_prob`

但它仍然没有形成对 `DCAS` 的实质反超，说明当前 `DCAS` 的优势不是简单建立在弱基线上，而是真正体现在跨文化惊喜度和长尾暴露上。

## 6. 下一步

这条线值得继续，但下一步不该再只是“盲调权重”，而应优先做：

1. 给 reranker 显式加入 calibration / target-aware 约束
- 减少它相对 `BPR` 的 `KL` 与 `target_prob` 回落

2. 做更标准的 listwise / LambdaRank 风格目标
- 继续把 pairwise ranker 向真正学习排序推进

3. 如果继续补更强平台代理
- 优先考虑 `GBDT / LambdaMART` 风格 reranker
- 而不是继续沿 `LightFM-like` 路线投入
