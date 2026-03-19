# V3 DCAS Calibrated 闭集增强与 BPR-hybrid Ser 调优记录（2026-03-19）

## 1. 背景

在上一轮 `V3_BPR_HYBRID_TUNING_2026-03-19_CN.md` 中，调优后的 `bpr_two_stage_hybrid` 已经能够在 `CultureMERT stage3` 协议下超过原始 `dcas_full_ot` 的 3 项指标：

- 更低的 `cultural_calibration_kl`
- 更高的 `minority_exposure_at_k`
- 更高的 `target_culture_prob_mean`

但它仍然在 `serendipity` 上显著落后。因此本轮同时推进两条线：

1. 为 `DCAS` 增加闭集校准感知 rerank，争取在尽量保住 `serendipity` 的情况下把其余三项拉起来。
2. 尝试把 `BPR-hybrid` 再往 `serendipity` 方向推进，看能否缩小与 `DCAS` 的 surprise gap。

## 2. 本轮改动

### 2.1 DCAS：新增 calibrated 闭集 rerank

在 `dcas/recommender.py` 中新增：

- `_source_inverse_scores`
- `_recommend_closed_target_rerank`
- `recommend_ot_calibrated`
- `recommend_knn_calibrated`

核心做法是在目标文化闭集候选池内，对 `DCAS` 原始 relevance 再叠加：

- `novelty`
- `target_affinity`
- `minority_boost`
- `source_inverse_frequency`
- 可选 `diversity_lambda`

并在 `dcas/scripts/run_recommender_benchmarks.py` 中新增 `ot_calibrated / knn_calibrated` 两种 benchmark 方法入口。

### 2.2 BPR-hybrid：加入显式 novelty 融合通道

在 `dcas/embedding_recommenders.py` 的 `recommend_embedding_bpr_two_stage_hybrid` 中加入：

- `novelty_weight`

并在 benchmark runner 中接入对应配置项，方便后续用同一套 JSON 做完整复测。

## 3. 快速子集搜索

为了避免每组权重都跑全量 benchmark，本轮先在平衡子集上快速搜索：

- 用户子集：每个 home culture 取 2 个用户，共 20 用户
- 评估单元：20 用户 x 10 target cultures = 200 user-culture evaluations

### 3.1 DCAS calibrated 子集搜索结果

子集基线：

- `dcas_ot_base`: `ser=0.8453`, `KL=2.0445`, `minority=0.2463`, `target=0.1918`
- `bpr_hybrid_tuned`: `ser=0.5213`, `KL=2.0039`, `minority=0.2878`, `target=0.2017`

搜索后最有希望的两组为：

1. `ot_calibrated_target`
- `relevance=0.48`
- `novelty=0.10`
- `target=0.22`
- `minority=0.14`
- `source=0.06`
- `diversity=0.03`

2. `ot_calibrated_minor`
- `relevance=0.46`
- `novelty=0.10`
- `target=0.14`
- `minority=0.22`
- `source=0.08`
- `diversity=0.04`

在子集上，两者都已经同时超过 `bpr_hybrid_tuned` 的四项指标，因此进入全量复测。

### 3.2 BPR-hybrid ser 调优结果

本轮尝试了两类 ser 方向调优：

1. `novelty_weight` 注入
2. 提高 `rerank / recall / bpr` 权重，降低 `target / minority` 权重

结果是两条路都没有超过当前 tuned 版。也就是说，在现有 ranker 下：

- 显式 novelty 融合没有带来更高的 `serendipity`
- 更偏 relevance 的权重配置也没有把 `serendipity` 拉起来

因此本轮没有产生值得进入全量 benchmark 的新 `BPR-hybrid` 候选，完整对照继续保留当前 tuned 版。

## 4. 全量 benchmark

配置文件：

- `configs/benchmark/recommender_benchmark_v3_culturemert_stage3_dcascal.run.json`

输出目录：

- `reports/benchmarks/v3_main_culturemert_stage3_dcascal/`

### 4.1 全量结果

| method | serendipity | cultural_calibration_kl | minority_exposure_at_k | target_culture_prob_mean |
|---|---:|---:|---:|---:|
| `bpr_mf` | 0.4916 | 2.0226 | 0.1491 | 0.1957 |
| `bpr_two_stage_hybrid` | 0.5102 | 2.0082 | 0.2838 | 0.2005 |
| `dcas_full_ot` | 0.8452 | 2.0430 | 0.2398 | 0.1921 |
| `dcas_full_ot_calibrated_target` | 0.8386 | 1.8793 | 0.3814 | 0.2349 |
| `dcas_full_ot_calibrated_minor` | 0.8404 | 1.9148 | 0.5190 | 0.2250 |

### 4.2 相对原始 `dcas_full_ot` 的变化

#### `dcas_full_ot_calibrated_target`

- `serendipity -0.79%`
- `KL +8.01%`，更低更好
- `minority exposure +59.07%`
- `target culture prob +22.28%`

#### `dcas_full_ot_calibrated_minor`

- `serendipity -0.57%`
- `KL +6.27%`，更低更好
- `minority exposure +116.42%`
- `target culture prob +17.15%`

### 4.3 相对 tuned `bpr_two_stage_hybrid` 的变化

#### `dcas_full_ot_calibrated_target`

- `serendipity +64.37%`
- `KL +6.42%`，更低更好
- `minority exposure +34.43%`
- `target culture prob +17.16%`

#### `dcas_full_ot_calibrated_minor`

- `serendipity +64.73%`
- `KL +4.65%`，更低更好
- `minority exposure +82.89%`
- `target culture prob +12.24%`

## 5. 结论

这轮结果很关键，因为它改变了上一轮“`BPR-hybrid` 在三项上压过 `DCAS`”的局面。

新的结论是：

1. `DCAS` 不需要重训，只靠闭集校准感知 rerank，就已经能把 `KL / minority / target_prob` 大幅拉升。
2. 两个 calibrated 候选都只付出了不到 `1%` 的 `serendipity` 代价。
3. 在全量 benchmark 上，两个 calibrated `DCAS` 版本都已经同时超过当前最强 `bpr_two_stage_hybrid` 的四项指标。
4. `BPR-hybrid` 的 `serendipity` 短板在现有排序器下较顽固，简单加 novelty 或调权重都未见有效突破。

## 6. 研究判断

从论文叙事看，本轮最重要的变化不是“`DCAS` 稍微变好了一点”，而是：

`DCAS` 已经从“高 surprise 但在 calibration/minority 上被强混合基线压制”变成了“通过 calibrated rerank 同时保住高 surprise 并反超强混合基线”的系统。

这意味着下一步更值得投入的是：

1. 将 calibrated rerank 固化为 `DCAS` 主结果的一部分。
2. 将 `calibrated_target` 作为更均衡的主线候选。
3. 将 `calibrated_minor` 作为更强调长尾/少数项曝光的 Pareto 分支。
4. 若继续做 `BPR-hybrid`，应优先改训练目标而不是继续手工调 inference 权重。
