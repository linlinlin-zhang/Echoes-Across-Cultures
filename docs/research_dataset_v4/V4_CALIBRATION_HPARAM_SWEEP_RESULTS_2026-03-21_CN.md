# V4 Calibration Hyperparameter Sweep Results (2026-03-21)

## 1. 目的

这轮补的是一组真正可写入论文的超参实验，不再停留在“只给两个 operating points”。

核心问题是：

- 当 `minority_weight` 逐步提高时，`serendipity / cultural_calibration_kl / minority_exposure_at_k` 会如何联动变化？
- 这种 trade-off 是否只在单一 backbone 上成立，还是在不同 backbone 上都能观察到？

本轮采用的是推理侧 `ot_calibrated` 重排权重 sweep，而不是重新训练大网格模型。原因是：

- 不需要重训 checkpoint，成本低
- 直接对应论文主结论
- 能立即形成可视化曲线和主文表

## 2. 固定设置

所有 sweep 点固定：

- `epsilon = 0.1`
- `iters = 200`
- `novelty_weight = 0.10`

扫的 5 个点为：

| point | relevance | target_affinity | minority | source | diversity_lambda | interpretation |
|---|---:|---:|---:|---:|---:|---|
| `P1` | 0.50 | 0.24 | 0.10 | 0.06 | 0.02 | 偏 target / 保守 |
| `P2` | 0.48 | 0.22 | 0.14 | 0.06 | 0.03 | 当前 `calibrated_target` |
| `P3` | 0.47 | 0.18 | 0.17 | 0.08 | 0.03 | balanced |
| `P4` | 0.46 | 0.14 | 0.22 | 0.08 | 0.04 | 当前 `calibrated_minor` |
| `P5` | 0.44 | 0.12 | 0.26 | 0.08 | 0.04 | ultra-minor |

## 3. 已完成的 sweep

- `V4 routeA_small + CultureMERT`
  - config: [recommender_benchmark_v4_routeA_small_culturemert_stage3_calibration_sweep.run.json](E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v4_routeA_small_culturemert_stage3_calibration_sweep.run.json)
  - output: [benchmark_summary.json](E:/Desktop/Echo/reports/hparam/v4_routeA_small_culturemert_stage3_calibration_sweep/benchmark_summary.json)
- `V4 main + CultureMERT`
  - config: [recommender_benchmark_v4_main_culturemert_stage3_calibration_sweep.run.json](E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v4_main_culturemert_stage3_calibration_sweep.run.json)
  - output: [benchmark_summary.json](E:/Desktop/Echo/reports/hparam/v4_main_culturemert_stage3_calibration_sweep/benchmark_summary.json)
- `V4 routeA_small + Gemini`
  - config: [recommender_benchmark_v4_routeA_small_gemini_stage3_calibration_sweep.run.json](E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v4_routeA_small_gemini_stage3_calibration_sweep.run.json)
  - output: [benchmark_summary.json](E:/Desktop/Echo/reports/hparam/v4_routeA_small_gemini_stage3_calibration_sweep/benchmark_summary.json)

已额外准备但本轮未全量执行：

- `V4 main + Gemini`
  - config: [recommender_benchmark_v4_main_gemini_stage3_calibration_sweep.run.json](E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v4_main_gemini_stage3_calibration_sweep.run.json)

## 4. 结果摘要

### 4.1 V4 routeA_small + CultureMERT

| method | serendipity | calibration_kl | minority@k |
|---|---:|---:|---:|
| `dcas_full_ot` | 0.850135 | 1.143915 | 0.302734 |
| `P1` | 0.837797 | 1.013613 | 0.437370 |
| `P2` | 0.840630 | 1.021299 | 0.509505 |
| `P3` | 0.838404 | 1.043575 | 0.579427 |
| `P4` | 0.837080 | 1.074745 | 0.679818 |
| `P5` | 0.838454 | 1.089537 | 0.752865 |

相对 `dcas_full_ot` 的关键变化：

- `P2`:
  - `serendipity` 下降约 `1.12%`
  - `calibration_kl` 改善约 `10.72%`
  - `minority@k` 提升约 `68.30%`
- `P4`:
  - `serendipity` 下降约 `1.54%`
  - `calibration_kl` 改善约 `6.92%`
  - `minority@k` 提升约 `124.58%`
- `P5`:
  - `serendipity` 下降约 `1.37%`
  - `calibration_kl` 改善约 `4.75%`
  - `minority@k` 提升约 `148.70%`

解读：

- 小数据集上，`minority_weight` 增加后，`minority@k` 单调上升
- `serendipity` 只小幅波动，没有出现“为了公平把推荐质量砸穿”的情况
- `P2` 到 `P4` 已经足够支撑主文中的 trade-off 图

### 4.2 V4 main + CultureMERT

| method | serendipity | calibration_kl | minority@k |
|---|---:|---:|---:|
| `dcas_full_ot` | 0.857861 | 2.082581 | 0.246021 |
| `P1` | 0.837089 | 2.022840 | 0.347917 |
| `P2` | 0.831564 | 2.029638 | 0.402333 |
| `P3` | 0.829856 | 2.039980 | 0.452542 |
| `P4` | 0.828158 | 2.047741 | 0.530271 |
| `P5` | 0.829623 | 2.052705 | 0.583792 |

相对 `dcas_full_ot` 的关键变化：

- `P2`:
  - `serendipity` 下降约 `3.06%`
  - `calibration_kl` 改善约 `2.54%`
  - `minority@k` 提升约 `63.54%`
- `P4`:
  - `serendipity` 下降约 `3.46%`
  - `calibration_kl` 改善约 `1.67%`
  - `minority@k` 提升约 `115.54%`
- `P5`:
  - `serendipity` 下降约 `3.29%`
  - `calibration_kl` 改善约 `1.43%`
  - `minority@k` 提升约 `137.29%`

解读：

- 在主数据集上，这条曲线更像“可控 trade-off”
- `P2` 是比较稳的主文 operating point
- `P4/P5` 适合在附录或 fairness-focused 叙事中使用

### 4.3 V4 routeA_small + Gemini

| method | serendipity | calibration_kl | minority@k |
|---|---:|---:|---:|
| `dcas_full_ot` | 0.860568 | 1.572332 | 0.242318 |
| `P1` | 0.866729 | 1.548636 | 0.435938 |
| `P2` | 0.864198 | 1.550196 | 0.499740 |
| `P3` | 0.861987 | 1.553454 | 0.568620 |
| `P4` | 0.858572 | 1.557353 | 0.652865 |
| `P5` | 0.856613 | 1.559496 | 0.708464 |

相对 `dcas_full_ot` 的关键变化：

- `P2`:
  - `serendipity` 提升约 `0.42%`
  - `calibration_kl` 改善约 `1.41%`
  - `minority@k` 提升约 `106.24%`
- `P4`:
  - `serendipity` 下降约 `0.23%`
  - `calibration_kl` 改善约 `0.95%`
  - `minority@k` 提升约 `169.43%`
- `P5`:
  - `serendipity` 下降约 `0.46%`
  - `calibration_kl` 改善约 `0.82%`
  - `minority@k` 提升约 `192.41%`

解读：

- Gemini 小线上的 sweep 非常干净
- 它证明这条 trade-off 不是 CultureMERT 特有现象
- 在小线 Gemini 上，`P1/P2` 甚至同时改善了 `serendipity` 与 `calibration_kl`

## 5. 论文写法建议

主文推荐放两种 operating point：

- `P2 (target)`:
  - 适合写“在保持总体推荐质量的同时提升 calibration 与 minority exposure”
- `P4 (minor)`:
  - 适合写“当系统更强调少数文化暴露时，可以进一步获得更强公平性结果，代价可控”

最稳的表述方式是：

> Across V4 splits, increasing the minority-oriented reranking weight produces a smooth Pareto-style trade-off. On V4 main with CultureMERT, moving from the uncalibrated OT ranker to the target-calibrated setting improves minority exposure by 63.5% while reducing serendipity by only 3.1%. The same qualitative pattern also holds on Gemini, suggesting that the calibration layer is backbone-agnostic rather than tied to one embedding model.

## 6. 当前边界

- 这轮是推理侧校准权重 sweep，不是训练侧全网格
- `TC/HSIC` 的训练侧敏感性仍建议在 `routeA_small + CultureMERT` 上补一版粗网格，作为 Appendix 稳定性分析
- `V4 main + Gemini` 的完整 sweep 已有配置，但本轮未全量执行

