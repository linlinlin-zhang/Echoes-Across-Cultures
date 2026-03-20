# V4 Gemini Stage3 Benchmark Results

日期: 2026-03-20

## 1. 本轮完成内容

本轮已经把 `Gemini` 这条 backbone 按照 `CultureMERT stage3` 的同构协议完整跑通：

- `V4 routeA_small`
  - tracks: `storage/public/research_dataset_v4/routeA_small/tracks_gemini_embedding2_mw3.npz`
  - train config: `configs/train/train_v4_routeA_small_gemini_stage3.run.json`
  - benchmark config: `configs/benchmark/recommender_benchmark_v4_routeA_small_gemini_stage3_lambdamart.run.json`
  - model: `storage/models/dcas_full_v4_routeA_small_gemini_stage3.pt`
  - result dir: `reports/benchmarks/v4_routeA_small_gemini_stage3_lambdamart/`
- `V4 main`
  - tracks: `storage/public/research_dataset_v4/main/tracks_gemini_embedding2_mw3.npz`
  - train config: `configs/train/train_v4_main_gemini_stage3.run.json`
  - benchmark config: `configs/benchmark/recommender_benchmark_v4_main_gemini_stage3_lambdamart.run.json`
  - model: `storage/models/dcas_full_v4_main_gemini_stage3.pt`
  - result dir: `reports/benchmarks/v4_main_gemini_stage3_lambdamart/`

## 2. Embedding 构建状态

### 2.1 V4 routeA_small

- manifest: `storage/public/research_dataset_v4/routeA_small/tracks_gemini_embedding2_mw3.npz.manifest.json`
- `n_tracks = 640`
- `dim = 768`
- `n_errors = 0`

### 2.2 V4 main

- manifest: `storage/public/research_dataset_v4/main/tracks_gemini_embedding2_mw3.npz.manifest.json`
- `n_tracks = 1122`
- `dim = 768`
- `n_errors = 0`
- `n_cache_hits = 485`

## 3. Benchmark 摘要

### 3.1 V4 routeA_small

result dir: `reports/benchmarks/v4_routeA_small_gemini_stage3_lambdamart/`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.660338 | 1.580523 | 0.000000 | 0.279544 |
| cosine | 0.790129 | 1.581755 | 0.191406 | 0.279284 |
| knn | 0.788096 | 1.580136 | 0.184245 | 0.279701 |
| lightfm_like | 0.684419 | 1.577308 | 0.046094 | 0.280381 |
| bpr_mf | 0.710395 | 1.572263 | 0.095833 | 0.281650 |
| bpr_two_stage_hybrid | 0.727958 | 1.568499 | 0.145182 | 0.282626 |
| bpr_listwise_hybrid | 0.733121 | 1.566272 | 0.165625 | 0.283203 |
| bpr_lambdamart_hybrid | 0.726665 | 1.567630 | 0.151302 | 0.282851 |
| dcas_full_ot | 0.860568 | 1.572332 | 0.242318 | 0.281668 |
| dcas_full_ot_calibrated_target | 0.864198 | 1.550196 | 0.499740 | 0.287335 |
| dcas_full_ot_calibrated_minor | 0.858572 | 1.557353 | 0.652865 | 0.285482 |

reference method: `dcas_full_ot_calibrated_target`

关键点:

- `calibrated_target` 在 `routeA_small` 上取得了整表最高 `serendipity`
- 相比 `dcas_full_ot`, `calibrated_target` 的 `serendipity` 略高 `+0.0036`, 但差异不显著, `p = 0.2488`
- 相比强基线, `calibrated_target` 的 `minority@k` 提升非常明显, 同时 `calibration_kl` 也更低
- `calibrated_minor` 继续把 `minority@k` 推到 `0.652865`, 代价是 `serendipity` 和 `calibration_kl` 略退

### 3.2 V4 main

result dir: `reports/benchmarks/v4_main_gemini_stage3_lambdamart/`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.757620 | 2.329054 | 0.000000 | 0.109537 |
| cosine | 0.851719 | 2.333392 | 0.230479 | 0.108755 |
| knn | 0.854249 | 2.333763 | 0.224563 | 0.108679 |
| lightfm_like | 0.766622 | 2.333701 | 0.116188 | 0.108631 |
| bpr_mf | 0.774738 | 2.321422 | 0.164521 | 0.111048 |
| bpr_two_stage_hybrid | 0.787001 | 2.319638 | 0.302542 | 0.111432 |
| bpr_listwise_hybrid | 0.792204 | 2.317852 | 0.286813 | 0.111787 |
| bpr_lambdamart_hybrid | 0.788364 | 2.318281 | 0.274625 | 0.111697 |
| dcas_full_ot | 0.854250 | 2.325043 | 0.195979 | 0.110348 |
| dcas_full_ot_calibrated_target | 0.824493 | 2.310406 | 0.375979 | 0.113267 |
| dcas_full_ot_calibrated_minor | 0.820860 | 2.312910 | 0.479958 | 0.112763 |

reference method: `dcas_full_ot_calibrated_target`

关键点:

- 在 `main` 上, `Gemini` 线没有表现为“所有指标绝对占优”
- `knn` 和 `dcas_full_ot` 的 `serendipity` 高于 `calibrated_target`
- 但 `calibrated_target` 的 `calibration_kl` 明显更低, 并把 `minority@k` 拉高到 `0.375979`
- `calibrated_minor` 继续把 `minority@k` 提到 `0.479958`, 但 `serendipity` 和 `calibration_kl` 都略退

## 4. 对论文有价值的解释

### 4.1 现在可以较稳地写

- `DCAS calibrated` 的 trade-off 机制不依赖单一 backbone
- 在 `CultureMERT` 与 `Gemini` 两条 backbone 上, `calibrated_target / calibrated_minor` 都能稳定地产生“校准与少数文化暴露提升”的效果
- `Gemini routeA_small` 上, `calibrated_target` 甚至取得了整表最高 `serendipity`

### 4.2 现在仍不宜写得过满

- 任意 backbone 下都能在每个主指标上绝对胜出
- `Gemini main` 已经完全复现 `CultureMERT main` 的优势形态

更准确的表述应是:

- `Gemini` 线验证了系统的可迁移性
- 但不同 backbone 会改变“纯相关性”与“校准/公平性”之间的前沿形状

## 5. 仍需正面留档的问题

1. `LightGBM/sklearn feature names` warning 依然存在
2. 音频解码阶段出现少量 `torchaudio/mpg123` warning
   - 当前命令返回码为 `0`
   - `manifest` 中 `n_errors = 0`
   - 更像数据解码层噪声, 不是本轮构建失败
3. 数据科学性问题没有因 backbone 改变而消失
   - `V4 main` 的 `source confound` 仍需在论文中正面披露
   - `routeA_small` 仍只适合作为小线 sanity-check 与可复现实验补充

## 6. 下一步建议

1. 把 `CultureMERT` 和 `Gemini` 两条 `V4 stage3` 结果并入论文主实验矩阵
2. 单独整理一个 cross-backbone 对比表，聚焦:
   - `serendipity`
   - `cultural_calibration_kl`
   - `minority_exposure_at_k`
3. 继续优先处理 `V4 main` 的 `source confound`
