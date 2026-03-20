# V4 CultureMERT Stage3 Benchmark Results

日期: 2026-03-20

## 1. 本轮补齐内容

本轮已经把 `V4 main` 补齐到与 `V4 routeA_small` 同构的主实验协议:

- backbone: `CultureMERT`
- tracks: `mw3`
- training: `stage3`
- reranker: `LambdaMART`

已完成产物:

- `V4 routeA_small`
  - train config: `configs/train/train_v4_routeA_small_culturemert_stage3.run.json`
  - benchmark config: `configs/benchmark/recommender_benchmark_v4_routeA_small_culturemert_stage3_lambdamart.run.json`
  - model: `storage/models/dcas_full_v4_routeA_small_culturemert_stage3.pt`
  - result dir: `reports/benchmarks/v4_routeA_small_culturemert_stage3_lambdamart/`
- `V4 main`
  - train config: `configs/train/train_v4_main_culturemert_stage3.run.json`
  - benchmark config: `configs/benchmark/recommender_benchmark_v4_main_culturemert_stage3_lambdamart.run.json`
  - model: `storage/models/dcas_full_v4_main_culturemert_stage3.pt`
  - result dir: `reports/benchmarks/v4_main_culturemert_stage3_lambdamart/`

论文正文应优先使用 `stage3_lambdamart` 结果。较早生成的 simplified benchmark 目录只建议保留为补充对照，不应与主线结果混写。

## 2. 训练完成状态

### 2.1 V4 routeA_small

- checkpoint: `storage/models/dcas_full_v4_routeA_small_culturemert_stage3.pt`
- constraints: `800`
- ranking examples: `3840`
- tracks input: `storage/public/research_dataset_v4/routeA_small/tracks_culturemert_mw3.npz`

### 2.2 V4 main

- checkpoint: `storage/models/dcas_full_v4_main_culturemert_stage3.pt`
- constraints: `1200`
- ranking examples: `9600`
- tracks input: `storage/public/research_dataset_v4/main/tracks_culturemert_mw3.npz`

## 3. Benchmark 摘要

### 3.1 V4 routeA_small

result dir: `reports/benchmarks/v4_routeA_small_culturemert_stage3_lambdamart/`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.386587 | 1.120704 | 0.000000 | 0.428283 |
| cosine | 0.430372 | 1.160764 | 0.212109 | 0.421012 |
| knn | 0.445182 | 1.174174 | 0.242448 | 0.416954 |
| lightfm_like | 0.384640 | 1.148551 | 0.119401 | 0.421706 |
| bpr_mf | 0.481093 | 1.129262 | 0.087891 | 0.427195 |
| bpr_two_stage_hybrid | 0.500245 | 1.126419 | 0.178776 | 0.429204 |
| bpr_listwise_hybrid | 0.504103 | 1.119935 | 0.192188 | 0.431334 |
| bpr_lambdamart_hybrid | 0.501788 | 1.113629 | 0.157292 | 0.433312 |
| dcas_full_ot | 0.850135 | 1.143915 | 0.302734 | 0.424368 |
| dcas_full_ot_calibrated_target | 0.840630 | 1.021299 | 0.509505 | 0.463446 |
| dcas_full_ot_calibrated_minor | 0.837080 | 1.074745 | 0.679818 | 0.445691 |

reference method: `dcas_full_ot_calibrated_target`

关键点:

- 相比最强 raw/hybrid 基线, `dcas_full_ot_calibrated_target` 在 `serendipity` 上提升约 `+0.3365 ~ +0.4560`
- 相比 `dcas_full_ot`, `calibrated_target` 的 `serendipity` 略降 `-0.0095`, 但 `calibration_kl` 改善 `-0.1226`, `minority@k` 提升 `+0.2068`
- 相比 `calibrated_target`, `calibrated_minor` 在 `minority@k` 继续提升 `+0.1703`, 但 `serendipity` 优势不显著, `p = 0.0796`

### 3.2 V4 main

result dir: `reports/benchmarks/v4_main_culturemert_stage3_lambdamart/`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.501464 | 2.173411 | 0.000000 | 0.151525 |
| cosine | 0.633324 | 2.233377 | 0.220667 | 0.141885 |
| knn | 0.644522 | 2.234130 | 0.213063 | 0.141954 |
| lightfm_like | 0.502662 | 2.185023 | 0.133625 | 0.149577 |
| bpr_mf | 0.537290 | 2.114678 | 0.164625 | 0.165561 |
| bpr_two_stage_hybrid | 0.554085 | 2.107243 | 0.260354 | 0.167769 |
| bpr_listwise_hybrid | 0.561431 | 2.098711 | 0.276833 | 0.169995 |
| bpr_lambdamart_hybrid | 0.555827 | 2.096581 | 0.268083 | 0.170585 |
| dcas_full_ot | 0.857861 | 2.082581 | 0.246021 | 0.174467 |
| dcas_full_ot_calibrated_target | 0.831564 | 2.029638 | 0.402333 | 0.187964 |
| dcas_full_ot_calibrated_minor | 0.828158 | 2.047741 | 0.530271 | 0.183386 |

reference method: `dcas_full_ot_calibrated_target`

关键点:

- 相比最强 raw/hybrid 基线, `dcas_full_ot_calibrated_target` 在 `serendipity` 上提升约 `+0.1982 ~ +0.3301`
- 相比 `dcas_full_ot`, `calibrated_target` 的 `serendipity` 下降 `-0.0263`, 但 `calibration_kl` 改善 `-0.0529`, `minority@k` 提升 `+0.1563`
- 相比 `calibrated_target`, `calibrated_minor` 在 `minority@k` 继续提升 `+0.1279`, 代价是 `calibration_kl` 略变差, `serendipity` 基本持平

## 4. 统计覆盖

### 4.1 routeA_small

- eval files: `11`
- comparison files: `20`
- users: `96`
- target cultures: `4`
- user-culture eval pairs: `384`

### 4.2 main

- eval files: `11`
- comparison files: `20`
- users: `240`
- target cultures: `10`
- user-culture eval pairs: `2400`

## 5. 论文可写边界

当前可以稳妥写入:

- 在 `V4 routeA_small` 和 `V4 main` 两条线上, `CultureMERT mw3 + stage3` 都能稳定支撑 `DCAS calibrated` 相对多种强弱基线的优势
- `calibrated_target` 与 `calibrated_minor` 呈现出稳定 trade-off:
  - `target` 更平衡
  - `minor` 更强调整 minority exposure
- 这种 trade-off 在小线和主线都能复现

当前不宜写得过满:

- `V4` 已经消除了 source bias
- `routeA_small` 与 `main` 具备同等强度的科学证据
- 任意 backbone 下都已经同样成立

最后一点仍然需要 `Gemini` 线补齐后再说。

## 6. 仍需正面留档的问题

1. 数据科学性问题仍在:
   - `V4 main`: `weighted_source_predictability_from_culture = 0.911765`
   - `V4 routeA_small`: `weighted_source_predictability_from_culture = 1.0`
2. `routeA_small` 的 `era` coverage 仍为 `0.0`
3. benchmark 运行期间出现大量 `LightGBM/sklearn feature names` warning
   - 当前命令返回码为 `0`
   - 结果文件完整生成
   - 更像是工程层面的特征名一致性问题, 不是直接的数值失败, 但建议后续清理

## 7. 下一步建议

1. 开始 `Gemini` 的 `V4 routeA_small -> V4 main` 对称线
2. 同步把 `EXPERIMENT_INDEX` 和论文草稿中的主实验矩阵更新到 `V4 stage3`
3. 若要进一步提升可辩护性, 优先处理 `V4 main` 的 source confound
