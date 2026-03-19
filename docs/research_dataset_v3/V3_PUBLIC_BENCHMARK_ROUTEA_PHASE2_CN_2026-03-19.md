# V3 公开多文化数据 Benchmark 测试记录（RouteA Phase2 CN，2026-03-19）

## 1. 这轮在回答什么

在完成 `LambdaMART / tree-based` 强排序基线之后，需要回答一个更关键的问题：

- 在公开数据构成的多文化数据集上，`DCAS` 和强基线的相对关系是否还能成立？

这里选择的是仓库内现成、可复现、且具备多文化域结构的公开数据聚合版：

- `storage/public/routeA_phase2_cn/tracks.npz`
- `storage/public/routeA_phase2_cn/interactions.csv`
- `storage/public/routeA_phase2_cn/model.pt`

说明：

- 这不是外部社区统一认领的“标准跨文化推荐 benchmark”。
- 但它是由公开来源音频构成的多文化数据集，能够作为当前项目里最可落地的 `public-data benchmark`。

## 2. 运行配置

配置文件：

- `configs/benchmark/recommender_benchmark_public_routeA_phase2_cn.run.json`

方法集合：

- `popularity`
- `cosine`
- `knn`
- `bpr_mf`
- `bpr_two_stage_hybrid`
- `bpr_listwise_hybrid`
- `bpr_lambdamart_hybrid`
- `dcas_full_ot`
- `dcas_full_ot_calibrated_target`
- `dcas_full_ot_calibrated_minor`

输出目录：

- `reports/benchmarks/public_routeA_phase2_cn_lambdamart`

## 3. 结果

### 3.1 主表

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| `popularity` | 0.5668 | 0.9961 | 0.0000 | 0.4798 |
| `cosine` | 0.6339 | 1.0880 | 0.2505 | 0.4543 |
| `knn` | 0.6684 | 1.1443 | 0.2342 | 0.4357 |
| `bpr_mf` | 0.6116 | 1.0096 | 0.1364 | 0.4744 |
| `bpr_two_stage_hybrid` | 0.6223 | 1.0712 | 0.2094 | 0.4548 |
| `bpr_listwise_hybrid` | 0.6371 | 1.0654 | 0.2186 | 0.4561 |
| `bpr_lambdamart_hybrid` | 0.6300 | 1.0017 | 0.2547 | 0.4788 |
| `dcas_full_ot` | 0.7653 | 1.2384 | 0.2661 | 0.4041 |
| `dcas_full_ot_calibrated_target` | 0.7429 | 1.0499 | 0.3741 | 0.4596 |
| `dcas_full_ot_calibrated_minor` | 0.7372 | 1.1020 | 0.4634 | 0.4443 |

### 3.2 结构性结论

这轮公开数据 benchmark 的结构和 `research_dataset_v3` 并不完全一样：

1. 原始 `dcas_full_ot` 仍然是最强的 `serendipity` 方法。
2. `dcas_full_ot_calibrated_target` 和 `dcas_full_ot_calibrated_minor` 继续显著提升了 `minority exposure`。
3. 但在这条公开数据线上，`LambdaMART` 与 `popularity/BPR` 在 `KL` 和 `target_prob` 上更有竞争力。

换句话说：

- `DCAS` 在公开多文化数据上仍然非常强于“惊喜度”和“长尾/少数项曝光”
- 但它不像在 `research_dataset_v3` 上那样形成“四项整体压制”

## 4. 与 LambdaMART 的对比

以当前最强树模型基线 `bpr_lambdamart_hybrid` 为参照：

### `dcas_full_ot_calibrated_target`

- `serendipity +17.92%`
- `KL -4.81%`，这里是更差，因为更高
- `minority exposure +46.87%`
- `target culture prob -4.01%`

### `dcas_full_ot_calibrated_minor`

- `serendipity +17.03%`
- `KL -10.02%`
- `minority exposure +81.96%`
- `target culture prob -7.21%`

### `dcas_full_ot`

- `serendipity +21.49%`
- `KL -23.63%`
- `minority exposure +4.48%`
- `target culture prob -15.60%`

## 5. 这说明什么

这轮结果其实很有价值，因为它说明了：

1. `DCAS` 的强项在公开多文化数据上依然稳定：
   - `serendipity`
   - `minority exposure`

2. `calibrated rerank` 仍然能把公开数据线上的 `minority` 拉高很多，但它并没有像在 `research_dataset_v3` 那样把 `KL / target_prob` 一起全面抬起来。

3. 因此更准确的论文表述应该是：

   - 在 `research_dataset_v3` 小规模受控设定中，`DCAS calibrated` 对已实现强基线保持整体优势
   - 在公开多文化数据 benchmark 上，`DCAS` 主要扩展了 `serendipity-minority` 前沿，而树模型基线在 `calibration / target affinity` 上仍有竞争力

## 6. 对论文的意义

这轮公开数据实验不会削弱论文，反而让叙事更真实：

- 你现在不需要写成“DCAS 在所有条件下四项都最强”
- 更好的写法是：
  - `DCAS` 在受控设定下展现整体优势
  - 在公开数据设定下，它最稳定的增益体现在 `serendipity` 与 `minority exposure`
  - 强树模型基线提醒我们，`calibration / target affinity` 仍是值得继续优化的方向

这种写法比“全面碾压”更可信，也更符合 reviewer 对真实 trade-off 的预期。

## 7. 补充说明

虽然这轮是 `public-data benchmark`，但仍然使用了仓库现有的交互文件，因此它不是“真实平台日志 benchmark”。

所以这轮更准确的定位是：

- `public multi-cultural audio dataset benchmark`
- 不是：
  - `industrial-scale real-user benchmark`
  - `community-standard cross-cultural recommendation benchmark`

## 8. 下一步建议

1. 若继续补公开 benchmark，优先级建议是：
   - `MSSD`
   - `Yambda-5B`

2. 若继续增强方法本身，方向应更聚焦于：
   - 提升公开数据线上的 `KL`
   - 提升 `target culture affinity`
   - 而不是只继续拉高 `serendipity`
