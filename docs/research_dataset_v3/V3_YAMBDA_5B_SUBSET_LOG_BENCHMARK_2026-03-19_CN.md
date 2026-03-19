# Yambda-5B 子集公开日志 Benchmark 记录（2026-03-19）

这轮补的是一条和当前 cross-cultural 四指标不同的公开日志评测线：

- 数据源：`Yandex / Yambda`
- 来源配置：`flat-multievent-5b`
- 评测目标：`Recall@K / NDCG@K / MRR@K`
- 目的：给论文补一条“真实日志来源、标准排序指标”的外部对照线

## 1. 为什么这轮要单独开新 runner

当前仓库原有 benchmark runner：

- `dcas/scripts/run_recommender_benchmarks.py`

它依赖：

- track-level `culture`
- per-user `target_culture`
- `serendipity / calibration / minority exposure / target_prob`

但 `Yambda-5B` 是工业级公开日志数据，原始 schema 是：

- `uid`
- `item_id`
- `timestamp`
- `is_organic`
- `event_type`
- `played_ratio_pct`

所以不能直接把它硬塞进当前 cross-cultural 评测定义里。这轮新增的是：

- `dcas/scripts/build_yambda_subset.py`
- `dcas/scripts/run_log_recommender_benchmarks.py`

前者把 `Yambda-5B` 官方数据转成仓库可消费的 `tracks/interactions` 子集，
后者在这个子集上跑标准 log-ranking 指标。

## 2. 这次实际跑的 Yambda 子集

子集产物：

- `storage/public/yambda_5b_subset/tracks.npz`
- `storage/public/yambda_5b_subset/metadata.csv`
- `storage/public/yambda_5b_subset/interactions.csv`
- `storage/public/yambda_5b_subset/subset_report.json`

采样协议：

- 交互源：`flat-multievent-5b`
- embedding 源：`embeddings.parquet` 的 `row_group=0`
- embedding 列：`normalized_embed`
- 只保留 `event_type=listen`
- `played_ratio_pct >= 50`
- 扫描前 `500000` 条事件
- 只保留交互数不少于 `30` 的用户

这次实际拿到：

- `58` 个用户
- `1352` 首歌
- `8393` 条交互

这里必须强调：

- 这是 **Yambda-5B 官方公开数据的可复现实验子集**
- 不是完整 `5B` 全量跑
- 也不是 cross-cultural benchmark

## 3. 配置与输出

配置：

- `configs/benchmark/log_benchmark_yambda_5b_subset.run.json`

结果目录：

- `reports/benchmarks/yambda_5b_subset_global_log_benchmark`

关键文件：

- `benchmark_summary.json`
- `benchmark_table.md`
- `eval/*.json`
- `comparisons/*`

## 4. 结果总表

| method | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | MRR@10 | MRR@20 |
|---|---:|---:|---:|---:|---:|---:|
| `popularity` | `0.0862` | `0.1552` | `0.0585` | `0.0770` | `0.0499` | `0.0555` |
| `cosine` | `0.1379` | `0.2069` | `0.0773` | `0.0939` | `0.0579` | `0.0621` |
| `knn` | `0.1897` | `0.2931` | `0.1148` | `0.1397` | `0.0904` | `0.0966` |
| `bpr_mf` | `0.3966` | `0.5172` | `0.1861` | `0.2178` | `0.1246` | `0.1339` |
| `bpr_two_stage_hybrid` | `0.4483` | `0.5345` | `0.2100` | `0.2323` | `0.1389` | `0.1453` |
| `bpr_lambdamart_hybrid` | `0.4655` | `0.5517` | `0.2257` | `0.2478` | `0.1534` | `0.1597` |
| `dcas_log_ot` | `0.0345` | `0.0862` | `0.0230` | `0.0367` | `0.0197` | `0.0238` |

## 5. 这轮最硬的结论

### 5.1 强排序基线在公开日志子集上明显更强

这轮的最强方法是：

- `bpr_lambdamart_hybrid`

它在 `Recall@20 / NDCG@20 / MRR@20` 上都排第一。

和它最接近的是：

- `bpr_two_stage_hybrid`

但差距仍然存在。

### 5.2 这条结果不能被写成“DCAS 在真实日志上也最强”

这轮 `dcas_log_ot` 的表现明显落后于协同过滤和树模型：

- `Recall@20 = 0.0862`
- `bpr_lambdamart_hybrid` 的 `Recall@20 = 0.5517`

因此如果论文要补这条外部证据，更稳的写法是：

- `DCAS` 的当前主优势仍然在 cross-cultural 目标与受控多目标优化
- 在公开大规模日志排序任务上，协同过滤 / BPR / LambdaMART 仍然更强

这个结论虽然没有“全面胜利”那么好听，但实际上更可信。

## 6. 这对论文叙事意味着什么

现在证据链可以拆成两块：

### A. 你自己的主战场

- `V3`
- `public routeA`
- `PAL`
- cross-cultural 四指标

这里是 `DCAS` 的主叙事空间。

### B. 外部公开日志补充线

- `Yambda-5B subset`
- 标准排序指标

这里更像是在说明：

- 你没有回避真实日志排序 benchmark
- 但 `DCAS` 当前版本并不是为这类工业排序目标专门优化的

对 reviewer 来说，这比“假装四处都赢”更容易信。

## 7. 为什么 MSSD 这轮仍然没法跑

`MSSD` 的问题不是代码，而是访问。

官方来源：

- AIcrowd challenge page:
  - https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge
- Spotify Research short paper:
  - https://research.atspotify.com/publications/the-music-streaming-sessions-dataset-short-paper/

当前公开状态是：

- AIcrowd 页面明确写明，自 `2024-07-08` 起数据不再公开下载
- 需要联系 `Spotify Research` 申请访问

所以这轮：

- `Yambda` 已经实际跑出一条可复现子集 benchmark
- `MSSD` 仍然是外部数据访问 blocker，不是本地脚本没写

## 8. 和主线进度的关系

以当前项目主线来看，现在更合理的判断是：

1. `LambdaMART` baseline 已补
2. 公开多文化线已补
3. `Yambda-5B` 官方日志子集 benchmark 已补
4. `MSSD` 被官方访问限制卡住
5. 剩下最值钱的主线动作仍然是：
   - 真人 `PAL`
   - 回灌
   - 论文最后一轮收敛

也就是说，项目离“可收尾”已经很近，但还没有近到可以跳过真人 `PAL`。
