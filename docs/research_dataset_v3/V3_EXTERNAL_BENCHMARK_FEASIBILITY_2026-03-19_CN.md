# V3 外部公开 Benchmark 可行性说明（2026-03-19）

这份说明回答两个问题：

1. 现在能不能把当前仓库里的 cross-cultural benchmark 直接跑到 `Yambda-5B` 和 `MSSD`
2. 如果不能，卡点到底是“数据拿不到”、还是“代码结构不兼容”

## 一句话结论

- `MSSD` 这轮不能直接跑，因为官方公开下载入口已经关闭。
- `Yambda-5B` 虽然公开可得，但不能直接复用当前仓库的 benchmark runner。
- 当前项目如果以 ISMIR/方法论文为主线，优先级仍然应该是：`真人 PAL -> 回灌 -> 改稿`。
- 如果后续要补“真实日志 benchmark”，建议单独开一条通用 recsys 评测线，而不是硬把它塞进现在这套 cross-cultural 四指标框架。

## 1. MSSD 当前状态

官方来源：

- Spotify Research short paper:
  - https://research.atspotify.com/publications/the-music-streaming-sessions-dataset-short-paper/
- AIcrowd challenge page:
  - https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge

关键事实：

- AIcrowd 页面明确写了 `Update 8th July 2024`：
  - 这个 challenge 对应的数据集已经不再提供公开下载
  - 需要直接联系 `Spotify Research` 申请访问

因此这轮不能在本地“立刻下载 + 立刻跑 benchmark”。

## 2. Yambda 当前状态

官方来源：

- Hugging Face dataset card:
  - https://huggingface.co/datasets/yandex/yambda
- Paper:
  - https://arxiv.org/abs/2505.22238

当前确认到的事实：

- `Yambda` 是公开数据集，可以直接通过 `datasets` / `huggingface_hub` 访问
- 数据卡给出了三档规模：
  - `50M`
  - `500M`
  - `5B`
- 本地探针结果：
  - `sequential/50m/multi_event.parquet` 约 `437,501,893` bytes
  - `sequential/500m/multi_event.parquet` 约 `4,388,605,489` bytes
  - `sequential/5b/multi_event.parquet` 约 `43,682,305,922` bytes
  - `embeddings.parquet` 约 `13,814,230,943` bytes

所以 `Yambda` 不是“拿不到”，而是“拿得到，但不能直接套当前代码”。

## 3. 为什么当前仓库不能直接跑 Yambda / MSSD

### 3.1 当前 benchmark 输入假设不匹配

当前仓库的主 benchmark runner 是：

- `dcas/scripts/run_recommender_benchmarks.py`

它默认消费的是：

- `tracks.npz`
  - 至少包含 `track_id`, `culture`, `embedding`
- `interactions.csv`
  - 至少包含 `user_id`, `track_id`, `weight`

当前评测指标又依赖：

- `target_culture`
- track-level `culture`
- `cultural_calibration_kl`
- `target_culture_prob_mean`

但：

- `Yambda` 的公开 schema 是 `uid/item_id/timestamp/is_organic/event_type/...`
- `MSSD` 的公开任务是 session-based skip prediction
- 这两者都不是当前仓库要求的“有文化标签的跨文化推荐输入”

也就是说，问题不只是“格式转换”，而是评测问题定义本身不同。

### 3.2 当前 BPR 实现也不具备工业规模可扩展性

当前 BPR 训练实现位于：

- `dcas/embedding_recommenders.py`

其中 `_build_bpr_training_state(...)` 会为每个用户显式构造“全集物品的补集负样本池”。

在当前小规模数据上可以工作，但到了：

- `Yambda-50M`: 约 `934k` items
- `Yambda-5B`: 约 `9.39M` items

这套做法在内存和时间上都不合适。

因此，即使先不谈文化标签，当前训练代码也不能把 `Yambda-5B` 当成“今天就能跑通”的数据集。

## 4. 现在最合理的工程判断

如果目标是：

- 完成当前跨文化音乐推荐论文主线
- 把现有证据链补硬
- 尽快进入真人 PAL

那么优先级应该是：

1. 整理并冻结当前 `V3 + LambdaMART + public routeA` 结果
2. 做真人 PAL
3. 跑一轮 PAL 回灌前后对比
4. 再做最后一轮 claim 和图表收敛

如果目标是：

- 额外补一个“真实平台日志 benchmark”

那建议另开一条分支，补一个新的通用评测器，核心变化至少包括：

1. 不再依赖 `culture` / `target_culture`
2. 指标切到 `Recall@K / NDCG@K / MRR / HitRate`
3. 训练端改成可扩展负采样，而不是显式补集负池
4. 先从 `Yambda-50M` 开始，而不是直接冲 `5B`
5. `MSSD` 则先完成访问申请，再决定是否接入

## 5. 对项目节奏的影响

以当前主线来看，项目已经接近“只差真人 PAL 和最后一轮改稿”。

更准确地说：

- 对 cross-cultural 论文主线：是的，离收尾已经不远了
- 对“再补一个真实日志 benchmark”这条附加线：还差一套新的评测协议，不是小修小补

## 6. 本轮产物

为了避免这部分只停留在口头判断，本轮还补了一个本地探针脚本：

- `dcas/scripts/probe_external_benchmarks.py`

默认会输出：

- `reports/external_benchmarks/public_benchmark_probe_2026-03-19.json`

这个 JSON 会记录：

- `Yambda` 的公开可访问性和关键文件大小
- `MSSD` 的当前访问状态
- 当前仓库对外部工业规模日志 benchmark 的结构性限制
