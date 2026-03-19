# V3 LightFM 风格基线测试记录
更新日期：2026-03-19

## 1. 目标

在 `BPR-MF` 已经证明自己是一个更像样的经典协同过滤对手后，继续补一条更接近主流平台混合推荐的代理基线：

- 保留 `CF` 主干
- 注入音频 embedding 内容特征
- 注入 culture/source 侧信息
- 允许在推理阶段做轻量 rerank

这条线的定位不是“精确复刻官方 LightFM”，而是构建一个 `LightFM-style content-enhanced CF proxy`，观察它相对 `BPR` 和 `DCAS stage3` 的表现。

## 2. 实现方式

代码位置：

- `E:/Desktop/Echo/dcas/embedding_recommenders.py`
- `E:/Desktop/Echo/dcas/scripts/run_recommender_benchmarks.py`

配置：

- `E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v3_culturemert_stage3_lightfmlike.run.json`

实现分两版：

1. 第一版 `lightfm_like_hybrid`
- `BPR` pairwise loss
- user id latent
- item id latent
- item content projection
- culture/source latent
- user history profile projection

2. 第二版 `lightfm_like_hybrid + rerank`
- 保留第一版训练得到的 checkpoint
- 推理时再融合
  - `MF score`
  - `content cosine`
  - `novelty`
  - `minority`
  - `source inverse frequency`

## 3. 结果

checkpoint：

- `E:/Desktop/Echo/storage/models/lightfm_like_v3_main_culturemert_stage3.pt`

最新原始评测文件：

- `E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert_stage3_lightfmlike/eval/lightfm_like_hybrid.json`

注意：

- 完整 benchmark 汇总在 20 分钟超时前没有稳定写完，因此 `benchmark_summary.json` 可能停留在第一版数值。
- 本文结论以最新 `eval/lightfm_like_hybrid.json` 中的 `summary` 为准。

第二版最终指标：

- `serendipity = 0.4773`
- `KL = 2.0889`
- `minority = 0.1583`
- `target_prob = 0.1805`

第一版相对第二版变化：

- `serendipity +1.47%`
- `KL -0.21%`
- `minority +20.61%`
- `target_prob -0.42%`

这里 `KL -0.21%` 表示第二版更差，因为 `KL` 更低才更好。

## 4. 与 BPR / DCAS 对比

相对 `BPR-MF`：

- `serendipity -2.90%`
- `KL -3.28%`
- `minority +6.10%`
- `target_prob -7.78%`

相对 `DCAS stage3 (dcas_full_ot)`：

- `serendipity -43.52%`
- `KL -2.20%`
- `minority -34.00%`
- `target_prob -6.41%`

换句话说：

- 它唯一明显占优的是相对 `BPR` 的 `minority exposure`
- 但在 `serendipity / KL / target_prob` 三项都没有赢过 `BPR`
- 更没有接近 `DCAS stage3`

## 5. 判断

这条 `LightFM-style` 路线目前不值得作为主线继续深挖，原因不是“完全没用”，而是：

- 单纯把内容特征接到 `MF/BPR` 上，并不会自动形成强平台式混合推荐
- 在当前数据协议下，它没有超过 `BPR`
- 推理端轻量 rerank 虽然把 `minority exposure` 拉上来一些，但提升幅度仍不足以改变整体结论

当前更有价值的下一步应当是：

1. 继续补更强的学习排序型 hybrid
- 例如 `GBDT / LambdaMART` 风格 reranker 代理

2. 继续强化 `DCAS` 自己的排序目标
- 让 `DCAS` 从“好表示”进一步逼近“强 ranker”

3. 如果继续补基线
- 优先做 `BPR + learned reranker` 或 `two-stage candidate generation + ranker`
- 不建议继续在这条 `LightFM-like` 线上投入过多时间
