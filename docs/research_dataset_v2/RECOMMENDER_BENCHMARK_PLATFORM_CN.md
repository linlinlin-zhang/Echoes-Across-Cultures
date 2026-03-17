# Recommender Benchmark Platform

这套平台把三类路线放进同一个 benchmark 入口：

- raw embedding baselines
  - `popularity`
  - `cosine`
  - `knn`
  - `shallow_mlp`
  - `hybrid_content_popularity_diversity`
- DCAS checkpoints
  - `dcas_full_ot`
  - `dcas_full_knn`
  - 也可以继续加 `no_ot / no_domain / no_constraints` 等消融检查点

## 设计目标

我们现在需要比较的不只是一个模型，而是一整条从弱到强的推荐梯度：

1. 纯流行度
2. raw embedding 余弦检索
3. raw embedding kNN
4. raw embedding + shallow MLP 排序头
5. 主流风格的混合推荐器
6. DCAS latent-space recommenders

这套平台的输入是统一的：

- `tracks.npz`
- `interactions.csv`

所以不管 backbone 是 `CultureMERT` 还是 `Gemini Embedding 2`，都可以复用同一套 benchmark runner。

## 当前实现

新增模块：

- `dcas/embedding_recommenders.py`
- `dcas/scripts/run_recommender_benchmarks.py`

新增配置模板：

- `configs/benchmark/recommender_benchmark_culturemert.example.json`
- `configs/benchmark/recommender_benchmark_gemini.example.json`

## Hybrid baseline 的含义

这里的 hybrid baseline 不是声称复现某一家工业系统，而是实现一类主流音乐推荐里很常见的 re-ranking 结构：

- content similarity
- neighborhood affinity
- popularity prior
- diversity / novelty encouragement

默认权重：

- cosine: `0.40`
- knn: `0.25`
- popularity: `0.20`
- novelty: `0.15`

## 统一运行方式

CultureMERT 例子：

```powershell
python E:\Desktop\Echo\dcas\scripts\run_recommender_benchmarks.py `
  --config E:\Desktop\Echo\configs\benchmark\recommender_benchmark_culturemert.example.json
```

Gemini 例子：

```powershell
python E:\Desktop\Echo\dcas\scripts\run_recommender_benchmarks.py `
  --config E:\Desktop\Echo\configs\benchmark\recommender_benchmark_gemini.example.json
```

## 产物

每个 suite 会输出：

- `eval/<method>.json`
- `comparisons/<baseline>_vs_<reference>.json`
- `comparisons/<baseline>_vs_<reference>.md`
- `benchmark_summary.json`
- `benchmark_table.md`

## 注意事项

- 如果 `interactions.csv` 不存在，runner 可以按配置自动调用 `synthesize_interactions.py`
- 当前 unified evaluator 使用 `tracks.embedding` 所在空间做 proxy metrics
- 这意味着：
  - 同一 backbone 内的方法比较最可信
  - 跨 backbone 的绝对数值比较需要更谨慎解释
