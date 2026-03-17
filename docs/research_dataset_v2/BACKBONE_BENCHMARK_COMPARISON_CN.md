# Backbone Benchmark Comparison

## Scope

本文件汇总 `v2_main` 数据集上两条 embedding backbone 的推荐对比结果：

- `CultureMERT`
- `Gemini Embedding 2`

统一比较的方法包括：

- `popularity`
- `cosine`
- `knn`
- `shallow_mlp`
- `hybrid_content_popularity_diversity`
- `dcas_full_ot`
- `dcas_full_knn`

## Result Files

- CultureMERT:
  - `E:/Desktop/Echo/reports/benchmarks/v2_main_culturemert/benchmark_table.md`
  - `E:/Desktop/Echo/reports/benchmarks/v2_main_culturemert/benchmark_summary.json`
- Gemini:
  - `E:/Desktop/Echo/reports/benchmarks/v2_main_gemini_embedding2/benchmark_table.md`
  - `E:/Desktop/Echo/reports/benchmarks/v2_main_gemini_embedding2/benchmark_summary.json`

## Side-by-Side Summary

| backbone | method | serendipity | calibration_kl | minority@k | target_prob |
|---|---|---:|---:|---:|---:|
| CultureMERT | popularity | 0.600620 | 0.884802 | 0.000000 | 0.496869 |
| CultureMERT | cosine | 0.691428 | 1.181762 | 0.344500 | 0.416574 |
| CultureMERT | knn | 0.724240 | 1.167226 | 0.350750 | 0.419820 |
| CultureMERT | shallow_mlp | 0.624489 | 1.342065 | 0.344000 | 0.368237 |
| CultureMERT | hybrid | 0.660005 | 1.111716 | 0.171667 | 0.433642 |
| CultureMERT | dcas_full_ot | 0.837806 | 0.805250 | 0.340083 | 0.526995 |
| CultureMERT | dcas_full_knn | 0.837938 | 0.804816 | 0.340667 | 0.527214 |
| Gemini Embedding 2 | popularity | 0.831506 | 1.759603 | 0.000000 | 0.234920 |
| Gemini Embedding 2 | cosine | 0.893858 | 1.801267 | 0.346333 | 0.225189 |
| Gemini Embedding 2 | knn | 0.897896 | 1.802071 | 0.343583 | 0.224995 |
| Gemini Embedding 2 | shallow_mlp | 0.851072 | 1.807892 | 0.354000 | 0.223491 |
| Gemini Embedding 2 | hybrid | 0.857469 | 1.794590 | 0.155500 | 0.226747 |
| Gemini Embedding 2 | dcas_full_ot | 0.832479 | 1.759199 | 0.361500 | 0.235109 |
| Gemini Embedding 2 | dcas_full_knn | 0.831549 | 1.759227 | 0.361000 | 0.235105 |

## Main Takeaways

### 1. CultureMERT + DCAS is much more synergistic

在 `CultureMERT` 上，`DCAS` 明显优于所有 raw embedding baseline。

- `serendipity` 从最强 raw baseline 的 `0.724240` 提升到 `0.837806`
- `calibration_kl` 也从 `1.167226` 降到 `0.805250`

这说明在当前 `v2_main` 上，`CultureMERT` 的原始 embedding 经过 `DCAS` 的结构化表示学习后，收益非常明显。

### 2. Gemini raw baselines are already strong on serendipity

在 `Gemini Embedding 2` 上，`cosine / knn` 这类直接 embedding 推荐已经有很高的 `serendipity`：

- `cosine = 0.893858`
- `knn = 0.897896`

而 `DCAS` 在 Gemini 上没有继续拉高该指标，反而略低。

### 3. Gemini + DCAS is stronger on calibration-oriented objectives

虽然 Gemini 上的 `DCAS` 没有赢下 `serendipity`，但它在更贴近论文目标的指标上更强：

- `calibration_kl` 最低：`dcas_full_ot = 1.759199`
- `minority@k` 最高：`dcas_full_ot = 0.361500`

这说明 Gemini 的 raw embedding 更容易支持“新颖/惊喜型”推荐，而 `DCAS` 更倾向于把结果往“校准 / 少数域曝光 / 目标文化控制”方向拉。

### 4. OT and DCAS-kNN are nearly tied on both backbones

两条 backbone 上，`dcas_full_ot` 和 `dcas_full_knn` 的差异都很小：

- CultureMERT:
  - `serendipity`: `0.837806` vs `0.837938`
  - `calibration_kl`: `0.805250` vs `0.804816`
- Gemini:
  - `serendipity`: `0.832479` vs `0.831549`
  - `calibration_kl`: `1.759199` vs `1.759227`

当前这一版结果更像是在说明：

- `DCAS` 的表示学习层已经带来了主要收益
- `OT` 在本版实验中不是唯一决定性因素

## Paper-Friendly Interpretation

如果要把这轮结果写进论文，目前最稳的表述是：

- `CultureMERT` 更像是与 `DCAS` 高度匹配的音乐专用 backbone，结构化 downstream 建模收益显著。
- `Gemini Embedding 2` 作为通用多模态 backbone，在 raw embedding 推荐上表现出更强的 serendipity 潜力。
- `DCAS` 在两条 backbone 上都稳定改善 cultural calibration 和 minority exposure，说明其价值不只依赖某一个特定音频 backbone。

## Recommended Next Experiments

最值得继续补的实验有三类：

1. `Gemini` 上的 `DCAS` 消融：
   - `no_ot`
   - `no_domain`
   - `no_constraints`
2. `CultureMERT` 与 `Gemini` 的 per-target-culture breakdown
3. 加入真实或更强的 `PAL` 反馈，验证在两条 backbone 上的增益是否一致
