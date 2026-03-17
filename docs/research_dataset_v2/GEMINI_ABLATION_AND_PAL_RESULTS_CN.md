# Gemini Ablation and PAL Results

## Files

- Ablation table:
  - `E:/Desktop/Echo/reports/ablation/v2_main_gemini/ablation_table_draft.md`
- Ablation summary:
  - `E:/Desktop/Echo/reports/ablation/v2_main_gemini/ablation_summary.json`
- Simulated PAL summary:
  - `E:/Desktop/Echo/reports/pal/v2_main_gemini_simulated/phase3_pal_summary.md`
- Simulated PAL constraints:
  - `E:/Desktop/Echo/storage/pal/v2_main_gemini_simulated/constraints_upto_round2.jsonl`

## Gemini DCAS Ablation

| setting | serendipity | calibration_kl | minority@k |
|---|---:|---:|---:|
| full | 0.822481 | 1.906046 | 0.362917 |
| no_domain | 0.831156 | 1.905758 | 0.347167 |
| no_constraints | 0.841324 | 1.906040 | 0.338583 |
| no_ot | 0.823858 | 1.906046 | 0.361167 |

## Ablation Reading

### no_domain

- `serendipity` 变高：`+0.008675`
- `minority@k` 下降：`-0.015750`

解释：

- 去掉 domain adversarial 之后，模型更容易利用文化风格差异
- 这会让推荐看起来更“新鲜”
- 但也更容易牺牲跨文化控制和少数域曝光稳定性

### no_constraints

- `serendipity` 提升最大：`+0.018843`
- `minority@k` 下降最大：`-0.024333`

解释：

- 约束会把模型往更稳定、更一致的跨文化边界上拉
- 去掉它后，模型更自由，也更容易追新颖性
- 但代价是少数文化内容的曝光控制变弱

### no_ot

- `serendipity` 仅微增：`+0.001377`
- 差异不显著

解释：

- 在当前 Gemini backbone 上，OT 不是决定性增益来源
- 当前版本的主要收益更像来自 DCAS 表示学习本身

## Simulated PAL

| run | serendipity | calibration_kl | minority@k |
|---|---:|---:|---:|
| baseline | 0.832479 | 1.906160 | 0.361500 |
| round1 | 0.812584 | 1.905917 | 0.370667 |
| round2 | 0.830208 | 1.905735 | 0.370500 |

## PAL Reading

### Round 1

- `serendipity` 明显下降
- `calibration_kl` 略微改善
- `minority@k` 略微上升

解释：

- 第一轮模拟约束更像在“收紧空间”
- 它先把模型拉得更保守
- 因而新颖性下降，但校准和曝光更稳

### Round 2

- `serendipity` 基本回到 baseline 附近
- `calibration_kl` 继续小幅改善
- `minority@k` 仍高于 baseline

解释：

- 两轮后，模型开始吸收约束而不是只被约束压缩
- 这说明 PAL 机制本身是通的
- 但当前模拟反馈信号还不够强，不足以带来特别大的最终增益

## Current Interpretation

这轮 Gemini 结果说明：

1. `DCAS` 的真正价值在于 calibration / exposure / 结构化控制
2. 纯追求 `serendipity` 时，去掉部分约束反而会更高
3. `PAL` 机制已经跑通，但当前 simulated feedback 更像“稳空间”，还没有展现出真实专家反馈应有的增益

## What This Means for the Paper

最稳的写法是：

- `DCAS` 不是单纯追求最大新颖性的系统
- 它更擅长把推荐往跨文化目标函数上拉：
  - 更稳定的 cultural calibration
  - 更高的 minority exposure
  - 更受控的 target-culture recommendation
- 在 Gemini 上，约束和 domain invariance 形成了明确 trade-off
- 这也正好说明为什么真实 PAL 值得做：它可能比当前 simulated constraints 更有信息量，能减少这种“只收紧、不增益”的问题
