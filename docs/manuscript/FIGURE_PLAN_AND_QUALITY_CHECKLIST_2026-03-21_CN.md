# Figure Plan And Quality Checklist (2026-03-21)

## 1. 主图计划

### Figure 1. System overview

- 内容：
  - `V4 dataset contract`
  - `factorized DCAS`
  - `calibrated reranking`
  - `PAL loop`
- 作用：
  - 支撑全文主叙事
- caption 模板：
  - `(a)` 总体流程
  - `(b)` 哪一步对接 backbone
  - `(c)` 哪一步对接 PAL

### Figure 2. Calibration sweep curve

- 横轴：
  - `minority_weight`
- 纵轴：
  - `serendipity`
  - `minority_exposure_at_k`
  - 可选第二图放 `cultural_calibration_kl`
- 最推荐数据：
  - `V4 main + CultureMERT`
  - `V4 routeA_small + Gemini`
- 作用：
  - 支撑“可控 trade-off”这一核心 claim

### Figure 3. Cross-backbone main result summary

- 形式：
  - 分组柱状图或 dot plot
- 组别：
  - `V4 main / routeA_small`
  - `CultureMERT / Gemini`
- 指标：
  - `serendipity`
  - `cultural_calibration_kl`
  - `minority_exposure_at_k`

### Figure 4. PAL workflow / annotation packet example

- 内容：
  - uncertainty scoring
  - task export
  - annotator packets
  - constraint rebuild
- 作用：
  - 把 PAL 从“概念”变成“可执行流程”

## 2. Caption 写法规范

每张图 caption 建议严格按三句写：

1. 先说总体趋势
2. 再说 1 到 2 个关键数字
3. 最后说这张图支撑了哪条方法论主张

示例句式：

> Increasing the minority-oriented reranking weight produces a smooth rise in minority exposure on V4 main. Moving from the uncalibrated OT ranker to the target-calibrated point increases minority exposure from 0.246 to 0.402 while reducing serendipity from 0.858 to 0.832. This supports the claim that calibration acts as a controllable trade-off layer rather than an ad hoc post-processing trick.

## 3. 图表风格统一

- 字体：
  - 与论文正文一致，优先 Times 系或会议模板默认 serif
- 颜色：
  - backbone 用一套色系
  - operating points 用另一套色系
  - 不要在同一图里同时用过多亮色
- 坐标轴：
  - 指标名用论文最终术语
  - `KL` 类指标明确标注 `lower is better`
- 图例：
  - `P2 target`
  - `P4 minor`
  - `OT uncalibrated`

## 4. Phase 4 术语与论证检查

### 术语统一

- `CultureMERT` 和 `Gemini` 称为 `backbones`
- `DCAS` 称为 `framework` 或 `downstream stack`
- `calibrated_target` 和 `calibrated_minor` 统一称 `operating points`
- `routeA_small` 统一称 `sanity-check track`

### Claim-Evidence 对照

- 每一条 claim 必须在 [CLAIM_EVIDENCE_MAP_2026-03-21.csv](E:/Desktop/Echo/docs/paper/CLAIM_EVIDENCE_MAP_2026-03-21.csv) 中找到证据
- 找不到证据的句子不要写进主文

### 工程语言转学术语言

- “调参试出来的” 改成 `selected through controlled sensitivity analysis`
- “这个点效果最好” 改成 `chosen as the default operating point due to a favorable trade-off`
- “先跑通再说” 改成 `staged validation under a unified experimental protocol`

## 5. 技术一致性检查清单

- Method 中的损失项命名要与 [dcas/models/dcas_vae.py](E:/Desktop/Echo/dcas/models/dcas_vae.py) 和 [dcas/pipelines.py](E:/Desktop/Echo/dcas/pipelines.py) 一致
- `target_affinity_weight / minority_weight / source_weight / diversity_lambda` 的表述要与 benchmark config 一致
- 指标名要与 [run_recommender_benchmarks.py](E:/Desktop/Echo/dcas/scripts/run_recommender_benchmarks.py) 输出一致
- `source confound` 的定性结论要与 [README.md](E:/Desktop/Echo/docs/research_dataset_v4/README.md) 保持一致
- PAL 状态统一写成 `PAL-ready` 或 `execution-ready`, 不要写成已完成真人实验

## 6. 建议下一步

- 补一张 `V4 main + Gemini` 的 calibration sweep 图，作为 cross-backbone 完整镜像
- 从现有 benchmark summary 自动导出一张主实验总表
- 开始把 [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex) 从旧四域 placeholder 叙事重写为 V4 双 backbone 叙事

