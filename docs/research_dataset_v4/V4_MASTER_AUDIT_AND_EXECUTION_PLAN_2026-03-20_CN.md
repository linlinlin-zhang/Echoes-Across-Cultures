# V4 主审计与执行计划

日期：2026-03-20

## 1. 目的

本文件用于把当前仓库从“多轮原型实验堆叠”收敛到“可复现、可审计、可写论文”的统一主线。

目标分为四类：

- 自检当前仓库的科学性、可复现性和留档完整性。
- 定义 `Research Dataset V4` 与小型公开对照集的新数据契约。
- 对 `CultureMERT` 线做一次完整重跑前的工程和协议修整。
- 为 `Gemini` 线补齐与 `CultureMERT` 对称的实验链路。

## 2. 当前仓库现状

主数据集与主线实验：

- 主数据集：`storage/public/research_dataset_v3/`
- 主线 benchmark：`configs/benchmark/recommender_benchmark_v3_culturemert_stage3_lambdamart.run.json`
- 主线模型：`storage/models/dcas_full_v3_main_culturemert_stage3.pt`

公开来源对照线：

- 小数据线：`storage/public/routeA_phase2_cn/`
- 对照 benchmark：`configs/benchmark/recommender_benchmark_public_routeA_phase2_cn.run.json`

Gemini 现状：

- 已有 pre-PAL 主线：`configs/benchmark/recommender_benchmark_v3_gemini.run.json`
- 已有 `mw3` embedding 产物：`storage/public/research_dataset_v3/tracks_gemini_embedding2_v3_main_mw3.npz`
- 缺口：没有与 `CultureMERT stage3` 对称的 `Gemini stage3` benchmark 配置和完整留档

论文现状：

- 草稿：`paper/ismir2026_draft.tex`
- 当前草稿仍有 placeholder 表述和旧数据规模，不可直接作为提交版

## 3. 当前最重要的问题

### 3.1 论文证据与仓库真实状态不一致

- `paper/ismir2026_draft.tex` 仍写有 placeholder 结果、旧的四域数据规模和过时的 baseline 描述。
- 论文主张如果不降到当前证据边界，会出现 claim-evidence mismatch。

### 3.2 数据链路不统一

`V3` 与 `routeA_phase2_cn` 当前不是同一套处理协议：

- `V3` 使用 `build_research_dataset_v3.py + merge_metadata_dedup.py + harmonize_v3_metadata.py`
- `routeA_phase2_cn` 仍主要使用 `merge_metadata.py`
- `V3` 当前主线用 `coarse_label + mixed_culture interactions + mw3`
- `routeA_phase2_cn` 仍保留更早期的简单 schema 和单阶段产物

### 3.3 embedding 构建协议不统一

- `V3 CultureMERT` 主线是 `30s + 3 windows + mean aggregate`
- `routeA_phase2_cn` 当前 `CultureMERT` manifest 还是 `max_seconds = 6.0`
- `Gemini` 与 `CultureMERT` 的多窗口产物存在，但 downstream benchmark 矩阵不对称

### 3.4 数据质量与数据治理仍有硬伤

- `V3 CultureMERT mw3` 构建丢了 `16` 条 track，并进一步丢了 `140` 条 interactions
- `V3` 仍存在明显 source confound，多个文化域几乎绑定单一来源
- `label` 与 `coarse_label` 的角色已经分化，但只在 `V3` 做了较系统 harmonize
- 小数据线和主数据线的 metadata 完整度、时长分布、来源条件差异仍然过大

### 3.5 交互协议仍不够严谨

- 合成交互依然是主证据来源之一，论文里必须明确写成 synthetic / weakly supervised interactions
- 交互合成策略在不同数据线之间并不完全一致
- 当前 benchmark 尚未把“真实日志、弱监督日志、PAL 反馈”当作三个不同证据层明确分层

### 3.6 评测矩阵不对称

- `CultureMERT` 线已经推进到 `stage3 + BPR + LambdaMART + calibrated DCAS`
- `Gemini` 线还停留在较早的 pre-PAL / harmonized open 轨道
- 这样无法支持“embedding backbone 换掉以后，系统依然稳定成立”的更强主张

### 3.7 缺少统一索引页

- 当前结果、runbook、技术说明、图表、benchmark 分散在 `docs/`、`reports/`、`storage/` 和 `paper/`
- 已有很多材料，但还没有一个“论文级版本索引页”把数据版本、模型版本、指标表和图表链接起来

## 4. 第一批必须修复的事项

这些问题会直接影响论文可信度，应优先于继续加模型：

- 冻结一版“当前主线真实状态”的索引与审计报告
- 建立 `V4` 统一 schema、目录约定和 manifest
- 统一 `V4 main` 与 `V4 routeA_small` 的清洗、验证和交互协议
- 修复 `CultureMERT mw3` 构建失败与丢样本问题
- 为 `Gemini` 建立与 `CultureMERT stage3` 对称的 benchmark 配置
- 把论文草稿里的 placeholder 和旧数据规模全部替换为真实版本边界

## 5. 分批执行方案

### Batch A：仓库自检与协议冻结

交付物：

- 本文档
- `V4` 数据契约文档
- `V4 manifest` 示例
- 一份后续 run 统一索引页

完成标准：

- 所有后续脚本都有统一输入输出约定
- 论文叙事边界不再漂移

### Batch B：`V4` 数据集重构

主目标：

- 建 `research_dataset_v4/main`
- 建 `research_dataset_v4/routeA_small`
- 用统一 schema 重做 raw -> clean -> harmonized -> release

完成标准：

- 主集和小集都能产出统一的 `metadata_*`、`tracks_*`、`interactions_*`、`validation_*`
- 有完整数据卡和质量量化结果

### Batch C：`CultureMERT` 线重跑

主目标：

- 解决 `mw3` 构建失败
- 增加 layer aggregation / layer sweep
- 在主集和小集上重跑从弱到强的一组 baseline

建议最小算法栈：

- `popularity`
- `cosine`
- `knn`
- `shallow_mlp`
- `LightFM-like`
- `BPR-MF`
- `two-stage hybrid`
- `listwise hybrid`
- `LambdaMART hybrid`
- `dcas_full_ot`
- `dcas_full_ot_calibrated_target`
- `dcas_full_ot_calibrated_minor`

### Batch D：`Gemini` 线补齐

主目标：

- 建对称的 `Gemini stage3` 配置
- 在主集和小集上按 `CultureMERT` 同协议重跑
- 明确比较“同协议下换 embedding backbone”的效果边界

### Batch E：论文收敛

主目标：

- 更新 `paper/ismir2026_draft.tex`
- 把正文、附录、图表、表格和 limitation 对齐到真实结果

## 6. 当前推荐的执行顺序

推荐按以下顺序推进，不建议并行打散：

1. 先冻结 `V4` 协议与目录规范
2. 再做 `V4` 数据清洗和质量评估
3. 先完整跑 `CultureMERT`
4. 再对称跑 `Gemini`
5. 最后更新论文和图表

## 7. 现有脚本的使用判断

优先复用：

- `dcas/scripts/build_research_dataset_v3.py`
- `dcas/scripts/merge_metadata_dedup.py`
- `dcas/scripts/harmonize_v3_metadata.py`
- `dcas/scripts/validate_dataset.py`
- `dcas/scripts/synthesize_interactions.py`
- `dcas/scripts/build_tracks_from_audio.py`
- `dcas/scripts/build_tracks_with_gemini.py`
- `dcas/scripts/run_recommender_benchmarks.py`
- `dcas/scripts/compare_recommender_runs.py`

建议新增：

- `dcas/scripts/build_research_dataset_v4.py`
- `dcas/scripts/harmonize_v4_metadata.py`
- `dcas/scripts/audit_dataset_v4.py`
- `dcas/scripts/fix_or_report_embedding_failures.py`
- `dcas/scripts/run_culturemert_stage3_matrix.py`
- `dcas/scripts/run_gemini_stage3_matrix.py`
- `dcas/scripts/build_benchmark_index.py`

## 8. 本轮之后的直接任务

下一批代码工作建议按下面顺序开始：

1. 建 `V4` 目录和 manifest 解析入口
2. 把 `routeA_phase2_cn` 升级到与 `V3` 一致的 harmonize 与 interaction 协议
3. 为 `CultureMERT` 增加 layer aggregation 可配置项
4. 追查并修复 `mw3` 失败样本
5. 生成对称的 `Gemini stage3` benchmark 配置
