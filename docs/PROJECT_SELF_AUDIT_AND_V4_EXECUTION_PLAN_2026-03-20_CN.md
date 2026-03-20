# 项目自检与 V4 执行计划

日期：2026-03-20

## 1. 这轮文档的目的

这份文档用于把当前项目状态收敛成一条可执行主线，避免继续在：

- 旧草稿与新实验结果脱节
- 数据版本与 benchmark 版本不清晰
- CultureMERT / Gemini 两条线不对称
- routeA / Yambda / V3 的证据层级混写

这四类问题上继续累积技术债和论文风险。

对应的自动化审计输出位于：

- `reports/audits/project_self_audit_2026-03-20/audit_report.md`
- `reports/audits/project_self_audit_2026-03-20/audit_report.json`

## 2. 当前最高优先级问题

### 2.1 论文与真实实验状态不同步

当前 `paper/ismir2026_draft.tex` 仍使用旧的“四域 1600 tracks + placeholder 结果”叙事，而仓库主证据已经转到：

- `Research Dataset V3`
- `public routeA_phase2_cn`
- `Yambda-5B subset`
- `LambdaMART / 强 hybrid / calibrated DCAS`

这会直接造成 claim-evidence mismatch。

### 2.2 CultureMERT 主线仍有工程缺口

当前 `tracks_culturemert_v3_main_mw3.npz` 对齐后只保留 `1106` 条 track，manifest 中记录 `16` 条 embedding 失败。

相关文件：

- `storage/public/research_dataset_v3/tracks_culturemert_v3_main_mw3.npz.manifest.json`
- `storage/public/research_dataset_v3/metadata_v3_main_harmonized_mw3.csv.align_report.json`

这意味着：

- 结果并非完全基于原始 `1122` 条样本
- 丢样本与交互丢失必须纳入后续结果解释

### 2.3 数据集主张仍受 source confound 限制

`V3` 里多个文化域仍强绑定单一数据源，例如：

- `india -> saraga_hindustani`
- `turkey -> bilal63/turkish_music_emotion_dataset`
- `modern_english_pop -> vtsouval/mtg_jamendo_autotagging`

这会让“学到文化”与“学到来源风格”之间存在混杂。

### 2.4 benchmark 矩阵仍不完整

当前已有：

- `v3_main_culturemert`
- `v3_main_culturemert_stage3`
- `v3_main_culturemert_stage3_lambdamart`
- `v3_main_gemini_embedding2`
- `public_routeA_phase2_cn_lambdamart`
- `yambda_5b_subset_global_log_benchmark`

当前缺失的关键线：

- `Gemini stage3`
- `Gemini stage3 + stronger baselines`
- `Gemini routeA`
- `MSSD`

### 2.5 PAL 仍需要严格区分 simulated 与 real

当前仓库已经有：

- pseudo-PAL / simulated constraints
- real PAL workflow

但真实专家标注证据还没有形成正式主结果，因此论文只能写：

- `PAL pipeline implemented`
- `real PAL pilot pending / partial`

不能写成“participatory feedback 已被完整实证验证”。

## 3. 证据层级的推荐组织方式

### 3.1 主证据：V3

回答的问题：

- 在自建但可复现的 controlled cross-cultural setting 上，`DCAS` 的结构性机制是否成立。

主指标：

- `serendipity`
- `cultural_calibration_kl`
- `minority_exposure_at_k`
- `target_culture_prob_mean`

### 3.2 辅助复核：routeA_phase2_cn

回答的问题：

- 结论是否严重依赖 `V3` 的构造方式。

推荐写法：

- `public-source, self-constructed multi-cultural benchmark`

不建议写成：

- `standard public benchmark`

### 3.3 边界测试：Yambda-5B subset

回答的问题：

- 如果切换到通用公开日志排序任务，系统边界在哪里。

推荐定位：

- appendix / supplementary
- 外部日志排序补充线

### 3.4 人类反馈：PAL

回答的问题：

- 小规模高信息量人工反馈是否能修复边界样本。

当前建议：

- 把 `simulated PAL` 与 `real PAL pilot` 分成两个子节，不混写。

## 4. 当前核心实验索引

| 层级 | 名称 | 数据 | embedding | 配置 | 结果目录 | 当前定位 |
|---|---|---|---|---|---|---|
| 主线 | `v3_main_culturemert_stage3_lambdamart` | `research_dataset_v3` | `CultureMERT mw3` | `configs/benchmark/recommender_benchmark_v3_culturemert_stage3_lambdamart.run.json` | `reports/benchmarks/v3_main_culturemert_stage3_lambdamart/` | 当前主 benchmark |
| 主线前序 | `v3_main_culturemert_stage3` | `research_dataset_v3` | `CultureMERT mw3` | `configs/benchmark/recommender_benchmark_v3_culturemert_stage3.run.json` | `reports/benchmarks/v3_main_culturemert_stage3/` | stage3 升级证据 |
| 旧主线对照 | `v3_main_culturemert` | `research_dataset_v3` | `CultureMERT` | `configs/benchmark/recommender_benchmark_v3_culturemert.run.json` | `reports/benchmarks/v3_main_culturemert/` | pre-PAL 对照 |
| embedding 对照 | `v3_main_gemini_embedding2` | `research_dataset_v3` | `Gemini Embedding 2` | `configs/benchmark/recommender_benchmark_v3_gemini.run.json` | `reports/benchmarks/v3_main_gemini_embedding2/` | 现有 Gemini 主对照 |
| 公开来源复核 | `public_routeA_phase2_cn_lambdamart` | `routeA_phase2_cn` | `CultureMERT` | `configs/benchmark/recommender_benchmark_public_routeA_phase2_cn.run.json` | `reports/benchmarks/public_routeA_phase2_cn_lambdamart/` | public-source sanity check |
| 外部日志补充 | `yambda_5b_subset_global_log_benchmark` | `Yambda-5B subset` | mixed / log setting | `configs/benchmark/log_benchmark_yambda_5b_subset.run.json` | `reports/benchmarks/yambda_5b_subset_global_log_benchmark/` | 边界测试 |

## 5. V4 数据集的建议目标

`V4` 不应该只是“继续堆更多歌”，而应解决以下科学性问题：

1. schema 真正统一  
2. `label / coarse_label / source_dataset / era / language / instrument_family` 形成稳定审计链  
3. source confound 有量化披露  
4. duplicate / near-duplicate / dropped rows 有门禁  
5. interactions 版本化，并区分：
   - single-culture synthetic
   - mixed-culture synthetic
   - future human-in-the-loop interactions

### 5.1 V4 最小目录建议

建议新增：

- `storage/public/research_dataset_v4/metadata_v4_main.csv`
- `storage/public/research_dataset_v4/summary_v4_main.json`
- `storage/public/research_dataset_v4/profiles/`
- `storage/public/research_dataset_v4/audits/`
- `storage/public/research_dataset_v4/interactions/`
- `storage/public/research_dataset_v4/manifests/`

### 5.2 V4 必做量化评估

- 每文化域样本量、时长分布、来源分布
- 单一来源占比
- `artist/title/license/language/instrument` 覆盖率
- duplicate / near-duplicate
- embedding 成功率与 dropped rows
- tracks/interactions/constraints 对齐损耗
- synthetic interactions 的 user activity 分布
- benchmark 可评测性门禁

## 6. 第一批执行顺序

### Batch 1：自检与规范收敛

目标：

- 先把“说法”和“真实状态”对齐。

交付物：

- 自动化审计：`dcas/scripts/audit_project_state.py`
- 审计结果：`reports/audits/project_self_audit_2026-03-20/`
- 本文档

### Batch 2：V4 数据构建与审计

目标：

- 生成 `V4 main`
- 生成 `routeA_phase2_cn_v2` 或等价小数据线新版
- 形成统一 profile / audit 报告

### Batch 3：CultureMERT 全量重跑

目标：

- 修复 build failed 样本问题
- 做 layer selection / layer weighting
- 补更完整 baseline 梯度

最小实验矩阵建议：

- `last layer`
- `mean(last 4 layers)` 或显式 `weighted layers`
- `V3 main`
- `routeA small line`
- baselines 从弱到强分层

### Batch 4：Gemini 全链路补齐

目标：

- 跑出与 CultureMERT 主线对齐的对照矩阵
- 证明 `DCAS` 不是只依赖某一个 backbone

最小实验矩阵建议：

- `V3 main + Gemini stage3`
- `V3 main + Gemini strong baselines`
- `routeA + Gemini`

## 7. 当前不建议做的事

- 继续扩更多外部 benchmark，而不先补齐主线矩阵
- 在论文里提前写强人类反馈结论
- 不经版本封版就直接开始大规模重跑
- 一边改模型结构一边改数据协议，导致解释失控

## 8. 下一步落地优先级

1. 同步论文草稿与真实实验状态  
2. 建立统一 experiment index  
3. 做 V4 schema + audit gate  
4. 重跑 CultureMERT  
5. 补齐 Gemini  
6. 再进入真实 PAL pilot
