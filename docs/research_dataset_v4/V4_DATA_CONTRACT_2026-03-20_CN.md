# V4 数据契约

日期：2026-03-20

## 1. 目标

`V4` 的目标不是再做一版“能跑”的数据，而是做一版：

- schema 统一
- 版本可追溯
- 主数据集与小数据集同协议
- 可直接服务 `CultureMERT` 与 `Gemini` 双线
- 可直接服务论文中的 Data Card、benchmark 表格和图表

## 2. 目录结构

建议目录如下：

`storage/public/research_dataset_v4/main/`

- `raw_sources/`
- `metadata_raw.csv`
- `metadata_clean.csv`
- `metadata_harmonized.csv`
- `metadata_release.csv`
- `interactions_synth_single.csv`
- `interactions_synth_mixed.csv`
- `tracks_culturemert_mw3.npz`
- `tracks_culturemert_mw3.npz.manifest.json`
- `tracks_gemini_embedding2_mw3.npz`
- `tracks_gemini_embedding2_mw3.npz.manifest.json`
- `validation_report.json`
- `data_card.json`

`storage/public/research_dataset_v4/routeA_small/`

- 与 `main/` 同结构

`reports/datasets/research_dataset_v4/main/`

- `dataset_profile.json`
- `dataset_profile.md`
- `split_report.json`
- `schema_report.json`
- `missingness_report.json`
- `duplicate_report.json`
- `source_confound_report.json`
- `embedding_build_report.json`

`reports/datasets/research_dataset_v4/routeA_small/`

- 与 `main/` 同结构

## 3. metadata schema

### 3.1 强制字段

- `track_id`
- `culture`
- `audio_path`
- `source_dataset`
- `source_split`
- `source_index`
- `duration_sec`
- `sample_rate`
- `channels`
- `coarse_label`
- `era`
- `region`

### 3.2 建议字段

- `fine_label`
- `label`
- `substyle`
- `instrument`
- `instrument_family`
- `language`
- `title`
- `artist`
- `license`
- `license_note`
- `url`
- `is_instrumental`
- `recording_condition`

### 3.3 V4 新增治理字段

- `schema_version`
- `dataset_version`
- `import_batch`
- `dedup_group_id`
- `dedup_keep`
- `qc_status`
- `qc_notes`
- `embedding_status_culturemert`
- `embedding_status_gemini`
- `drop_reason`

## 4. label 约定

`V4` 中必须明确区分三层标签：

- `culture`
  - 文化域主标签
  - 用于主任务与公平性分析
- `coarse_label`
  - 跨来源可比较的粗粒度类型
  - 用于交互合成和基础统计
- `fine_label`
  - 来源特有的细粒度标签
  - 用于附录分析，不直接作为跨源主标签

要求：

- 主实验默认使用 `coarse_label`
- `label` 仅作为原始保留列，不再直接当主实验字段

## 5. 交互协议

`V4` 必须明确区分三类交互：

- `interactions_synth_single.csv`
  - 单文化偏好弱监督交互
- `interactions_synth_mixed.csv`
  - 跨文化混合弱监督交互
- `interactions_pal_feedback.csv`
  - 真人 PAL 或专家反馈衍生交互/约束

要求：

- benchmark 主线默认使用 `mixed`
- 论文必须显式说明 synthetic 与 human feedback 的边界

## 6. embedding 协议

### 6.1 CultureMERT

统一约定：

- `model_id = ntua-slp/CultureMERT-95M`
- `max_seconds = 30.0`
- `window_count = 3`
- `window_strategy = uniform`
- `window_aggregate = mean`

需要额外比较：

- `layer_indices = [-1]`
- `layer_indices = [-4, -3, -2, -1]`
- `layer_indices = [-4, -3, -2, -1]` and `layer_weights = [0.1, 0.2, 0.3, 0.4]`

### 6.2 Gemini

统一约定：

- `model_id = gemini-embedding-2-preview`
- `max_seconds = 30.0`
- `window_count = 3`
- `window_strategy = uniform`
- `window_aggregate = mean`
- `output_dimensionality = 768`

## 7. 数据质量审计指标

`V4` 数据画像至少要输出以下指标：

- 每文化样本量
- 每来源样本量
- 文化与来源的耦合矩阵
- 时长分布
- 采样率分布
- 声道分布
- 缺失字段覆盖率
- 重复样本比例
- 跨源近重复比例
- `coarse_label` 覆盖率
- `fine_label` 覆盖率
- `artist/title/license` 覆盖率
- embedding 构建失败率
- embedding 构建失败的文化分布
- embedding 构建失败的来源分布
- split 后的文化分布漂移
- interactions 的 user activity 分布
- interactions 的 track coverage
- minority track 占比
- source predictability probe

## 8. 处理顺序

`V4` 必须按以下顺序构建：

1. raw import
2. metadata merge
3. schema normalize
4. dedup
5. harmonize label
6. qc and validation
7. synthesize interactions
8. build embeddings
9. align metadata/interactions to built tracks
10. split and freeze
11. export data card

## 9. 可复用脚本

可以直接复用或轻改：

- `dcas/scripts/import_hf_audio_dataset.py`
- `dcas/scripts/import_hf_repo_audio_archive.py`
- `dcas/scripts/merge_metadata_dedup.py`
- `dcas/scripts/harmonize_v3_metadata.py`
- `dcas/scripts/validate_dataset.py`
- `dcas/scripts/synthesize_interactions.py`
- `dcas/scripts/build_tracks_from_audio.py`
- `dcas/scripts/build_tracks_with_gemini.py`
- `dcas/scripts/make_splits.py`

## 10. 必须新增的脚本

`V4` 至少需要新增：

- `build_research_dataset_v4.py`
  - 统一驱动 main 和 routeA_small
- `harmonize_v4_metadata.py`
  - 统一 schema 与 label 层
- `audit_dataset_v4.py`
  - 汇总所有质量指标
- `fix_or_report_embedding_failures.py`
  - 追查和处理 embedding 丢样本
- `build_data_card.py`
  - 输出论文可用的数据卡
