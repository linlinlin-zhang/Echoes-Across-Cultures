# 路线A-结束文档（Phase 1：多文化公开集基线）

时间：2026-02-27  
负责人：Codex

## 本阶段新增能力
1. `dcas/scripts/import_hf_audio_dataset.py`
- 新增 `hf://datasets/...` 远程路径下载兼容（通过 `huggingface_hub`）。
- 修复 `--limit` 在导入失败场景下不生效的问题（按扫描行数截止）。
- 增加 ClassLabel 自动映射（标签 id -> 标签文本）。

2. 新增多源元数据合并脚本：
- `dcas/scripts/merge_metadata.py`

3. 新增批量推荐评估脚本：
- `dcas/scripts/evaluate_recommender.py`
- 支持 `all users x all target cultures` 的汇总指标输出。

## 路线A Phase 1 实跑结果

### 数据导入（公开集）
- west: `sanchit-gandhi/gtzan` 导入 24
- india: `neerajaabhyankar/hindustani-raag-small` 导入 24
- turkey: `bilal63/turkish_music_emotion_dataset` 导入 24

合并后：
- 72 tracks，3 cultures，分布均衡（24/24/24）

### 表征构建（冻结基座）
- 使用 `ntua-slp/CultureMERT-95M`
- 输出：`tracks.npz`，`dim=768`，`n_tracks=72`

### 下游训练（DCAS）
- 训练参数：`epochs=8`, `batch_size=32`, `lr=2e-3`
- loss 收敛：`1.1884 -> 0.7802`

### 评估（OT 推荐）
- 用户数：24
- 评估对数：72（24 users x 3 target cultures）
- `serendipity_mean = 0.8273`
- `serendipity_std = 0.0474`
- `cultural_calibration_kl_mean = 1.0986`

## 产物路径
- 数据集：`storage/public/routeA_phase1/`
- 数据检查：`reports/routeA_phase1_dataset_profile.json`
- 切分报告：`reports/routeA_phase1_splits/split_report.json`
- 评估报告：`reports/routeA_phase1_eval.json`

## 结论
- 路线 A 的第一轮多文化公开集链路已跑通：
  导入 -> 合并 -> embedding -> 训练 -> 批量评估。
- 当前属于“冻结基座 embedding + 下游 DCAS 训练”，符合路线 A 定义。

## 下一步（Phase 2）
- 扩样本规模（每文化 >= 200）并补充更多文化域。
- 基于 `evaluate_recommender.py` 增加按文化对的指标拆分与置信区间。
- 引入更真实的交互数据（或 PAL 标注约束）替代全部弱监督交互。
