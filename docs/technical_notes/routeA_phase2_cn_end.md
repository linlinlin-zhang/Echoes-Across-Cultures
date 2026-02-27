# 路线A-结束文档（Phase 2 CN：加入中国文化域）

时间：2026-02-27  
负责人：Codex

## 本阶段完成内容
1. 四文化公开数据构建：
- west（复用 `routeA_phase2/gtzan`）：160
- india（复用 `routeA_phase2/hindustani`）：160
- turkey（复用 `routeA_phase2/turkish`）：160
- china（新增 `ccmusic-database/erhu_playing_tech`）：160
- 合计：640 tracks

2. 数据合并与交互构建：
- 合并元数据：`storage/public/routeA_phase2_cn/metadata_merged.csv`（640行）
- 弱监督交互：`storage/public/routeA_phase2_cn/interactions.csv`（2682行，80 users）

3. 表征构建（冻结基座）：
- 模型：`ntua-slp/CultureMERT-95M`
- 输出：`storage/public/routeA_phase2_cn/tracks.npz`（`n_tracks=640`, `dim=768`）

4. 质量检查与切分：
- 数据验证：`status=warn`（无 error）
- 告警：`interactions.low_user_activity`（部分用户交互<5）
- 切分：`train/val/test = 0.8/0.1/0.1`（通过）

5. 训练与评估：
- 训练参数：`epochs=10`, `batch_size=64`, `lr=2e-3`
- 训练 loss：`1.3182 -> 0.9330`
- 评估规模：`n_user_culture_evals=320`，`n_skipped=0`

## 指标结果（四文化）
- `serendipity_mean = 0.8109801431`
- `serendipity_std = 0.0719484179`
- `cultural_calibration_kl_mean = 1.3862943611`
- `cultural_calibration_kl_std = 0.0`

### 分文化 serendipity
- china: `0.7914917588`
- india: `0.7903582628`
- turkey: `0.8226207709`
- west: `0.8394497800`

## 与三文化 Phase 2 对比
- 三文化 `serendipity_mean = 0.8619925425`
- 四文化 `serendipity_mean = 0.8109801431`
- 变化：`-0.0510123994`

- 三文化 `cultural_calibration_kl_mean = 1.0986122887`
- 四文化 `cultural_calibration_kl_mean = 1.3862943611`
- 变化：`+0.2876820725`

规模变化：
- users：`60 -> 80`
- cultures：`3 -> 4`
- user-culture evals：`180 -> 320`

## 产物路径
- 数据与模型：`storage/public/routeA_phase2_cn/`
- 数据画像：`reports/routeA_phase2_cn_dataset_profile.json`
- 切分报告：`reports/routeA_phase2_cn_splits/split_report.json`
- 评估报告：`reports/routeA_phase2_cn_eval.json`
- 开始文档：`docs/technical_notes/routeA_phase2_cn_start.md`

## 说明与后续
- 四文化接入已完成，`china` 文化域可作为目标文化进入训练与评估。
- 当前交互仍为弱监督合成数据，适合工程基线验证，不等同真实用户结论。
- 下一步建议：引入真实或半真实跨文化行为数据，并加入统计显著性检验与消融实验，以支撑 ISMIR 论文级结论。
