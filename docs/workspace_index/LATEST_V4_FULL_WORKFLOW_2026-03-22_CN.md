# 最新版项目全流程：V4 主线

日期：2026-03-22

这份文档只回答一个问题：如果现在把工作区视为一个“最新版本项目”，那么完整主流程应该怎么走，应该看哪些文件。

## 0. 主线定义

- 当前主线版本：`v4`
- 主证据集：`V4 main`
- 小线验证集：`routeA_small`
- 主训练范式：`stage3`
- 主 backbones：
  - `CultureMERT`
  - `Gemini`
- 主论文入口：[ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex)

## 1. 数据契约与数据集构建

### 目标
- 统一多来源音乐数据的 metadata、culture、source、embedding 接口
- 构成后续训练、benchmark、PAL 都能复用的数据底座

### 关键配置
- [research_dataset_v4_main_from_v3.json](E:/Desktop/Echo/configs/dataset/research_dataset_v4_main_from_v3.json)
- [research_dataset_v4_routeA_small.json](E:/Desktop/Echo/configs/dataset/research_dataset_v4_routeA_small.json)

### 关键说明
- [V4_DATA_CONTRACT_2026-03-20_CN.md](E:/Desktop/Echo/docs/research_dataset_v4/V4_DATA_CONTRACT_2026-03-20_CN.md)
- [README.md](E:/Desktop/Echo/docs/research_dataset_v4/README.md)

### 数据产物
- 主数据集目录：[main](E:/Desktop/Echo/storage/public/research_dataset_v4/main)
- 小线目录：[routeA_small](E:/Desktop/Echo/storage/public/research_dataset_v4/routeA_small)

### 主线阅读重点
- `metadata_release.csv`
- `manifest.snapshot.json`
- `data_card.json`
- `validation_report.json`
- `interactions_synth_mixed.csv`

## 2. Backbone Embedding 产物

### CultureMERT
- 主线输入：[tracks_culturemert_mw3.npz](E:/Desktop/Echo/storage/public/research_dataset_v4/main/tracks_culturemert_mw3.npz)
- sanity-check 输入：[tracks_culturemert_mw3.npz](E:/Desktop/Echo/storage/public/research_dataset_v4/routeA_small/tracks_culturemert_mw3.npz)

### Gemini
- 主线输入：[tracks_gemini_embedding2_mw3.npz](E:/Desktop/Echo/storage/public/research_dataset_v4/main/tracks_gemini_embedding2_mw3.npz)
- sanity-check 输入：[tracks_gemini_embedding2_mw3.npz](E:/Desktop/Echo/storage/public/research_dataset_v4/routeA_small/tracks_gemini_embedding2_mw3.npz)

### 说明文档
- [V4_CULTUREMERT_BUILD_VALIDATION_2026-03-20_CN.md](E:/Desktop/Echo/docs/research_dataset_v4/V4_CULTUREMERT_BUILD_VALIDATION_2026-03-20_CN.md)
- [V4_GEMINI_EXECUTION_READINESS_2026-03-20_CN.md](E:/Desktop/Echo/docs/research_dataset_v4/V4_GEMINI_EXECUTION_READINESS_2026-03-20_CN.md)

## 3. Stage-3 训练

### 主训练配置
- CultureMERT 主线：[train_v4_main_culturemert_stage3.run.json](E:/Desktop/Echo/configs/train/train_v4_main_culturemert_stage3.run.json)
- Gemini 主线：[train_v4_main_gemini_stage3.run.json](E:/Desktop/Echo/configs/train/train_v4_main_gemini_stage3.run.json)

### sanity-check 训练配置
- CultureMERT 小线：[train_v4_routeA_small_culturemert_stage3.run.json](E:/Desktop/Echo/configs/train/train_v4_routeA_small_culturemert_stage3.run.json)
- Gemini 小线：[train_v4_routeA_small_gemini_stage3.run.json](E:/Desktop/Echo/configs/train/train_v4_routeA_small_gemini_stage3.run.json)

### 主要 checkpoint
- [dcas_full_v4_main_culturemert_stage3.pt](E:/Desktop/Echo/storage/models/dcas_full_v4_main_culturemert_stage3.pt)
- [dcas_full_v4_main_gemini_stage3.pt](E:/Desktop/Echo/storage/models/dcas_full_v4_main_gemini_stage3.pt)
- [dcas_full_v4_routeA_small_culturemert_stage3.pt](E:/Desktop/Echo/storage/models/dcas_full_v4_routeA_small_culturemert_stage3.pt)
- [dcas_full_v4_routeA_small_gemini_stage3.pt](E:/Desktop/Echo/storage/models/dcas_full_v4_routeA_small_gemini_stage3.pt)

## 4. 主 Benchmark

### 主证据 benchmark 配置
- [recommender_benchmark_v4_main_culturemert_stage3_lambdamart.run.json](E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v4_main_culturemert_stage3_lambdamart.run.json)
- [recommender_benchmark_v4_main_gemini_stage3_lambdamart.run.json](E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v4_main_gemini_stage3_lambdamart.run.json)

### sanity-check benchmark 配置
- [recommender_benchmark_v4_routeA_small_culturemert_stage3_lambdamart.run.json](E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v4_routeA_small_culturemert_stage3_lambdamart.run.json)
- [recommender_benchmark_v4_routeA_small_gemini_stage3_lambdamart.run.json](E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v4_routeA_small_gemini_stage3_lambdamart.run.json)

### benchmark 结果目录
- [v4_main_culturemert_stage3_lambdamart](E:/Desktop/Echo/reports/benchmarks/v4_main_culturemert_stage3_lambdamart)
- [v4_main_gemini_stage3_lambdamart](E:/Desktop/Echo/reports/benchmarks/v4_main_gemini_stage3_lambdamart)
- [v4_routeA_small_culturemert_stage3_lambdamart](E:/Desktop/Echo/reports/benchmarks/v4_routeA_small_culturemert_stage3_lambdamart)
- [v4_routeA_small_gemini_stage3_lambdamart](E:/Desktop/Echo/reports/benchmarks/v4_routeA_small_gemini_stage3_lambdamart)

### 结果解读文档
- [V4_CULTUREMERT_STAGE3_BENCHMARK_RESULTS_2026-03-20_CN.md](E:/Desktop/Echo/docs/research_dataset_v4/V4_CULTUREMERT_STAGE3_BENCHMARK_RESULTS_2026-03-20_CN.md)
- [V4_GEMINI_STAGE3_BENCHMARK_RESULTS_2026-03-20_CN.md](E:/Desktop/Echo/docs/research_dataset_v4/V4_GEMINI_STAGE3_BENCHMARK_RESULTS_2026-03-20_CN.md)

## 5. Calibration Hyperparameter Sweep

### 已完成 sweep
- [recommender_benchmark_v4_main_culturemert_stage3_calibration_sweep.run.json](E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v4_main_culturemert_stage3_calibration_sweep.run.json)
- [recommender_benchmark_v4_routeA_small_culturemert_stage3_calibration_sweep.run.json](E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v4_routeA_small_culturemert_stage3_calibration_sweep.run.json)
- [recommender_benchmark_v4_routeA_small_gemini_stage3_calibration_sweep.run.json](E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v4_routeA_small_gemini_stage3_calibration_sweep.run.json)

### 结果目录
- [v4_main_culturemert_stage3_calibration_sweep](E:/Desktop/Echo/reports/hparam/v4_main_culturemert_stage3_calibration_sweep)
- [v4_routeA_small_culturemert_stage3_calibration_sweep](E:/Desktop/Echo/reports/hparam/v4_routeA_small_culturemert_stage3_calibration_sweep)
- [v4_routeA_small_gemini_stage3_calibration_sweep](E:/Desktop/Echo/reports/hparam/v4_routeA_small_gemini_stage3_calibration_sweep)

### 文档
- [V4_CALIBRATION_HPARAM_SWEEP_RESULTS_2026-03-21_CN.md](E:/Desktop/Echo/docs/research_dataset_v4/V4_CALIBRATION_HPARAM_SWEEP_RESULTS_2026-03-21_CN.md)

## 6. PAL-ready 流程

### 配置
- [pal_v4_main_culturemert_prepare.run.json](E:/Desktop/Echo/configs/pal/pal_v4_main_culturemert_prepare.run.json)
- [pal_v4_main_culturemert_real.run.json](E:/Desktop/Echo/configs/pal/pal_v4_main_culturemert_real.run.json)

### 说明文档
- [V4_REAL_PAL_WORKFLOW_2026-03-21_CN.md](E:/Desktop/Echo/docs/research_dataset_v4/V4_REAL_PAL_WORKFLOW_2026-03-21_CN.md)

### 证据边界
- 当前可以写：`PAL-ready`
- 当前不宜写：`已完成真人闭环并得到最终论文级结果`

## 7. 审计与论文资产

### 工作区诊断
- [phase1_workspace_scan_2026-03-21](E:/Desktop/Echo/reports/audits/phase1_workspace_scan_2026-03-21)
- [phase2_academic_archaeology_2026-03-21](E:/Desktop/Echo/reports/audits/phase2_academic_archaeology_2026-03-21)

### 论文素材
- [manuscript](E:/Desktop/Echo/docs/manuscript)

### 当前主稿
- [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex)

## 8. 现在不应当混进最新版主线的内容

### 历史但仍有参考价值
- `docs/research_dataset_v2/`
- `docs/research_dataset_v3/`
- `reports/baseline_comparison/v3_*`
- `reports/ablation/v2_*`

### 支线
- `storage/public/routeA_phase*`
- `reports/routeA_phase*`

### 临时或 smoke
- `configs/embedding/*tmp.json`
- `storage/public/research_dataset_v4/routeA_small_smoke/`
- 根目录 `tmp_*`

## 9. 如果现在要从头复现最新版本，最短路径是

1. 用 `V4 main` 数据产物
2. 选择 `CultureMERT` 或 `Gemini` backbone 输入
3. 跑 `train_v4_main_*_stage3.run.json`
4. 跑 `recommender_benchmark_v4_main_*_stage3_lambdamart.run.json`
5. 如需写论文超参段，再接 calibration sweep
6. 如需 human-in-the-loop 叙事，再接 `pal_v4_main_culturemert_*`
7. 最后把结果接到 `docs/manuscript/` 与 `paper/ismir2026_draft.tex`

一句话总结：
- `v4` 是当前主线
- `V4 main` 是当前主证据
- `routeA_small` 是 sanity-check
- `docs/manuscript + paper/ismir2026_draft.tex` 是当前论文写作终点
