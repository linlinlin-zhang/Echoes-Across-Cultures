# 项目实验库存与覆盖自检

- 生成时间：`2026-03-20T02:15:11.9802445+08:00`

## 顶层发现
- Gemini 线缺少 stage3 与 public RouteA 对应配置，当前无法与 CultureMERT stage3 做对称对照。
- CultureMERT mw3 embedding 构建存在 16 条失败记录，需要在 V4 前修复或单独解释。
- CultureMERT mw3 对齐后丢失 16 条 track，存在选择性掉样本风险。

## Git 状态
- `## feature/research-v2-platform-and-results...origin/feature/research-v2-platform-and-results [ahead 3]`
- ` D .claude/settings.local.json`
- ` M README.md`
- ` M dcas/embeddings/culturemert.py`
- ` M dcas/scripts/build_tracks_from_audio.py`
- ` M dcas/scripts/run_culturemert_embedding_build.py`
- `?? configs/embedding/culturemert_v3_main_multiwindow_layerweighted.example.json`
- `?? dcas/scripts/attach_metadata_to_tracks.py`
- `?? dcas/scripts/audit_experiment_inventory.py`
- `?? dcas/scripts/audit_project_state.py`
- `?? dcas/scripts/generate_project_figures_zh.py`
- `?? reports/audits/`
- `?? reports/baseline_comparison/v3_main_culturemert/comparisons/`
- `?? reports/baseline_comparison/v3_main_culturemert/eval_beta_vae__seed_42.json`
- `?? reports/baseline_comparison/v3_main_culturemert/eval_beta_vae__seed_43.json`
- `?? reports/baseline_comparison/v3_main_culturemert/eval_beta_vae__seed_44.json`
- `?? reports/baseline_comparison/v3_main_culturemert/eval_factorvae__seed_42.json`
- `?? reports/baseline_comparison/v3_main_culturemert/eval_factorvae__seed_43.json`
- `?? reports/baseline_comparison/v3_main_culturemert/eval_factorvae__seed_44.json`
- `?? reports/baseline_comparison/v3_main_culturemert/eval_three_factor_dcas__seed_42.json`
- `?? reports/baseline_comparison/v3_main_culturemert/eval_three_factor_dcas__seed_43.json`
- `?? reports/baseline_comparison/v3_main_culturemert/eval_three_factor_dcas__seed_44.json`
- `?? reports/baseline_comparison/v3_main_culturemert/eval_vae__seed_42.json`
- `?? reports/baseline_comparison/v3_main_culturemert/eval_vae__seed_43.json`
- `?? reports/baseline_comparison/v3_main_culturemert/eval_vae__seed_44.json`
- `?? reports/baseline_comparison/v3_main_gemini/comparisons/`
- `?? reports/baseline_comparison/v3_main_gemini/eval_beta_vae__seed_42.json`
- `?? reports/baseline_comparison/v3_main_gemini/eval_beta_vae__seed_43.json`
- `?? reports/baseline_comparison/v3_main_gemini/eval_beta_vae__seed_44.json`
- `?? reports/baseline_comparison/v3_main_gemini/eval_factorvae__seed_42.json`
- `?? reports/baseline_comparison/v3_main_gemini/eval_factorvae__seed_43.json`
- `?? reports/baseline_comparison/v3_main_gemini/eval_factorvae__seed_44.json`
- `?? reports/baseline_comparison/v3_main_gemini/eval_three_factor_dcas__seed_42.json`
- `?? reports/baseline_comparison/v3_main_gemini/eval_three_factor_dcas__seed_43.json`
- `?? reports/baseline_comparison/v3_main_gemini/eval_three_factor_dcas__seed_44.json`
- `?? reports/baseline_comparison/v3_main_gemini/eval_vae__seed_42.json`
- `?? reports/baseline_comparison/v3_main_gemini/eval_vae__seed_43.json`
- `?? reports/baseline_comparison/v3_main_gemini/eval_vae__seed_44.json`
- `?? reports/benchmarks/v3_main_culturemert/comparisons/`
- `?? reports/benchmarks/v3_main_culturemert/eval/`
- `?? reports/benchmarks/v3_main_culturemert_open_prepal/comparisons/`
- `?? reports/benchmarks/v3_main_culturemert_open_prepal/eval/`
- `?? reports/benchmarks/v3_main_culturemert_stage3_lightfmlike/`
- `?? reports/benchmarks/v3_main_gemini_embedding2/comparisons/`
- `?? reports/benchmarks/v3_main_gemini_embedding2/eval/`
- `?? reports/benchmarks/v3_main_gemini_harmonized_open_prepal/`
- `?? reports/benchmarks/v3_main_gemini_open_prepal/`
- `?? reports/figures/project_overview_zh_2026-03-20/`

## Embedding Manifest
| path | model_id | n_tracks | dim | max_seconds | window_count | n_errors |
|---|---|---:|---:|---:|---:|---:|
| storage/public/research_dataset_v3/tracks_culturemert_v3_main.npz.manifest.json | ntua-slp/CultureMERT-95M | 1122 | 768 | 30.0 | 1 | 0 |
| storage/public/research_dataset_v3/tracks_culturemert_v3_main_mw3.npz.manifest.json | ntua-slp/CultureMERT-95M | 1106 | 768 | 30.0 | 3 | 16 |
| storage/public/research_dataset_v3/tracks_gemini_embedding2_main.npz.manifest.json | gemini-embedding-2-preview | 1122 | 768 | 30.0 | 1 | 0 |
| storage/public/research_dataset_v3/tracks_gemini_embedding2_v3_main_mw3.npz.manifest.json | gemini-embedding-2-preview | 1122 | 768 | 30.0 | 3 | 0 |

## Train Config
- `train_v3_culturemert_prepal_source.run.json`
- `train_v3_culturemert_stage3.run.json`
- `train_v3_gemini_prepal_source.run.json`
- `train_v3_gemini_prepal_source_harmonized.run.json`

## Benchmark Config
| config | suite_name | raw_kinds | dcas_kinds |
|---|---|---|---|
| log_benchmark_yambda_5b_subset.run.json | yambda_5b_subset_global_log_benchmark | bpr, bpr_tree_hybrid, bpr_two_stage_hybrid, cosine, knn, popularity | ot |
| recommender_benchmark_culturemert.example.json | v2_main_culturemert | cosine, hybrid, knn, mlp, popularity | knn, ot |
| recommender_benchmark_culturemert.run.json | v2_main_culturemert | cosine, hybrid, knn, mlp, popularity | knn, ot |
| recommender_benchmark_gemini.example.json | v2_main_gemini_embedding2 | cosine, hybrid, knn, mlp, popularity | knn, ot |
| recommender_benchmark_gemini.run.json | v2_main_gemini_embedding2 | cosine, hybrid, knn, mlp, popularity | knn, ot |
| recommender_benchmark_public_routeA_phase2_cn.run.json | public_routeA_phase2_cn_lambdamart | bpr, bpr_listwise_hybrid, bpr_tree_hybrid, bpr_two_stage_hybrid, cosine, knn, popularity | ot, ot_calibrated |
| recommender_benchmark_toy.example.json | toy_small_benchmark | cosine, hybrid, knn, mlp, popularity | knn, ot |
| recommender_benchmark_v3_culturemert.run.json | v3_main_culturemert | cosine, hybrid, knn, mlp, popularity | knn, ot |
| recommender_benchmark_v3_culturemert_open.run.json | v3_main_culturemert_open_prepal | cosine, hybrid, knn, mlp, popularity | knn, knn_open, ot, ot_open |
| recommender_benchmark_v3_culturemert_stage3.run.json | v3_main_culturemert_stage3 | cosine, hybrid, knn, mlp, popularity | knn, knn_open, ot, ot_open |
| recommender_benchmark_v3_culturemert_stage3_bpr.run.json | v3_main_culturemert_stage3_bpr | bpr, cosine, hybrid, knn, mlp, popularity | knn, knn_open, ot, ot_open |
| recommender_benchmark_v3_culturemert_stage3_bprhybrid.run.json | v3_main_culturemert_stage3_bprhybrid | bpr, bpr_two_stage_hybrid, cosine, hybrid, knn, popularity | knn, ot |
| recommender_benchmark_v3_culturemert_stage3_bprlistwise.run.json | v3_main_culturemert_stage3_bprlistwise | bpr, bpr_listwise_hybrid, bpr_two_stage_hybrid | ot_calibrated |
| recommender_benchmark_v3_culturemert_stage3_bprlistwise_tuned.run.json | v3_main_culturemert_stage3_bprlistwise_tuned | bpr, bpr_listwise_hybrid, bpr_two_stage_hybrid | ot_calibrated |
| recommender_benchmark_v3_culturemert_stage3_dcascal.run.json | v3_main_culturemert_stage3_dcascal | bpr, bpr_two_stage_hybrid | ot, ot_calibrated |
| recommender_benchmark_v3_culturemert_stage3_lambdamart.run.json | v3_main_culturemert_stage3_lambdamart | bpr, bpr_listwise_hybrid, bpr_tree_hybrid, bpr_two_stage_hybrid | ot_calibrated |
| recommender_benchmark_v3_culturemert_stage3_lightfmlike.run.json | v3_main_culturemert_stage3_lightfmlike | cosine, hybrid, knn, lightfm_like, mlp, popularity | knn, knn_open, ot, ot_open |
| recommender_benchmark_v3_culturemert_stage3_stronghybrid.run.json | v3_main_culturemert_stage3_stronghybrid | cosine, hybrid, knn, mlp, popularity, two_stage_hybrid | knn, knn_open, ot, ot_open |
| recommender_benchmark_v3_gemini.run.json | v3_main_gemini_embedding2 | cosine, hybrid, knn, mlp, popularity | knn, ot |
| recommender_benchmark_v3_gemini_harmonized_open.run.json | v3_main_gemini_harmonized_open_prepal | cosine, hybrid, knn, mlp, popularity | knn, knn_open, ot, ot_open |
| recommender_benchmark_v3_gemini_open.run.json | v3_main_gemini_open_prepal | cosine, hybrid, knn, mlp, popularity | knn, knn_open, ot, ot_open |

## Coverage Summary
- 缺失的 Gemini 对称配置：`10`
  - `train_v3_gemini_stage3.run.json`
  - `recommender_benchmark_v3_gemini_stage3.run.json`
  - `recommender_benchmark_v3_gemini_stage3_bpr.run.json`
  - `recommender_benchmark_v3_gemini_stage3_bprhybrid.run.json`
  - `recommender_benchmark_v3_gemini_stage3_bprlistwise.run.json`
  - `recommender_benchmark_v3_gemini_stage3_dcascal.run.json`
  - `recommender_benchmark_v3_gemini_stage3_lambdamart.run.json`
  - `recommender_benchmark_v3_gemini_stage3_lightfmlike.run.json`
  - `recommender_benchmark_v3_gemini_stage3_stronghybrid.run.json`
  - `recommender_benchmark_public_routeA_phase2_cn_gemini.run.json`
- CultureMERT stage3 report 目录：`9`
- Gemini report 目录：`4`

## CultureMERT mw3 掉样本概览
- 掉样本数：`16`
- 按 culture：
  - `france`: `3`
  - `germany`: `4`
  - `great_britain`: `2`
  - `india`: `3`
  - `italy`: `1`
  - `russia`: `3`
- 按 source_dataset：
  - `Free Music Archive`: `13`
  - `saraga_hindustani`: `3`

## 结果目录
- `public_routeA_phase2_cn_lambdamart` [summary,table]
- `toy_small` [summary,table]
- `v2_main_culturemert` [summary,table]
- `v2_main_gemini_embedding2` [summary,table]
- `v3_main_culturemert` [summary,table]
- `v3_main_culturemert_open_prepal` [summary,table]
- `v3_main_culturemert_stage3` [summary,table]
- `v3_main_culturemert_stage3_bpr` [summary,table]
- `v3_main_culturemert_stage3_bprhybrid` [summary,table]
- `v3_main_culturemert_stage3_bprlistwise` [summary,table]
- `v3_main_culturemert_stage3_bprlistwise_tuned` [summary,table]
- `v3_main_culturemert_stage3_dcascal` [summary,table]
- `v3_main_culturemert_stage3_lambdamart` [summary,table]
- `v3_main_culturemert_stage3_lightfmlike` [summary,table]
- `v3_main_culturemert_stage3_stronghybrid` [summary,table]
- `v3_main_gemini_embedding2` [summary,table]
- `v3_main_gemini_harmonized_open_prepal` [summary,table]
- `v3_main_gemini_open_prepal` [summary,table]
- `yambda_5b_subset_global_log_benchmark` [summary,table]

