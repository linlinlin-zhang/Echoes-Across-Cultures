# Experiment Index

This file is the single entry point for mapping each core experiment to its:

- dataset line
- embedding backbone
- benchmark or training config
- result directory
- intended paper usage

## Workspace Orientation

- `docs/workspace_index/PROJECT_SPINE_2026-04-18_CN.md`: current high-level map of the repository, including the paper mainline, PAL line, prototype line, and historical residue.

## Cross-Cultural Main Line

| tag | dataset | embedding | config | result | paper usage |
|---|---|---|---|---|---|
| `v3_main_culturemert` | `research_dataset_v3` | `CultureMERT` | `configs/benchmark/recommender_benchmark_v3_culturemert.run.json` | `reports/benchmarks/v3_main_culturemert/` | pre-PAL backbone comparison |
| `v3_main_culturemert_stage3` | `research_dataset_v3` | `CultureMERT mw3` | `configs/benchmark/recommender_benchmark_v3_culturemert_stage3.run.json` | `reports/benchmarks/v3_main_culturemert_stage3/` | stage3 training protocol evidence |
| `v3_main_culturemert_stage3_lambdamart` | `research_dataset_v3` | `CultureMERT mw3` | `configs/benchmark/recommender_benchmark_v3_culturemert_stage3_lambdamart.run.json` | `reports/benchmarks/v3_main_culturemert_stage3_lambdamart/` | current main benchmark |
| `v4_main_culturemert_stage3_lambdamart` | `research_dataset_v4/main` | `CultureMERT mw3` | `configs/benchmark/recommender_benchmark_v4_main_culturemert_stage3_lambdamart.run.json` | `reports/benchmarks/v4_main_culturemert_stage3_lambdamart/` | current V4 main benchmark |
| `v4_routeA_small_culturemert_stage3_lambdamart` | `research_dataset_v4/routeA_small` | `CultureMERT mw3` | `configs/benchmark/recommender_benchmark_v4_routeA_small_culturemert_stage3_lambdamart.run.json` | `reports/benchmarks/v4_routeA_small_culturemert_stage3_lambdamart/` | public-source V4 sanity check |
| `v4_main_gemini_stage3_lambdamart` | `research_dataset_v4/main` | `Gemini Embedding 2 mw3` | `configs/benchmark/recommender_benchmark_v4_main_gemini_stage3_lambdamart.run.json` | `reports/benchmarks/v4_main_gemini_stage3_lambdamart/` | V4 main backbone transfer check |
| `v4_routeA_small_gemini_stage3_lambdamart` | `research_dataset_v4/routeA_small` | `Gemini Embedding 2 mw3` | `configs/benchmark/recommender_benchmark_v4_routeA_small_gemini_stage3_lambdamart.run.json` | `reports/benchmarks/v4_routeA_small_gemini_stage3_lambdamart/` | V4 small backbone transfer sanity check |
| `v4_main_culturemert_lambdamart` | `research_dataset_v4/main` | `CultureMERT mw3` | `configs/benchmark/recommender_benchmark_v4_main_culturemert_lambdamart.run.json` | `reports/benchmarks/v4_main_culturemert_lambdamart/` | simplified V4 main ablation |
| `v4_routeA_small_culturemert_lambdamart` | `research_dataset_v4/routeA_small` | `CultureMERT mw3` | `configs/benchmark/recommender_benchmark_v4_routeA_small_culturemert_lambdamart.run.json` | `reports/benchmarks/v4_routeA_small_culturemert_lambdamart/` | simplified V4 small ablation |
| `public_routeA_phase2_cn_lambdamart` | `routeA_phase2_cn` | `CultureMERT` | `configs/benchmark/recommender_benchmark_public_routeA_phase2_cn.run.json` | `reports/benchmarks/public_routeA_phase2_cn_lambdamart/` | public-source sanity check |

## Embedding Comparison Line

| tag | dataset | embedding | config | result | paper usage |
|---|---|---|---|---|---|
| `v3_main_gemini_embedding2` | `research_dataset_v3` | `Gemini Embedding 2` | `configs/benchmark/recommender_benchmark_v3_gemini.run.json` | `reports/benchmarks/v3_main_gemini_embedding2/` | backbone comparison against CultureMERT |

## External Log Benchmark Line

| tag | dataset | metric family | config | result | paper usage |
|---|---|---|---|---|---|
| `yambda_5b_subset_global_log_benchmark` | `Yambda-5B subset` | `Recall/NDCG/MRR` | `configs/benchmark/log_benchmark_yambda_5b_subset.run.json` | `reports/benchmarks/yambda_5b_subset_global_log_benchmark/` | appendix / external log boundary test |

## Training Runs

| tag | tracks | constraints | interactions | config | model |
|---|---|---|---|---|---|
| `train_v3_culturemert_stage3` | `storage/public/research_dataset_v3/tracks_culturemert_v3_main_mw3.npz` | `storage/pal/v3_main_prepal/pseudo_constraints_v1_mw3.jsonl` | `storage/public/research_dataset_v3/interactions_v3_main_mixed_mw3.csv` | `configs/train/train_v3_culturemert_stage3.run.json` | `storage/models/dcas_full_v3_main_culturemert_stage3.pt` |
| `train_v4_main_culturemert_stage3` | `storage/public/research_dataset_v4/main/tracks_culturemert_mw3.npz` | `storage/pal/v4_main_prepal/pseudo_constraints_v1.jsonl` | `storage/public/research_dataset_v4/main/interactions_synth_mixed.csv` | `configs/train/train_v4_main_culturemert_stage3.run.json` | `storage/models/dcas_full_v4_main_culturemert_stage3.pt` |
| `train_v4_routeA_small_culturemert_stage3` | `storage/public/research_dataset_v4/routeA_small/tracks_culturemert_mw3.npz` | `storage/pal/v4_routeA_small_prepal/pseudo_constraints_v1.jsonl` | `storage/public/research_dataset_v4/routeA_small/interactions_synth_mixed.csv` | `configs/train/train_v4_routeA_small_culturemert_stage3.run.json` | `storage/models/dcas_full_v4_routeA_small_culturemert_stage3.pt` |
| `train_v4_main_gemini_stage3` | `storage/public/research_dataset_v4/main/tracks_gemini_embedding2_mw3.npz` | `storage/pal/v4_main_prepal/pseudo_constraints_v1.jsonl` | `storage/public/research_dataset_v4/main/interactions_synth_mixed.csv` | `configs/train/train_v4_main_gemini_stage3.run.json` | `storage/models/dcas_full_v4_main_gemini_stage3.pt` |
| `train_v4_routeA_small_gemini_stage3` | `storage/public/research_dataset_v4/routeA_small/tracks_gemini_embedding2_mw3.npz` | `storage/pal/v4_routeA_small_prepal/pseudo_constraints_v1.jsonl` | `storage/public/research_dataset_v4/routeA_small/interactions_synth_mixed.csv` | `configs/train/train_v4_routeA_small_gemini_stage3.run.json` | `storage/models/dcas_full_v4_routeA_small_gemini_stage3.pt` |
| `train_v3_culturemert_prepal_source` | `storage/public/research_dataset_v3/tracks_culturemert_v3_main.npz` | `storage/pal/v3_main_prepal/pseudo_constraints_v1.jsonl` | `storage/public/research_dataset_v3/interactions_v3_main.csv` | `configs/train/train_v3_culturemert_prepal_source.run.json` | `storage/models/dcas_full_v3_main_culturemert.pt` |
| `train_v3_gemini_prepal_source` | `storage/public/research_dataset_v3/tracks_gemini_embedding2_main.npz` | `storage/pal/v3_main_prepal/pseudo_constraints_v1.jsonl` | `storage/public/research_dataset_v3/interactions_v3_main.csv` | `configs/train/train_v3_gemini_prepal_source.run.json` | `storage/models/dcas_full_v3_main_gemini.pt` |

## PAL Workflow

| artifact | path | usage |
|---|---|---|
| real PAL workflow | `docs/research_dataset_v3/V3_REAL_PAL_WORKFLOW_CN.md` | real pilot procedure |
| V4 real PAL workflow | `docs/research_dataset_v4/V4_REAL_PAL_WORKFLOW_2026-03-21_CN.md` | current V4 CultureMERT human PAL runbook |
| V4 PAL migration log | `docs/research_dataset_v4/V4_PAL_MIGRATION_AND_ALIGNMENT_2026-04-18_CN.md` | records how `v4_main_annotation` human PAL was aligned to the current V4 main benchmark line |
| phase3 PAL runner | `dcas/scripts/run_phase3_pal.py` | compare before/after PAL rounds |
| PAL platform runner | `dcas/scripts/run_pal_platform.py` | annotation workflow |
| V4 real PAL bundle prep | `dcas/scripts/prepare_real_pal_bundle.py` | generate candidate pool, pilot sheet, and round-1 sheet |
| V4 PAL prep config | `configs/pal/pal_v4_main_culturemert_prepare.run.json` | build current real PAL bundle |
| V4 PAL real config | `configs/pal/pal_v4_main_culturemert_real.run.json` | run baseline vs real PAL after annotation returns |
| V4 PAL migrated stage3 config | `configs/pal/pal_v4_main_culturemert_real_from_v4_main_annotation_stage3.run.json` | benchmark-aligned warm-start real PAL run using `workspace_assets/pal_exports/pal_v4_main_annotation_human_export_200pairs.csv` |
| V4 PAL migrated calibration sweep | `configs/benchmark/recommender_benchmark_v4_main_culturemert_real_pal_stage3_calibration_sweep.run.json` | searches calibrated rerank operating points for the migrated real PAL checkpoint |
| V4 PAL ultralight focus benchmark | `configs/benchmark/recommender_benchmark_v4_main_culturemert_real_pal_ultralight_stage3_focus.run.json` | checks whether extra light PAL fine-tuning improves over the best migrated PAL operating points |
| annotation sheet export | `dcas/scripts/export_pal_annotation_sheet.py` | human labeling |

## Current Gaps

These lines are still expected but not yet present as completed benchmark outputs:

- `v3_main_gemini_stage3`
- `v3_main_gemini_stage3_lambdamart`
- `public_routeA_phase2_cn_gemini`
- `mssd`
