# Project Self Audit

## Repo

```text
## feature/research-v2-platform-and-results...origin/feature/research-v2-platform-and-results [ahead 3]
 D .claude/settings.local.json
 M README.md
 M dcas/embeddings/culturemert.py
 M dcas/scripts/build_tracks_from_audio.py
 M dcas/scripts/run_culturemert_embedding_build.py
?? configs/embedding/culturemert_v3_main_multiwindow_layerweighted.example.json
?? dcas/scripts/attach_metadata_to_tracks.py
?? dcas/scripts/audit_project_state.py
?? dcas/scripts/generate_project_figures_zh.py
?? reports/baseline_comparison/v3_main_culturemert/comparisons/
?? reports/baseline_comparison/v3_main_culturemert/eval_beta_vae__seed_42.json
?? reports/baseline_comparison/v3_main_culturemert/eval_beta_vae__seed_43.json
?? reports/baseline_comparison/v3_main_culturemert/eval_beta_vae__seed_44.json
?? reports/baseline_comparison/v3_main_culturemert/eval_factorvae__seed_42.json
?? reports/baseline_comparison/v3_main_culturemert/eval_factorvae__seed_43.json
?? reports/baseline_comparison/v3_main_culturemert/eval_factorvae__seed_44.json
?? reports/baseline_comparison/v3_main_culturemert/eval_three_factor_dcas__seed_42.json
?? reports/baseline_comparison/v3_main_culturemert/eval_three_factor_dcas__seed_43.json
?? reports/baseline_comparison/v3_main_culturemert/eval_three_factor_dcas__seed_44.json
?? reports/baseline_comparison/v3_main_culturemert/eval_vae__seed_42.json
?? reports/baseline_comparison/v3_main_culturemert/eval_vae__seed_43.json
?? reports/baseline_comparison/v3_main_culturemert/eval_vae__seed_44.json
?? reports/baseline_comparison/v3_main_gemini/comparisons/
?? reports/baseline_comparison/v3_main_gemini/eval_beta_vae__seed_42.json
?? reports/baseline_comparison/v3_main_gemini/eval_beta_vae__seed_43.json
?? reports/baseline_comparison/v3_main_gemini/eval_beta_vae__seed_44.json
?? reports/baseline_comparison/v3_main_gemini/eval_factorvae__seed_42.json
?? reports/baseline_comparison/v3_main_gemini/eval_factorvae__seed_43.json
?? reports/baseline_comparison/v3_main_gemini/eval_factorvae__seed_44.json
?? reports/baseline_comparison/v3_main_gemini/eval_three_factor_dcas__seed_42.json
?? reports/baseline_comparison/v3_main_gemini/eval_three_factor_dcas__seed_43.json
?? reports/baseline_comparison/v3_main_gemini/eval_three_factor_dcas__seed_44.json
?? reports/baseline_comparison/v3_main_gemini/eval_vae__seed_42.json
?? reports/baseline_comparison/v3_main_gemini/eval_vae__seed_43.json
?? reports/baseline_comparison/v3_main_gemini/eval_vae__seed_44.json
?? reports/benchmarks/v3_main_culturemert/comparisons/
?? reports/benchmarks/v3_main_culturemert/eval/
?? reports/benchmarks/v3_main_culturemert_open_prepal/comparisons/
?? reports/benchmarks/v3_main_culturemert_open_prepal/eval/
?? reports/benchmarks/v3_main_culturemert_stage3_lightfmlike/
?? reports/benchmarks/v3_main_gemini_embedding2/comparisons/
?? reports/benchmarks/v3_main_gemini_embedding2/eval/
?? reports/benchmarks/v3_main_gemini_harmonized_open_prepal/
?? reports/benchmarks/v3_main_gemini_open_prepal/
?? reports/figures/project_overview_zh_2026-03-20/
```

## Key Findings

| severity | code | message |
|---|---|---|
| warn | paper.placeholder_content | paper/ismir2026_draft.tex still contains placeholder wording and should be synchronized with real experiments. |
| warn | paper.outdated_dataset_description | paper draft still references the older four-domain/1600-track setup instead of the current V3/routeA evidence structure. |
| warn | culturemert.embedding_failures | CultureMERT mw3 embedding build dropped 16 rows and needs audit or recovery. |
| warn | dataset.mw3_alignment_drop | mw3 alignment dropped 16 metadata rows and 140 interactions. |
| warn | dataset.source_confound_risk | V3 has cultures dominated by a single source dataset: france, germany, great_britain, india, italy, modern_english_pop, russia, turkey. |
| info | dataset.metadata_sparse_artist | Some V3 cultures still have zero artist metadata coverage: modern_english_pop, turkey. |
| warn | benchmark.matrix_incomplete | Expected benchmark lines are still missing: v3_main_gemini_stage3_expected, v3_main_gemini_stage3_lambdamart_expected, public_routeA_phase2_cn_gemini_expected, mssd_expected. |
| info | benchmark.mssd_missing | MSSD benchmark artifacts are absent; current repo evidence still lacks that external log line. |

## Dataset Signals

- V3 cultures: `10`
- Single-source cultures: `france, germany, great_britain, india, italy, modern_english_pop, russia, turkey`
- Zero-artist cultures: `modern_english_pop, turkey`
- CultureMERT mw3 embedding errors: `16`
- mw3 metadata rows dropped: `16`
- mw3 interaction rows dropped: `140`

## Benchmark Matrix

| suite | exists | method_count |
|---|---|---:|
| v3_main_culturemert | true | 7 |
| v3_main_culturemert_stage3 | true | 9 |
| v3_main_culturemert_stage3_lambdamart | true | 6 |
| v3_main_gemini_embedding2 | true | 7 |
| public_routeA_phase2_cn_lambdamart | true | 10 |
| yambda_5b_subset_global_log_benchmark | true | 7 |
| v3_main_gemini_stage3_expected | false |  |
| v3_main_gemini_stage3_lambdamart_expected | false |  |
| public_routeA_phase2_cn_gemini_expected | false |  |
| mssd_expected | false |  |

## Paper Audit

| flag | value |
|---|---|
| exists | True |
| contains_placeholder | True |
| contains_draft_evaluation | True |
| contains_four_domain_1600 | True |
| contains_synthetic_placeholder_outcomes | True |
