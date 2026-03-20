# Research Dataset V4

## 当前状态

`V4` 目前不只是“可执行骨架”，而是已经跑通了两条 backbone 的主实验入口：

- manifest 审计：`dcas/scripts/audit_dataset_manifest.py`
- schema 归一化：`dcas/scripts/harmonize_v4_metadata.py`
- 数据集审计：`dcas/scripts/audit_dataset_v4.py`
- 构建入口：`dcas/scripts/build_research_dataset_v4.py`
- `CultureMERT stage3`：已完成 `routeA_small + main`
- `Gemini stage3`：已完成 `routeA_small + main`

## 当前 manifest

- 主数据集骨架：`configs/dataset/research_dataset_v4_main_from_v3.json`
- 小数据集骨架：`configs/dataset/research_dataset_v4_routeA_small.json`
- 示例模板：`configs/dataset/research_dataset_v4_manifest.example.json`

## 当前输出

- 主数据集输出：`storage/public/research_dataset_v4/main/`
- 小数据集输出：`storage/public/research_dataset_v4/routeA_small/`
- 主数据集审计：`reports/datasets/research_dataset_v4/main/dataset_profile.md`
- 小数据集审计：`reports/datasets/research_dataset_v4/routeA_small/dataset_profile.md`

## 已暴露出的关键问题

- `main` 仍有明显 `source confound`，`weighted_source_predictability_from_culture = 0.911765`
- `routeA_small` 的 `source confound` 更强，`weighted_source_predictability_from_culture = 1.0`
- `routeA_small` 的 `era` 目前覆盖率仍为 `0.0`
- `duration_sec / sample_rate / channels` 已经在构建阶段通过音频 probing 自动补齐

这些问题已经被正式写入审计输出，不再被 schema 表面完整性掩盖。

## 下一步

1. 把 `CultureMERT` 和 `Gemini` 的 `V4 stage3` 结果并入论文主实验矩阵。
2. 补齐 `routeA_small` 的 `era`，并检查是否需要把它降为推荐字段而不是强制字段。
3. 在 `V4 main` 里增加每个文化的多来源覆盖，降低 source confound。
4. 将 `V4` 审计结果和双 backbone 结果接到论文里的 `Data Card`、`Limitations`、`Experimental Setup`。

## Latest Run Log

- `CultureMERT` build + validation log:
  - `docs/research_dataset_v4/V4_CULTUREMERT_BUILD_VALIDATION_2026-03-20_CN.md`
- `CultureMERT` stage3 benchmark log:
  - `docs/research_dataset_v4/V4_CULTUREMERT_STAGE3_BENCHMARK_RESULTS_2026-03-20_CN.md`
- `Gemini` execution readiness log:
  - `docs/research_dataset_v4/V4_GEMINI_EXECUTION_READINESS_2026-03-20_CN.md`
- `Gemini` stage3 benchmark log:
  - `docs/research_dataset_v4/V4_GEMINI_STAGE3_BENCHMARK_RESULTS_2026-03-20_CN.md`
