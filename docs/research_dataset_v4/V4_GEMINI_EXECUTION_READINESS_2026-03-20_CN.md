# V4 Gemini 执行准备状态

> 说明：本文件记录的是启动前准备状态。`2026-03-20` 当天 `V4 routeA_small` 与 `V4 main` 的 `Gemini stage3` 已经实际跑通，结果见 `docs/research_dataset_v4/V4_GEMINI_STAGE3_BENCHMARK_RESULTS_2026-03-20_CN.md`。

日期: 2026-03-20

## 1. 当前结论

`Gemini` 线在当前仓库里已经具备完整的工程骨架，但还没有真正跑通 `V4 routeA_small` 和 `V4 main`。

现状可以分成两部分:

- 已具备: `V4` 数据构建脚本、`Gemini` embedding 构建入口、`V3 Gemini` 训练与 benchmark 参考配置、以及本轮新增的 `V4 Gemini stage3` train/benchmark 配置
- 仍缺失: `V4` 的真实 `Gemini` tracks、对应 checkpoint、benchmark 报告，以及可用的 `GEMINI_API_KEY`

## 2. 可直接复用的工程入口

- 数据构建:
  - `dcas/scripts/build_research_dataset_v4.py`
  - `dcas/scripts/build_tracks_with_gemini.py`
  - `dcas/scripts/run_gemini_embedding_build.py`
- `V4` manifest:
  - `configs/dataset/research_dataset_v4_routeA_small.json`
  - `configs/dataset/research_dataset_v4_main_from_v3.json`
- `V3 Gemini` 参考配置:
  - `configs/train/train_v3_gemini_prepal_source.run.json`
  - `configs/train/train_v3_gemini_prepal_source_harmonized.run.json`
  - `configs/benchmark/recommender_benchmark_v3_gemini.run.json`
  - `configs/benchmark/recommender_benchmark_v3_gemini_open.run.json`
  - `configs/benchmark/recommender_benchmark_v3_gemini_harmonized_open.run.json`
- 本轮新增 `V4 Gemini stage3` 配置:
  - `configs/train/train_v4_routeA_small_gemini_stage3.run.json`
  - `configs/train/train_v4_main_gemini_stage3.run.json`
  - `configs/benchmark/recommender_benchmark_v4_routeA_small_gemini_stage3_lambdamart.run.json`
  - `configs/benchmark/recommender_benchmark_v4_main_gemini_stage3_lambdamart.run.json`

## 3. 当前明确缺口

### 3.1 真实 embedding 产物尚未生成

当前不存在以下文件:

- `storage/public/research_dataset_v4/routeA_small/tracks_gemini_embedding2_mw3.npz`
- `storage/public/research_dataset_v4/main/tracks_gemini_embedding2_mw3.npz`

因此以下产物也都还不存在:

- `storage/models/dcas_full_v4_routeA_small_gemini_stage3.pt`
- `storage/models/dcas_full_v4_main_gemini_stage3.pt`
- `reports/benchmarks/v4_routeA_small_gemini_stage3_lambdamart/`
- `reports/benchmarks/v4_main_gemini_stage3_lambdamart/`

### 3.2 认证尚未配置

`Gemini` live build 当前最大的阻塞是认证:

- 代码读取的环境变量是 `GEMINI_API_KEY`
- 两份 `V4 manifest` 目前都没有填写 `api_key` 或 `api_key_file`
- 仓库内现有 `V4 Gemini` 只看到 smoke dry-run，而不是真实 `.npz` 构建

## 4. 建议执行顺序

1. 先补 `GEMINI_API_KEY` 或在 manifest 中配置 `api_key_file`
2. 先跑小线:
   - `python -m dcas.scripts.build_research_dataset_v4 --manifest configs/dataset/research_dataset_v4_routeA_small.json --stages embeddings --embedding_targets gemini`
3. 再跑主线:
   - `python -m dcas.scripts.build_research_dataset_v4 --manifest configs/dataset/research_dataset_v4_main_from_v3.json --stages embeddings --embedding_targets gemini`
4. tracks 落地后，依次执行:
   - `python -m dcas.scripts.run_train_from_json --config configs/train/train_v4_routeA_small_gemini_stage3.run.json`
   - `python -m dcas.scripts.run_recommender_benchmarks --config configs/benchmark/recommender_benchmark_v4_routeA_small_gemini_stage3_lambdamart.run.json`
   - `python -m dcas.scripts.run_train_from_json --config configs/train/train_v4_main_gemini_stage3.run.json`
   - `python -m dcas.scripts.run_recommender_benchmarks --config configs/benchmark/recommender_benchmark_v4_main_gemini_stage3_lambdamart.run.json`

## 5. 论文写作边界

当前可以写:

- 系统设计上支持 `CultureMERT` 与 `Gemini` 两类 backbone
- `Gemini` 线的工程协议已经与 `CultureMERT stage3` 对齐

当前还不能写:

- 任意 backbone 下都已经完成对称实证
- `Gemini` 在 `V4` 上已经重现实验优势
