# V4 CultureMERT 构建与验证记录

日期：2026-03-20

## 1. 本轮执行内容

本轮在 `V4` 数据骨架已经完成 `merge / harmonize / interactions / audit` 的前提下，继续完成了 `CultureMERT` 这条 backbone 的真实构建与验证：

- `routeA_small`
  - manifest: `configs/dataset/research_dataset_v4_routeA_small.json`
  - command:
    ```bash
    python -m dcas.scripts.build_research_dataset_v4 \
      --manifest configs/dataset/research_dataset_v4_routeA_small.json \
      --stages embeddings \
      --embedding_targets culturemert
    ```
- `main`
  - manifest: `configs/dataset/research_dataset_v4_main_from_v3.json`
  - command:
    ```bash
    python -m dcas.scripts.build_research_dataset_v4 \
      --manifest configs/dataset/research_dataset_v4_main_from_v3.json \
      --stages embeddings \
      --embedding_targets culturemert
    ```

两条线都使用同一组 `CultureMERT` 提取设置：

- `model_id = ntua-slp/CultureMERT-95M`
- `pooling = mean`
- `layer_indices = [-4, -3, -2, -1]`
- `window_count = 3`
- `window_strategy = uniform`
- `window_aggregate = mean`
- `max_seconds = 30.0`

## 2. 关键产物

### 2.1 routeA_small

- tracks:
  - `storage/public/research_dataset_v4/routeA_small/tracks_culturemert_mw3.npz`
- manifest:
  - `storage/public/research_dataset_v4/routeA_small/tracks_culturemert_mw3.npz.manifest.json`
- validation:
  - `reports/datasets/research_dataset_v4/routeA_small/validate_culturemert/report.json`
- metadata audit:
  - `reports/datasets/research_dataset_v4/routeA_small/dataset_profile.json`
- source confound:
  - `reports/datasets/research_dataset_v4/routeA_small/source_confound_report.json`

### 2.2 main

- tracks:
  - `storage/public/research_dataset_v4/main/tracks_culturemert_mw3.npz`
- manifest:
  - `storage/public/research_dataset_v4/main/tracks_culturemert_mw3.npz.manifest.json`
- validation:
  - `reports/datasets/research_dataset_v4/main/validate_culturemert/report.json`
- metadata audit:
  - `reports/datasets/research_dataset_v4/main/dataset_profile.json`
- source confound:
  - `reports/datasets/research_dataset_v4/main/source_confound_report.json`

## 3. 结果摘要

| 数据线 | tracks | cultures | sources | embedding errors | validate status | interactions | track coverage | source predictability from culture |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| `V4 routeA_small` | 640 | 4 | 4 | 0 | pass | 3840 | 0.995313 | 1.000000 |
| `V4 main` | 1122 | 10 | 8 | 0 | pass | 9600 | 1.000000 | 0.911765 |

补充字段覆盖情况：

| 数据线 | `era` coverage | `region` coverage | `artist` coverage |
|---|---:|---:|---:|
| `V4 routeA_small` | 0.000000 | 1.000000 | 0.000000 |
| `V4 main` | 1.000000 | 1.000000 | 0.647950 |

## 4. 解释

### 4.1 这轮最重要的正面结果

1. `CultureMERT` 在 `V4 routeA_small` 和 `V4 main` 两条线上都完成了真实全量构建。
2. 两条线都没有出现 embedding 丢样本：
   - `routeA_small`: `n_tracks = 640`, `n_errors = 0`
   - `main`: `n_tracks = 1122`, `n_errors = 0`
3. 两条线的 `validate_dataset` 都是 `pass`。
4. `routeA_small` 和 `main` 都已经具备进入下一步 benchmark 的稳定输入条件。

### 4.2 这轮仍然暴露出的科学性问题

1. `source confound` 没有因为 V4 schema 清洗而自动消失。
   - `V4 routeA_small` 仍然是完全绑定：`weighted_source_predictability_from_culture = 1.0`
   - `V4 main` 虽然比 `routeA_small` 好，但仍然很高：`0.911765`
2. `routeA_small` 的 `era` 仍是 `0.0` 覆盖，不能把它当成一条字段完整的主数据线。
3. 两条线都没有 `affect_label`，这不阻止推荐实验，但会限制 affect 相关分析。

### 4.3 目前可以怎么写

当前可以较稳地写成：

- `V4` 已经完成统一 schema、metadata audit、interaction protocol 和 `CultureMERT` 全量构建。
- `CultureMERT` 在 `V4` 主线和小线上的构建稳定性已经被验证。
- 但 `V4` 仍存在明显的 `source confound`，尤其 `routeA_small` 只能作为小型公开来源 sanity-check，而不适合作为“弱 source bias”的补充证据。

不宜过度写成：

- `V4` 已经充分解决跨文化数据偏差
- `routeA_small` 可以与 `main` 同等强度地支撑主结论

## 5. 下一步优先级

1. 基于当前 `tracks_culturemert_mw3.npz` 接出 `V4` 的训练与 benchmark 配置。
2. 优先推进 `V4 main` 的 benchmark，因为它已是当前更完整的多文化主线。
3. `routeA_small` 保留为公开来源小线，但在论文中应明确承认其 `source confound = 1.0`。
4. 进入 `Gemini` 线前，先决定是做 `dry_run -> 小线真跑 -> 主线真跑`，还是直接在 `routeA_small` 上做第一轮真实 API 运行。
