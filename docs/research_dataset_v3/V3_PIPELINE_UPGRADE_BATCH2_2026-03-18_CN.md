# V3 流程升级 Batch 2

## 本批目标

第二批的目标是把第一批的“临时能力”变成正式可运行的实验资产：

- 把 harmonized metadata / mixed interactions / pseudo constraints 固化到正式路径
- 给 Gemini 和 CultureMERT 都补齐配置化 runner
- 给多窗口 embedding、预 PAL 训练、开放式 benchmark 补齐 run config
- 做一轮配置级烟测，确认第二批不是纸面方案

## 正式化产物

### 1. Harmonized metadata

文件：
- [metadata_v3_main_harmonized.csv](E:/Desktop/Echo/storage/public/research_dataset_v3/metadata_v3_main_harmonized.csv)
- [metadata_v3_main_harmonized.report.json](E:/Desktop/Echo/storage/public/research_dataset_v3/metadata_v3_main_harmonized.report.json)

运行命令：

```powershell
python -m dcas.scripts.harmonize_v3_metadata `
  --metadata E:\Desktop\Echo\storage\public\research_dataset_v3\metadata_v3_main.csv `
  --out E:\Desktop\Echo\storage\public\research_dataset_v3\metadata_v3_main_harmonized.csv
```

结果：
- `1122` 条
- `coarse_label` 已写入正式表

### 2. Mixed interactions

文件：
- [interactions_v3_main_mixed.csv](E:/Desktop/Echo/storage/public/research_dataset_v3/interactions_v3_main_mixed.csv)

运行命令：

```powershell
python -m dcas.scripts.synthesize_interactions `
  --metadata E:\Desktop\Echo\storage\public\research_dataset_v3\metadata_v3_main_harmonized.csv `
  --out E:\Desktop\Echo\storage\public\research_dataset_v3\interactions_v3_main_mixed.csv `
  --users_per_culture 24 `
  --tracks_per_user 40 `
  --genre_column coarse_label `
  --mode mixed_culture `
  --secondary_cultures 2 `
  --home_share 0.6 `
  --seed 42
```

结果：
- `240` 个用户
- `8413` 条交互

### 3. Pseudo constraints

文件：
- [pseudo_constraints_v1.jsonl](E:/Desktop/Echo/storage/pal/v3_main_prepal/pseudo_constraints_v1.jsonl)
- [pseudo_constraints_v1.jsonl.report.json](E:/Desktop/Echo/storage/pal/v3_main_prepal/pseudo_constraints_v1.jsonl.report.json)

运行命令：

```powershell
python -m dcas.scripts.build_pseudo_pal_constraints `
  --metadata E:\Desktop\Echo\storage\public\research_dataset_v3\metadata_v3_main_harmonized.csv `
  --out E:\Desktop\Echo\storage\pal\v3_main_prepal\pseudo_constraints_v1.jsonl `
  --n_positive 600 `
  --n_negative 600 `
  --per_track_cap 4
```

结果：
- `1200` 条伪约束
- 正对中：
  - `cross_culture_ratio = 0.7333`
  - `cross_source_ratio = 0.7333`

## 新增 runner

### 1. CultureMERT config runner

文件：
- [run_culturemert_embedding_build.py](E:/Desktop/Echo/dcas/scripts/run_culturemert_embedding_build.py)

作用：
- 让 CultureMERT 也能像 Gemini 一样用 JSON 配置启动 embedding build

### 2. Train config runner

文件：
- [run_train_from_json.py](E:/Desktop/Echo/dcas/scripts/run_train_from_json.py)

作用：
- 用 JSON 直接组织预 PAL / source-aware 训练
- 把 `constraints / lambda_source / source_balanced_batch` 都纳入可配置项

## 新增配置

### Embedding

- [culturemert_v3_main_multiwindow.run.json](E:/Desktop/Echo/configs/embedding/culturemert_v3_main_multiwindow.run.json)
- [gemini_embedding2_v3_main_multiwindow.example.json](E:/Desktop/Echo/configs/embedding/gemini_embedding2_v3_main_multiwindow.example.json)
- [gemini_embedding2_v3_main_multiwindow.local.example.json](E:/Desktop/Echo/configs/embedding/gemini_embedding2_v3_main_multiwindow.local.example.json)

目标：
- 把 V3 主表升级成 `3-window uniform + mean aggregate`

预期输出：
- `tracks_culturemert_v3_main_mw3.npz`
- `tracks_gemini_embedding2_v3_main_mw3.npz`

### Train

- [train_v3_culturemert_prepal_source.run.json](E:/Desktop/Echo/configs/train/train_v3_culturemert_prepal_source.run.json)
- [train_v3_gemini_prepal_source.run.json](E:/Desktop/Echo/configs/train/train_v3_gemini_prepal_source.run.json)

目标：
- 训练 `pseudo constraints + source-aware` 的 pre-PAL 模型

关键项：
- `lambda_constraints = 0.15`
- `lambda_source = 0.1`
- `source_balanced_batch = true`

### Benchmark

- [recommender_benchmark_v3_culturemert_open.run.json](E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v3_culturemert_open.run.json)
- [recommender_benchmark_v3_gemini_open.run.json](E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v3_gemini_open.run.json)

特点：
- 默认使用 harmonized metadata
- 默认使用 mixed interactions
- 默认加入 `dcas_full_ot_open / dcas_full_knn_open`

## 配置级烟测

### 1. CultureMERT multi-window runner

环境：
- 使用 `E:\venvs\echo-gpu\Scripts\python.exe`

结果：
- `3` 条 smoke 样本成功生成
- 输出：[tracks_culturemert_v3_mw3_smoke.npz](E:/Desktop/Echo/tmp/tracks_culturemert_v3_mw3_smoke.npz)

### 2. Gemini multi-window dry-run

结果：
- `2` 条 smoke 样本成功准备
- dry-run 报出了 `payload=2880132 bytes`
- 输出 manifest：
  - [tracks_gemini_mw3_smoke_dryrun.npz.manifest.json](E:/Desktop/Echo/tmp/tracks_gemini_mw3_smoke_dryrun.npz.manifest.json)

顺手修复的问题：
- `build_tracks_with_gemini.py` 里 dry-run 还在读旧字段 `payload_bytes`
- 现在已经兼容新字段 `total_payload_bytes`

### 3. Train runner

结果：
- `run_train_from_json.py` smoke 成功
- 输出 checkpoint：[dcas_train_from_json_smoke.pt](E:/Desktop/Echo/tmp/dcas_train_from_json_smoke.pt)

### 4. Open benchmark smoke

使用资产：
- 旧的 `tracks_culturemert_v3_main.npz`
- 旧的 `dcas_full_v3_main_culturemert.pt`
- 新的 `metadata_v3_main_harmonized.csv`
- 新的 `interactions_v3_main_mixed.csv`

结果摘要：
- `cosine`
  - `serendipity_mean = 0.6146`
  - `cultural_calibration_kl_mean = 2.2332`
  - `minority_exposure_at_k_mean = 0.2618`
  - `target_culture_prob_mean = 0.1529`
- `dcas_full_ot_open`
  - `serendipity_mean = 0.3227`
  - `cultural_calibration_kl_mean = 2.3641`
  - `minority_exposure_at_k_mean = 0.3156`
  - `target_culture_prob_mean = 0.1265`

解释：
- 这不是失败，而是一个非常有价值的早期信号
- 说明“开放式两阶段推荐”光加上去还不够，必须和：
  - 多窗口 embedding
  - 伪约束训练
  - source-aware DCAS
  一起重跑，才能公平判断

## 环境提醒

第二批已经确认：
- `Anaconda` 里的当前 Python 环境不适合跑 CultureMERT 相关 runner
- 正确环境应使用：

```powershell
E:\venvs\echo-gpu\Scripts\python.exe
```

尤其是：
- `run_culturemert_embedding_build.py`
- `run_gemini_embedding_build.py`
- `run_train_from_json.py`

## 当前判断

到这里为止，第二批已经把实验资产准备齐了：

- 正式 metadata
- 正式 mixed interactions
- 正式 pseudo constraints
- 双 backbone 的多窗口 embedding 配置
- 预 PAL source-aware 训练配置
- 开放式 benchmark 配置

也就是说，下一步已经可以直接进入真正的重跑阶段，而不是继续补基础设施。

## 第三批建议

建议按这个顺序继续：

1. 用多窗口配置重建 `tracks_culturemert_v3_main_mw3.npz`
2. 用多窗口配置重建 `tracks_gemini_embedding2_v3_main_mw3.npz`
3. 跑 `train_v3_culturemert_prepal_source.run.json`
4. 跑 `recommender_benchmark_v3_culturemert_open.run.json`
5. 再决定 Gemini 线是否也全量重跑
