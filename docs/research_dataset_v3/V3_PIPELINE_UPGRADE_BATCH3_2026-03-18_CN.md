# V3 流程升级 Batch 3

## 本批目标

第三批的目标是把 `CultureMERT` 主线真正跑完一次：

1. 重建 `mw3` 多窗口 embedding
2. 对齐 `metadata / interactions / pseudo constraints`
3. 训练 `source-aware + pseudo-PAL` 的 pre-PAL DCAS
4. 跑开放式 benchmark

## 实际产物

### 1. Multi-window tracks

文件：
- [tracks_culturemert_v3_main_mw3.npz](E:/Desktop/Echo/storage/public/research_dataset_v3/tracks_culturemert_v3_main_mw3.npz)
- [tracks_culturemert_v3_main_mw3.npz.manifest.json](E:/Desktop/Echo/storage/public/research_dataset_v3/tracks_culturemert_v3_main_mw3.npz.manifest.json)

运行命令：

```powershell
E:\venvs\echo-gpu\Scripts\python.exe `
  -m dcas.scripts.run_culturemert_embedding_build `
  --config E:\Desktop\Echo\configs\embedding\culturemert_v3_main_multiwindow.run.json
```

结果：
- 原始主表：`1122`
- 成功 embedding：`1106`
- 失败：`16`
- 维度：`768`
- `source_dataset` 已写入 tracks

失败条目分布：
- `india`: `3`
- `germany`: `4`
- `france`: `3`
- `italy`: `1`
- `great_britain`: `2`
- `russia`: `3`

说明：
- 失败并不是整域崩掉，而是少量音频在多窗口阶段出现空张量错误
- 我已经补了“单窗口失败可跳过”的容错到：
  - [culturemert.py](E:/Desktop/Echo/dcas/embeddings/culturemert.py)
  - [gemini_embedding2.py](E:/Desktop/Echo/dcas/embeddings/gemini_embedding2.py)
- 但这次 full run 是在补丁前启动的，所以当前 `mw3` 主产物仍保留 `1106` 条结果

### 2. 对齐后的资产

文件：
- [metadata_v3_main_harmonized_mw3.csv](E:/Desktop/Echo/storage/public/research_dataset_v3/metadata_v3_main_harmonized_mw3.csv)
- [metadata_v3_main_harmonized_mw3.csv.align_report.json](E:/Desktop/Echo/storage/public/research_dataset_v3/metadata_v3_main_harmonized_mw3.csv.align_report.json)
- [interactions_v3_main_mixed_mw3.csv](E:/Desktop/Echo/storage/public/research_dataset_v3/interactions_v3_main_mixed_mw3.csv)
- [pseudo_constraints_v1_mw3.jsonl](E:/Desktop/Echo/storage/pal/v3_main_prepal/pseudo_constraints_v1_mw3.jsonl)

运行命令：

```powershell
E:\venvs\echo-gpu\Scripts\python.exe `
  -m dcas.scripts.align_assets_to_tracks `
  --tracks E:\Desktop\Echo\storage\public\research_dataset_v3\tracks_culturemert_v3_main_mw3.npz `
  --metadata_in E:\Desktop\Echo\storage\public\research_dataset_v3\metadata_v3_main_harmonized.csv `
  --metadata_out E:\Desktop\Echo\storage\public\research_dataset_v3\metadata_v3_main_harmonized_mw3.csv `
  --interactions_in E:\Desktop\Echo\storage\public\research_dataset_v3\interactions_v3_main_mixed.csv `
  --interactions_out E:\Desktop\Echo\storage\public\research_dataset_v3\interactions_v3_main_mixed_mw3.csv `
  --constraints_in E:\Desktop\Echo\storage\pal\v3_main_prepal\pseudo_constraints_v1.jsonl `
  --constraints_out E:\Desktop\Echo\storage\pal\v3_main_prepal\pseudo_constraints_v1_mw3.jsonl
```

结果：
- metadata：`1122 -> 1106`
- interactions：`8413 -> 8273`
- pseudo constraints：`1200 -> 1192`

这一步很重要，因为它把训练和 benchmark 的输入严格收敛到了真实可用的 `mw3` track 集合。

### 3. Source-aware pre-PAL model

文件：
- [dcas_full_v3_main_culturemert_open_prepal.pt](E:/Desktop/Echo/storage/models/dcas_full_v3_main_culturemert_open_prepal.pt)

运行命令：

```powershell
E:\venvs\echo-gpu\Scripts\python.exe `
  -m dcas.scripts.run_train_from_json `
  --config E:\Desktop\Echo\configs\train\train_v3_culturemert_prepal_source.run.json
```

结果：
- `10` 个 culture
- `8` 个 source
- `1192` 条伪约束参与训练
- `10` epoch 完成

训练尾部 loss：
- epoch `7`: `1.6707`
- epoch `8`: `1.6335`
- epoch `9`: `1.6332`

说明：
- 这一版已经不是单纯的 DCAS baseline
- 它同时带了：
  - `pseudo constraints`
  - `source_balanced_batch`
  - `source adversarial head`

### 4. Open benchmark

文件：
- [benchmark_summary.json](E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert_open_prepal/benchmark_summary.json)
- [benchmark_table.md](E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert_open_prepal/benchmark_table.md)

运行命令：

```powershell
E:\venvs\echo-gpu\Scripts\python.exe `
  -m dcas.scripts.run_recommender_benchmarks `
  --config E:\Desktop\Echo\configs\benchmark\recommender_benchmark_v3_culturemert_open.run.json
```

## 结果总览

### Raw baselines

- `popularity`
  - `serendipity = 0.4543`
  - `KL = 2.0709`
  - `minority = 0.0000`
  - `target_prob = 0.1842`
- `cosine`
  - `serendipity = 0.5542`
  - `KL = 2.1760`
  - `minority = 0.2265`
  - `target_prob = 0.1645`
- `knn`
  - `serendipity = 0.5705`
  - `KL = 2.1846`
  - `minority = 0.2264`
  - `target_prob = 0.1624`
- `shallow_mlp`
  - `serendipity = 0.4813`
  - `KL = 2.1310`
  - `minority = 0.1674`
  - `target_prob = 0.1725`
- `hybrid`
  - `serendipity = 0.5369`
  - `KL = 2.1638`
  - `minority = 0.0700`
  - `target_prob = 0.1663`

### DCAS 系列

- `dcas_full_ot`
  - `serendipity = 0.8332`
  - `KL = 2.0699`
  - `minority = 0.2370`
  - `target_prob = 0.1850`
- `dcas_full_knn`
  - `serendipity = 0.8342`
  - `KL = 2.0697`
  - `minority = 0.2351`
  - `target_prob = 0.1852`
- `dcas_full_ot_open`
  - `serendipity = 0.2572`
  - `KL = 2.2957`
  - `minority = 0.4444`
  - `target_prob = 0.1334`
- `dcas_full_knn_open`
  - `serendipity = 0.2452`
  - `KL = 2.2962`
  - `minority = 0.4464`
  - `target_prob = 0.1334`

## 关键解读

### 1. 主胜线已经出来了

当前最值得继续推进的不是 `open rerank`，而是：

- `CultureMERT mw3`
- `pseudo-PAL`
- `source-aware DCAS`
- `target-only dcas_full_ot / dcas_full_knn`

因为在这一版 benchmark 里，`dcas_full_ot` 和 `dcas_full_knn` 已经对 raw baselines 展现出完整优势。

### 2. 相对最强 raw baseline 的提升

以 `dcas_full_ot` 为主线，相对每个维度上的最强 raw baseline：

- `serendipity`
  - 最强 raw：`knn = 0.5705`
  - `dcas_full_ot = 0.8332`
  - 提升：`+46.06%`
- `cultural_calibration_kl`
  - 最强 raw：`popularity = 2.07095`
  - `dcas_full_ot = 2.06989`
  - 改善：`+0.05%`
- `minority_exposure_at_k`
  - 最强 raw：`cosine = 0.2265`
  - `dcas_full_ot = 0.2370`
  - 提升：`+4.62%`
- `target_culture_prob_mean`
  - 最强 raw：`popularity = 0.18423`
  - `dcas_full_ot = 0.18496`
  - 提升：`+0.40%`

这意味着：
- `DCAS` 在这轮里不再只是“一个指标看起来好”
- 而是已经在四个核心指标上同时压过了最强 raw baseline

### 3. Open 两阶段推荐的现状

`dcas_full_ot_open` 的特征很鲜明：

- `minority_exposure_at_k` 相对 `dcas_full_ot` 提升：`+87.49%`
- 但同时：
  - `serendipity` 降了 `-69.13%`
  - `KL` 变差 `-10.91%`
  - `target_prob` 降了 `-27.88%`

所以当前的结论不是“开放式两阶段推荐失败”，而是：

- 它已经明显拉高了长尾曝光
- 但现在的 rerank 权重和候选策略还太激进
- 还没有找到能同时守住 `serendipity + calibration + target affinity` 的平衡点

也就是说，`open recommendation` 现在更像一条需要继续调权和结构优化的分支，而不是当前的主胜线。

## 当前结论

如果只看第三批的实际结果，我会给出非常明确的判断：

- 这次真正有效的改进不是 `open rerank`
- 而是：
  - `多窗口 embedding`
  - `伪 PAL 约束`
  - `source-aware 训练`
  - `mixed interactions`

在它们组合起来之后，`CultureMERT + DCAS` 已经出现了很像论文主结果的形态：

- 高 `serendipity`
- 不输的 `KL`
- 更好的 `minority exposure`
- 更高的 `target culture affinity`

## 下一步建议

第四批建议这样走：

1. 保持 `dcas_full_ot / dcas_full_knn` 作为主线
2. 对 `open` 分支单独做小规模调权实验，不要立刻拿它当主模型
3. 再决定是否重跑 `Gemini mw3`
4. 在当前这条 `CultureMERT` 主胜线上开始准备真人 PAL

如果要给一句最简洁的总结：

`第三批已经把 CultureMERT 主线从“有潜力”推进到了“结果开始像样”。`
