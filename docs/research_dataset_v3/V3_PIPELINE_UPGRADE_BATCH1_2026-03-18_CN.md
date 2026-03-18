# V3 流程升级 Batch 1

## 本批目标

这一批先把最值得动的前四项和第六项真正落到代码里，并为第五项 `source confound control` 铺出可训练入口：

- 开放式两阶段推荐
- 多窗口 embedding
- 伪 PAL 约束
- 混合文化合成交互
- metadata 统一层
- source-aware training 的第一版支撑

## 已完成改动

### 1. 开放式两阶段推荐

文件：
- [recommender.py](E:/Desktop/Echo/dcas/recommender.py)
- [run_recommender_benchmarks.py](E:/Desktop/Echo/dcas/scripts/run_recommender_benchmarks.py)

新增方法：
- `recommend_open_ot`
- `recommend_open_knn`

当前实现逻辑：
1. 第一阶段从全库召回，不再先按目标文化硬过滤
2. 第二阶段在召回池里按多目标分数 rerank

当前分数结构：
- `relevance_za`
- `novelty_zs`
- `target_culture_affinity_zs`
- `minority_boost`
- `diversity_penalty`

说明：
- 现在 benchmark runner 已支持 `ot_open` 和 `knn_open`
- 权重已做成方法级配置项，便于后续调参

### 2. 多窗口 embedding

文件：
- [culturemert.py](E:/Desktop/Echo/dcas/embeddings/culturemert.py)
- [gemini_embedding2.py](E:/Desktop/Echo/dcas/embeddings/gemini_embedding2.py)
- [build_tracks_from_audio.py](E:/Desktop/Echo/dcas/scripts/build_tracks_from_audio.py)
- [build_tracks_with_gemini.py](E:/Desktop/Echo/dcas/scripts/build_tracks_with_gemini.py)
- [run_gemini_embedding_build.py](E:/Desktop/Echo/dcas/scripts/run_gemini_embedding_build.py)

新增配置项：
- `window_count`
- `window_strategy`
- `window_aggregate`

当前支持：
- 单窗口
- 均匀多窗口采样
- 多窗口均值聚合

说明：
- 这已经足够让 `india` 和长时长传统曲目从“只看开头 30 秒”升级成“多段取样后聚合”
- 当前聚合先落成 `mean`，后续如果要试 `attention / concat / learned pooling` 可以继续往上叠

### 3. 伪 PAL 约束

文件：
- [harmonize_v3_metadata.py](E:/Desktop/Echo/dcas/scripts/harmonize_v3_metadata.py)
- [build_pseudo_pal_constraints.py](E:/Desktop/Echo/dcas/scripts/build_pseudo_pal_constraints.py)

新增 metadata 统一字段：
- `coarse_label`
- `is_instrumental`

伪约束生成依据：
- `coarse_label`
- `era`
- `instrument_family`
- `language`
- `substyle`
- `source_dataset`

策略：
- 先构造高置信正对和高置信负对
- 正对会优先鼓励一定比例的 `cross_culture / cross_source`
- 用来做真人 PAL 之前的“基础常识约束预热”

### 4. 混合文化合成交互

文件：
- [synthesize_interactions.py](E:/Desktop/Echo/dcas/scripts/synthesize_interactions.py)
- [run_recommender_benchmarks.py](E:/Desktop/Echo/dcas/scripts/run_recommender_benchmarks.py)

新增模式：
- `single_culture`
- `mixed_culture`

混合文化用户画像支持：
- `home culture`
- `1-2` 个 secondary cultures
- `home_share`

说明：
- benchmark 的自动合成交互入口已经可以把这些参数从 config 往下传
- 后续可以继续把“新颖度偏好 / 长尾接受度 / 文化转移方向”再做细

### 5. Source confound control 第一版

文件：
- [npz_tracks.py](E:/Desktop/Echo/dcas/data/npz_tracks.py)
- [torch_dataset.py](E:/Desktop/Echo/dcas/data/torch_dataset.py)
- [batch.py](E:/Desktop/Echo/dcas/data/batch.py)
- [utils.py](E:/Desktop/Echo/dcas/utils.py)
- [dcas_vae.py](E:/Desktop/Echo/dcas/models/dcas_vae.py)
- [pipelines.py](E:/Desktop/Echo/dcas/pipelines.py)
- [train.py](E:/Desktop/Echo/dcas/cli/train.py)
- [build_tracks_from_audio.py](E:/Desktop/Echo/dcas/scripts/build_tracks_from_audio.py)
- [build_tracks_with_gemini.py](E:/Desktop/Echo/dcas/scripts/build_tracks_with_gemini.py)

已接入内容：
- `tracks.npz` 可选携带 `source_dataset`
- `TrackDataset` 可编码 source label
- `Batch` 已支持 `source_label`
- DCAS 训练已支持：
  - `--lambda_source`
  - `--source_balanced_batch`
- 模型已加入 source adversarial head 的第一版实现

说明：
- 这一版 source adversary 放在 `za` 上，目标是先减轻推荐相关 latent 被数据来源污染
- 真正的效果要等重新构建带 `source_dataset` 的 tracks 并重跑训练后再看

## 真实烟测

### 1. Metadata 统一层

运行：

```powershell
python -m dcas.scripts.harmonize_v3_metadata `
  --metadata E:\Desktop\Echo\storage\public\research_dataset_v3\metadata_v3_main.csv `
  --out E:\Desktop\Echo\tmp\v3_harmonized_metadata.csv
```

结果：
- `1122` 条
- 输出：[v3_harmonized_metadata.csv](E:/Desktop/Echo/tmp/v3_harmonized_metadata.csv)
- 报告：[v3_harmonized_metadata.report.json](E:/Desktop/Echo/tmp/v3_harmonized_metadata.report.json)

粗标签分布：
- `modern_song 355`
- `soundtrack_classical 144`
- `traditional_instrumental 120`
- `art_music 108`
- `jazz_blues 100`
- `folk_acoustic 94`
- `modern_pop_song 50`
- `instrumental_ambient 51`
- `traditional_vocal 30`
- `unknown 70`

### 2. 伪 PAL 约束

运行：

```powershell
python -m dcas.scripts.build_pseudo_pal_constraints `
  --metadata E:\Desktop\Echo\tmp\v3_harmonized_metadata.csv `
  --out E:\Desktop\Echo\tmp\v3_pseudo_constraints.jsonl `
  --n_positive 600 `
  --n_negative 600 `
  --per_track_cap 4
```

结果：
- 候选正对：`97,145`
- 候选负对：`390,783`
- 最终约束：`1,200`
- 输出：[v3_pseudo_constraints.jsonl](E:/Desktop/Echo/tmp/v3_pseudo_constraints.jsonl)
- 报告：[v3_pseudo_constraints.jsonl.report.json](E:/Desktop/Echo/tmp/v3_pseudo_constraints.jsonl.report.json)

当前选中正对里：
- `cross_culture_ratio = 0.7333`
- `cross_source_ratio = 0.7333`

### 3. 混合文化交互

运行：

```powershell
python -m dcas.scripts.synthesize_interactions `
  --metadata E:\Desktop\Echo\tmp\v3_harmonized_metadata.csv `
  --out E:\Desktop\Echo\tmp\v3_interactions_mixed.csv `
  --users_per_culture 8 `
  --tracks_per_user 30 `
  --genre_column coarse_label `
  --mode mixed_culture `
  --secondary_cultures 2 `
  --home_share 0.6 `
  --seed 42
```

结果：
- `80` 个用户
- `2246` 条交互
- 输出：[v3_interactions_mixed.csv](E:/Desktop/Echo/tmp/v3_interactions_mixed.csv)

### 4. Source-aware 训练烟测

做法：
- 从现有 `tracks_culturemert_v3_main.npz` 抽 `64` 条
- 补上 `source_dataset`
- 跑 `1 epoch` 训练

结果：
- 训练成功
- 输出 checkpoint：[dcas_source_smoke.pt](E:/Desktop/Echo/tmp/dcas_source_smoke.pt)

## 当前判断

这批最重要的成果不是“指标已经优化完”，而是：

- 现在已经有了真正的开放式推荐骨架
- 已经能做多窗口 embedding
- 已经能先跑伪 PAL，再做真人 PAL
- 已经能模拟更像跨文化用户的交互
- 已经有 source-aware 训练入口

换句话说，下一批工作可以直接进入：
- benchmark 重跑与调权
- 真实多窗口 embedding 重建
- 带伪约束的 baseline 训练
- source-aware DCAS 的正式对比实验

## 下一批建议

建议按这个顺序继续：

1. 重新构建带 `source_dataset` 的 `tracks_gemini` 和 `tracks_culturemert`
2. 用 harmonized metadata 生成 mixed interactions
3. 用 pseudo constraints 训练 `pre-PAL` 版 DCAS
4. 在 benchmark 里加入 `ot_open / knn_open`
5. 调整 rerank 权重，重点看：
   - `serendipity`
   - `minority_exposure_at_k`
   - `cultural_calibration_kl`
   - `target_culture_prob_mean`
