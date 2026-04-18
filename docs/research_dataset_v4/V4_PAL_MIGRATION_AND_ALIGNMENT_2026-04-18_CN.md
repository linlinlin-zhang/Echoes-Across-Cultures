# V4 真人 PAL 迁移与对齐记录

日期：`2026-04-18`

## 1. 这份记录解决什么问题

这份文档专门回答一个这两天已经反复出现、而且很容易把论文叙事带偏的问题：

- 这次真人 `PAL` 标注到底是基于哪个前端包做的？
- 它和当前正式的 `V4 main + CultureMERT stage3` 主实验线是什么关系？
- 这批真人 `PAL` 标注能不能直接迁移到当前 `V4` 主线？
- 迁移后，结果到底是“真的改善了系统”，还是只是把流程跑通了？

结论先写在前面：

1. 这次真人 `PAL` 标注实际是基于 `storage/pal/v4_main_annotation/` 这套离线标注前端完成的。
2. 这套前端包不是当前仓库里最规范的正式 `V4 real PAL bundle`，但它对应的标注内容可以映射到当前 `V4 main` 主线。
3. `workspace_assets/pal_exports/pal_v4_main_annotation_human_export_200pairs.csv` 中的 `200` 条标注任务都能在当前 `V4 main` 里找到对应曲目，两端涉及的 `335` 个唯一 `track_id` 也全部存在于 `V4 main` 元数据中。
4. 因此，这批真人 `PAL` 结果在技术上是可以迁移的；但迁移后是否能带来论文主张层面的增益，还需要看训练与 rerank 的具体配置。

## 2. 这次 PAL 的资产关系

### 2.1 实际使用过的离线标注包

- 前端包目录：`storage/pal/v4_main_annotation/`
- 实际导出的真人标注：`workspace_assets/pal_exports/pal_v4_main_annotation_human_export_200pairs.csv`

这条线的关键点不是“目录名字像不像正式 V4”，而是：

- 它确实承载了这次真人标注的任务集合；
- 它和当前正式 V4 PAL 包不是完全同一个目录；
- 但它挑出来的曲目对，仍然来自当前主实验语境里可追溯的 V4 曲目空间。

### 2.2 当前正式的 V4 真人 PAL 主线

- 正式准备包目录：`storage/pal/v4_main_culturemert_real/`
- 准备配置：`configs/pal/pal_v4_main_culturemert_prepare.run.json`
- 正式回灌配置：`configs/pal/pal_v4_main_culturemert_real.run.json`

这条线才是仓库里原本预留给论文主叙事的“正式 V4 真人 PAL workflow”。

### 2.3 为什么会发生混淆

混淆的根本原因不是数据真的完全错乱，而是仓库里同时存在：

- 一个实际被用于离线标注的 `v4_main_annotation` 包；
- 一个更规范、面向正式流程的 `v4_main_culturemert_real` 包；
- 以及更早的 `V3` PAL 文档与 simulated PAL 叙事。

如果不专门留档，后面很容易出现两种误判：

1. 误以为这次真人 `PAL` 还停留在旧版 `V3`；
2. 误以为 `v4_main_annotation` 本身就是当前正式主实验的唯一 PAL 包。

## 3. 一致性核查结果

### 3.1 `v4_main_annotation` 目录内部并不完全一致

这也是为什么它不能直接被当成“正式主线包”来写论文：

- `storage/pal/v4_main_annotation/tasks.jsonl`：`200` 条任务
- `storage/pal/v4_main_annotation/pal_tasks_embedded.js`：`200` 条任务
- `storage/pal/v4_main_annotation/pal_tasks.json`：`120` 条任务
- `storage/pal/v4_main_annotation/annotation_sheet.csv`：`60` 条标注表行

这说明 `v4_main_annotation` 更像一个被迭代过、用于实际收集标注的工作目录，而不是一个已经彻底清洗好的正式发布包。

### 3.2 真人标注文件本身是完整的

- `workspace_assets/pal_exports/pal_v4_main_annotation_human_export_200pairs.csv`：`200` 行标注
- 其中涉及的唯一曲目数：`335`
- 这 `335` 个唯一 `track_id` 全部都存在于 `storage/public/research_dataset_v4/main/metadata_release.csv`
- 因而，`workspace_assets/pal_exports/pal_v4_main_annotation_human_export_200pairs.csv` 中的 `200` 个曲目对都可以映射到当前 `V4 main`

这部分结论很关键，因为它说明：

- 当前问题不是“这批 PAL 完全不能用”；
- 而是“这批 PAL 需要从非正式标注目录，迁移到正式 V4 主实验协议里使用”。

## 4. 这次我做了什么迁移与对齐

### 4.1 先做了一个直接迁移的 smoke test

配置：

- `configs/pal/pal_v4_main_culturemert_real_from_v4_main_annotation.run.json`

这个版本的目的只有一个：验证 `workspace_assets/pal_exports/pal_v4_main_annotation_human_export_200pairs.csv` 能不能被当前 PAL 平台直接吃进去，并且完成一次 `baseline -> constraints -> retrain -> evaluate` 的闭环。

这个 smoke test 成功说明：

- 标注文件格式本身没有问题；
- 任务对在 V4 主线里可映射；
- PAL 平台对这批真人标注能够正常构建约束并回灌。

但这个版本不适合直接用于论文，因为它当时还没有完全对齐当前 `V4 stage3 + calibrated rerank` 的正式 benchmark 协议。

### 4.2 然后做了 benchmark-aligned 的迁移版本

配置：

- `configs/pal/pal_v4_main_culturemert_real_from_v4_main_annotation_stage3.run.json`

这一版做了几个关键对齐：

1. 使用当前正式 `V4 main` 的 `tracks / metadata / interactions`
2. 从当前主线 checkpoint `storage/models/dcas_full_v4_main_culturemert_stage3.pt` warm-start
3. 保留 ranking signal，而不是把 PAL 训练变成脱离主 benchmark 的单独重训
4. 评估时使用 `ot_calibrated`，而不是旧式 raw `ot`
5. 输出显式的约束报告，避免 duplicate / conflict 被静默吞掉

## 5. 约束构建后的真实情况

对应报告：

- `reports/pal/v4_main_culturemert_real_from_v4_main_annotation_stage3/real_constraints_report.json`

这次真人 `PAL` 最终构建出的有效约束为：

- 总约束数：`188`
- positive：`113`
- negative：`75`
- duplicate pair：`10`
- consistent duplicate：`8`
- conflicting duplicate：`2`
- dropped conflicting：`2`

这也是本轮代码改动的重要原因之一：

- 之前的构建逻辑更接近“最后一次写入覆盖前面结果”；
- 现在已经支持显式 `conflict_policy`；
- 当前迁移实验固定使用 `annotation_conflict_policy = drop`

换句话说，这批真人 `PAL` 不是“脏到不能用”，但它确实存在少量冲突样本，不能再用含糊策略处理。

## 6. 当前最重要的实验结果

### 6.1 对齐后的 stage3 迁移结果

对应报告：

- `reports/pal/v4_main_culturemert_real_from_v4_main_annotation_stage3/compare_baseline_vs_real_pal.md`
- `reports/pal/v4_main_culturemert_real_from_v4_main_annotation_stage3/compare_baseline_vs_real_pal.json`

在当前默认的 `target-calibrated` operating point 下，迁移后的真人 `PAL` 相对于当前 `V4 main + CultureMERT stage3` baseline 的变化是：

- `serendipity`: `0.83157 -> 0.83695`，`+0.00538`
- `cultural_calibration_kl`: `2.37596 -> 2.37606`，略差
- `minority_exposure_at_k`: `0.40235 -> 0.38656`，`-0.01579`
- `target_culture_prob_mean`: `0.100047 -> 0.100027`，略差

如何解释这组结果：

1. 真人 `PAL` 迁移并不是“没有信号”，因为 `serendipity` 的确提升了，而且有统计显著性。
2. 但这次默认配置下，PAL 让模型变得更尖锐，代价是 `minority exposure` 和 `calibration` 被轻微拉差。
3. 所以当前结论不能写成“真人 PAL 已经全面提升系统”，而应该写成：
   真人 PAL 已经能对当前 V4 主线产生可检测的行为影响，但还需要进一步调平衡，才能成为论文里的稳定增益证据。

### 6.2 它和当前正式主 benchmark 的关系

当前主 benchmark 参考结果见：

- `reports/benchmarks/v4_main_culturemert_stage3_lambdamart/benchmark_summary.json`

正式主线的 reference method 仍然是：

- `dcas_full_ot_calibrated_target`

对应均值大致为：

- `serendipity = 0.83156`
- `cultural_calibration_kl = 2.02964`
- `minority_exposure_at_k = 0.40233`

这意味着当前迁移 PAL 在“默认 target-calibrated 权重”下的状态更接近：

- 在“探索性/风格惊喜度”上有正向推动；
- 但还没有在主实验最重视的整体平衡点上超过现有 reference。

### 6.3 关键改进：在 PAL checkpoint 上做 calibrated sweep 后，已经找到优于当前主线 reference 的 operating point

对应配置与输出：

- sweep config：`configs/benchmark/recommender_benchmark_v4_main_culturemert_real_pal_stage3_calibration_sweep.run.json`
- sweep summary：`reports/hparam/v4_main_culturemert_real_pal_stage3_calibration_sweep/benchmark_summary.json`
- 与当前主线 reference 的比较：
  - `reports/hparam/v4_main_culturemert_real_pal_stage3_calibration_sweep/comparisons/pal_ot_cal_p3_balanced_vs_stage3_target.md`
  - `reports/hparam/v4_main_culturemert_real_pal_stage3_calibration_sweep/comparisons/pal_ot_cal_p5_target_minor_vs_stage3_target.md`

这一轮 sweep 最重要的发现是：

1. 问题不在于“真人 PAL 没有用”；
2. 问题在于最初评估时沿用了 baseline 的默认 target-calibrated 权重；
3. 一旦把当前 PAL checkpoint 当作新的候选模型重新做 rerank sweep，就能找到比当前正式 reference 更好的 Pareto 点。

目前最值得写进论文主结果补充材料的两个点是：

#### 点 A：`pal_ot_cal_p3_balanced`

相对于当前正式主线 `dcas_full_ot_calibrated_target`：

- `serendipity`: `+0.00609`，`p = 0.00332`
- `cultural_calibration_kl`: `-0.00848`，`p = 0.00332`
- `minority_exposure_at_k`: `+0.03917`，`p = 0.00332`
- `target_culture_prob_mean`: `+0.00199`，`p = 0.00332`

这组结果很重要，因为它说明：

- 不需要额外重训新的大模型；
- 只基于当前迁移后的真人 PAL checkpoint；
- 通过更合适的 calibrated rerank，就已经能在四个核心指标上同时超过当前主线 reference。

#### 点 B：`pal_ot_cal_p5_target_minor`

相对于当前正式主线 `dcas_full_ot_calibrated_target`：

- `serendipity`: `+0.00309`，`p = 0.02326`
- `cultural_calibration_kl`: `-0.00983`，`p = 0.00332`
- `minority_exposure_at_k`: `+0.06158`，`p = 0.00332`
- `target_culture_prob_mean`: `+0.00241`，`p = 0.00332`

这组点更像“偏 exposure / calibration 取向”的版本：

- 相比 `p3_balanced`，它的 `minority exposure` 提升更明显；
- `serendipity` 仍然保持正增益；
- 因而更适合在论文里作为“PAL 带来可调 Pareto frontier”的证据。

### 6.4 一个重要的口径说明：`run_pal_platform` 的评估结果不能直接和论文主 benchmark 数值混读

这一轮继续推进时还发现了一个必须明确写下来的工程事实：

- `run_pal_platform.py` 内部调用的是 `evaluate_recommender.py`
- 论文主实验一直使用的是 `run_recommender_benchmarks.py`

两者都基于同一套推荐函数，但聚合方式和输出字段并不完全相同，因此：

- `run_pal_platform` 输出更适合做“同一配置内的 before/after PAL 闭环检查”
- `run_recommender_benchmarks` 才是论文主结果、跨实验横向比较、以及和现有 reference 方法对齐时应该使用的正式口径

这也解释了为什么：

- `run_pal_platform` 里会看到 `target_culture_prob_mean` 约 `0.10`
- 而主 benchmark 汇总里对应数值约在 `0.18 ~ 0.19`

后续只要涉及“能不能写进论文主结果”，都应优先以 benchmark 口径为准。

## 7. 继续推进后的结果：额外轻量 fine-tune 没有超过当前最佳 PAL checkpoint

为了把 PAL 从“依赖 rerank 找点”进一步推进到“模型本身更稳”，本轮又尝试了两组更轻的 real PAL fine-tune：

- `configs/pal/pal_v4_main_culturemert_real_from_v4_main_annotation_stage3_light_p3.run.json`
- `configs/pal/pal_v4_main_culturemert_real_from_v4_main_annotation_stage3_ultralight_p3.run.json`

其中 `ultralight` 是更值得保留的一组，于是我进一步把它放到了正式 benchmark 口径下：

- config：`configs/benchmark/recommender_benchmark_v4_main_culturemert_real_pal_ultralight_stage3_focus.run.json`
- output：`reports/benchmarks/v4_main_culturemert_real_pal_ultralight_stage3_focus/benchmark_summary.json`

### 7.1 `ultralight` 相对于当前正式主线 reference 仍然是“混合增益”

相对于 `dcas_full_ot_calibrated_target`：

- `serendipity`: `+0.00504`
- `minority_exposure_at_k`: `+0.06448`
- `cultural_calibration_kl`: `+0.00292`，变差
- `target_culture_prob_mean`: `-0.00234`，变差

这说明：

- 额外轻量 fine-tune 仍然可以带来风格惊喜度和 minority exposure 的提升；
- 但它会牺牲一部分 calibration / target alignment；
- 因而还不能替代当前“旧 PAL checkpoint + better rerank point”的最佳证据线。

### 7.2 `ultralight` 相对于当前最佳 PAL 点并没有更优

把它和本轮已经找到的旧 PAL 最优点相比：

#### 相对于 `pal_ot_cal_p3_balanced`

- `serendipity`: `-0.00105`，无显著优势
- `cultural_calibration_kl`: `+0.01141`，明显更差
- `minority_exposure_at_k`: `+0.02531`
- `target_culture_prob_mean`: `-0.00433`，明显更差

#### 相对于 `pal_ot_cal_p5_target_minor`

- `serendipity`: `+0.00236`，但不显著
- `cultural_calibration_kl`: `+0.01052`，明显更差
- `minority_exposure_at_k`: `+0.01790`
- `target_culture_prob_mean`: `-0.00424`，明显更差

也就是说，`ultralight` 的 trade-off 更像：

- 用更多 exposure 去换 calibration 和 target alignment；
- 但换来的 `serendipity` 增益并不足以让它成为新的最佳点。

因此到目前为止，额外轻量 fine-tune 的定位应该是：

- 它证明了 PAL 仍有“继续向训练侧推进”的空间；
- 但它还没有好到足以替换当前的最佳 benchmark 候选。

## 8. 当前最稳妥的论文表述

到这一步，最安全也最真实的写法不是：

- “真人 PAL 已经显著提升了所有主指标”

而应是：

1. 我们已经完成一轮可迁移到 `V4 main` 的真人 `PAL` 标注。
2. 这批标注在当前主线上可以稳定构建真实 pairwise constraints。
3. 在 warm-start 且 benchmark-aligned 的设定下，真人 `PAL` 已经对推荐行为产生显著影响。
4. 直接沿用 baseline 默认权重时，提升主要先体现在 `serendipity` 上。
5. 但在当前 PAL checkpoint 上做 calibrated rerank sweep 后，已经能够找到同时优于当前主线 reference 的 operating point。
6. 后续额外做的轻量 fine-tune 目前尚未超过这些最佳 operating point，因此当前论文主结果仍应以“旧 PAL checkpoint + tuned rerank”作为最稳证据。

因此现在更稳妥也更有力的写法不再是：

- “真人 PAL 还只是能动模型，但没有正结果”

而是：

- “真人 PAL 在默认 operating point 下呈现混合效应，但通过 calibration-aware rerank，可以恢复并放大它在 cross-cultural recommendation 上的整体正增益。”

这个表述比“PAL 一上来就全面全赢”更真实，也比“PAL 还没有形成结果”更符合当前证据。

## 9. 当前已经落实的工程改动

为了让这条 PAL 线后续可以继续复用，本轮已经补上的工程能力包括：

- `dcas/scripts/build_pal_constraints_from_annotations.py`
  - 支持 `conflict_policy`
  - 支持输出 `report_json`
- `dcas/pipelines.py`
  - 支持从 baseline checkpoint warm-start
- `dcas/scripts/evaluate_recommender.py`
  - 支持 `ot_calibrated / knn_calibrated`
  - 支持显式 rerank 权重
- `dcas/scripts/run_pal_platform.py`
  - 支持 benchmark-aligned real-round
  - 支持 warm-start / ranking-signal preservation / calibrated evaluation

也就是说，后面如果我们继续迭代 PAL，不需要再重新解释“为什么这条线之前跑出来的结果不可信”，因为现在流程已经被补到了和当前 V4 主 benchmark 可直接对齐的程度。

## 10. 下一步最值得继续做什么

基于当前最新结果，下一步最值得优先做的不是继续把 PAL 训得更重，而是：

1. 固定 `annotation_conflict_policy = drop`
2. 把 `pal_ot_cal_p3_balanced` 和 `pal_ot_cal_p5_target_minor` 作为当前最值得保留的两个论文 operating point
3. 如果还想继续做训练侧提升，应避免再沿当前这条 `ultralight` 方向盲目扩网格，而应更有针对性地控制 calibration / target alignment 不被破坏
4. 在没有更强训练侧证据前，论文正文优先写“真人 PAL + calibrated rerank”的结果，而不是把重心转回新一轮 checkpoint 搜索

这也是本轮迁移记录最重要的落脚点：

- 真人 `PAL` 已经不是“待定计划”
- 它已经变成当前 `V4 main` 主线上的一个真实、可复用、可继续优化的实验资产
- 而且它现在已经能够通过合适的 calibrated operating point，转化为比当前主线 reference 更强的可写论文结果
