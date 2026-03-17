# Dataset Upgrade Notes (2026-03-17)

本文档记录 2026-03-17 这轮对最终数据集底座的两项实际升级：

1. Germany 归并为单目录 final 版
2. Norway 从 probe 扩成完整域，并接入 6 域候选主版

## 1. Germany final

此前 Germany 音频虽然已经都在本地，但分散在多个目录中：
- `germany/`
- `germany_cursor_expand/`
- `germany_cursor_expand_220/`
- `germany_q_*`

本轮已经把 Germany 的 `253` 条音频统一归并到：
- [germany_final](E:/Desktop/Echo/storage/public/research_dataset_v2/germany_final)

对应文件：
- [metadata.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/germany_final/metadata.csv)
- [import_report.json](E:/Desktop/Echo/storage/public/research_dataset_v2/germany_final/import_report.json)

这意味着 Germany 现在已经从“多目录拼接域”升级成了：
- **单目录 final 域**

## 2. Clean 5-domain main

在 Germany 归并完成后，已经重新生成了一版更干净的 5 域主版：
- [metadata_v2_main_clean.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_v2_main_clean.csv)

规模：
- `china = 250`
- `india = 250`
- `anglo_pop = 250`
- `kazakhstan = 250`
- `germany = 253`
- 总计 `1253`

对应 merge report：
- [metadata_v2_main_clean.csv.merge_report.json](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_v2_main_clean.csv.merge_report.json)

这版相比旧的 `metadata_v2_main.csv`，最大的改进是：
- Germany 不再依赖多个散落目录
- 更适合作为“冻结后不再改动”的最终主版底座

## 3. Norway full domain

此前 Norway 只有 12 条 probe。

本轮已经将 `Bots4M/HF2-Hardanger-fiddle-dataset` 全量扩成完整域：
- [norway](E:/Desktop/Echo/storage/public/research_dataset_v2/norway)

对应文件：
- [metadata.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/norway/metadata.csv)
- [import_report.json](E:/Desktop/Echo/storage/public/research_dataset_v2/norway/import_report.json)

当前规模：
- `norway = 119`

## 4. 6-domain candidate main

在不覆盖现有主版的前提下，已经生成了一个 6 域候选主版：
- [metadata_v2_main_6domains_candidate.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_v2_main_6domains_candidate.csv)

规模：
- `china = 250`
- `india = 250`
- `anglo_pop = 250`
- `kazakhstan = 250`
- `germany = 253`
- `norway = 119`
- 总计 `1372`

对应 merge report：
- [metadata_v2_main_6domains_candidate.csv.merge_report.json](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_v2_main_6domains_candidate.csv.merge_report.json)

## 5. 当前可选的数据冻结版本

### 方案 A：冻结 clean 5-domain main

使用：
- [metadata_v2_main_clean.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_v2_main_clean.csv)

优点：
- 最稳
- 所有当前实验已与 5 域主版高度对齐
- 不引入新域变量

### 方案 B：冻结 6-domain candidate

使用：
- [metadata_v2_main_6domains_candidate.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_v2_main_6domains_candidate.csv)

优点：
- 跨文化覆盖更广
- 从 `5` 域提升到 `6` 域
- Norway 许可清楚、接入成本低

缺点：
- 需要重跑 6 域版本的 embeddings 与 benchmark
- Norway 样本量 `119`，低于其他主域

## 6. 当前建议

如果目标是：
- 尽快进入真人 PAL
- 尽快冻结最终版本

更稳的做法是：
- 把 `metadata_v2_main_clean.csv` 当作当前稳定底座
- 把 `metadata_v2_main_6domains_candidate.csv` 当作“可以升级，但要重新跑主实验”的候选版

也就是说：
- **5 域 clean 版已经可以直接冻结**
- **6 域 Norway 版已经可以作为增强候选**
