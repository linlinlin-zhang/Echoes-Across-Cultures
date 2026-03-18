# Research Dataset V3 Indonesia 主域补充说明

日期：2026-03-18

## 1. 本次变更

`Indonesia` 已从原先的 `indonesia_probe` 提升为正式主文化域，并并入：

- `storage/public/research_dataset_v3/metadata_v3_main.csv`

当前 `Indonesia` 主域由两部分组成：

- `55` 条传统 `gamelan_orchestra` 长片段
- `19` 条来自 `Free Music Archive` 的 Indonesia 关联现代补充

最终主域规模：

- `74` 条

## 2. 当前状态

Indonesia 主域文件：

- `storage/public/research_dataset_v3/indonesia/metadata.csv`

主表：

- `storage/public/research_dataset_v3/metadata_v3_main.csv`

汇总：

- `storage/public/research_dataset_v3/summary_v3_main.json`

## 3. 并入后的主表规模

当前主表共：

- `1123` 条
- `10` 个主文化域

主文化域为：

- `china`
- `india`
- `turkey`
- `indonesia`
- `germany`
- `france`
- `italy`
- `great_britain`
- `russia`
- `modern_english_pop`

## 4. Indonesia 域概况

- `n_rows = 74`
- `n_artists = 7`
- `duration_min = 79.233333`
- `duration_median = 119.718753`
- `duration_max = 487.479728`

来源：

- `gamelan_music_dataset`
- `Free Music Archive`

## 5. 解释

这次提升的意义不是让 `Indonesia` 变成一个大规模主域，而是让它从“只有传统片段的 probe”变成一个：

- 数量达到 `70+`
- 同时带有 `traditional + modern supplement`
- 能纳入主表统一流程

的正式文化域。

它目前仍然是主表中样本最少的文化域，但已经具备进入 V3 主流程的基本可用性。
