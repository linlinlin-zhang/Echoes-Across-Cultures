# Research Dataset V3 构建报告

日期：2026-03-18

## 1. 结果概览

`research_dataset_v3` 已经实际构建完成，主数据集包含 9 个文化域，共 `1049` 条音频，全部满足：

- 每个主文化域 `> 100` 条且 `< 200` 条
- 所有主数据音频 `duration_sec >= 30`
- 西方国家域严格排除了 `experimental / electronic / rock / novelty / international / spoken / hip-hop`

主数据集位置：

- `storage/public/research_dataset_v3/metadata_v3_main.csv`
- `storage/public/research_dataset_v3/summary_v3_main.json`

构建脚本：

- `dcas/scripts/build_research_dataset_v3.py`

主数据集音频总体积约：

- 主表 9 域：约 `10.27 GiB`
- 额外 Indonesia 探针域：约 `1.73 GiB`

## 2. 主文化域清单

| culture | tracks | min duration (s) | source |
|---|---:|---:|---|
| `india` | 108 | 87.577 | `saraga_hindustani` |
| `turkey` | 150 | 30.000 | `bilal63/turkish_music_emotion_dataset` |
| `china` | 146 | 30.731 | `compmusic_jingju_acappella` + `ccmusic-database/CTIS` |
| `modern_english_pop` | 120 | 30.063 | `vtsouval/mtg_jamendo_autotagging` 派生清洗子集 |
| `germany` | 105 | 41.000 | `Free Music Archive` |
| `france` | 105 | 86.933 | `Free Music Archive` |
| `italy` | 105 | 65.278 | `Free Music Archive` |
| `great_britain` | 105 | 53.207 | `Free Music Archive` |
| `russia` | 105 | 90.372 | `Free Music Archive` |

## 3. 非西方文化域说明

### 3.1 India

- 已下载到本地并落盘。
- 当前主域使用 `Saraga Hindustani`。
- 该域片段长度明显充足，时长中位数高于 1000 秒，适合作为强主域。

### 3.2 Turkey

- 已下载到本地并落盘。
- 当前主域使用本地可用的土耳其现代音乐数据源，统一筛成 `>=30s`。
- 该域当前更偏现代，不是纯传统 makam 语料。

### 3.3 China

- 已下载到本地并落盘。
- 当前主域使用：
  - `Jingju Acappella`
  - `CTIS` 中 `>=30s` 的传统器乐/片段
- 该域满足时长要求，但目前整体更偏传统中国音乐，不是现代中文流行。

说明：

- 原计划中的 `OpenCpop` 没有进入主表，因为它本质上是单歌手、utterance 导向语料，不适合直接满足本轮“每条 >30 秒、可直接做域级训练”的约束。

## 4. 西方国家域说明

本轮主表中的 `Germany / France / Italy / Great Britain / Russia` 均来自 `Free Music Archive`，并已实际下载到本地。

筛选规则：

- 先按国家映射筛选
- 再剔除 `experimental / electronic / rock / novelty / international / spoken / hip-hop`
- 只保留 `>=30s`
- 每国控制在 `105` 条

实现细节：

- 使用本地 `fma_metadata.zip` 做国家候选筛选
- 直接使用 `raw_tracks.csv` 中的 `track_url`
- 再从公开 FMA track page 抓取 `fileUrl`
- 对大量历史失效页面做自动跳过和候选池补位

这一步已经证明可复现，但也说明 FMA 历史链接老化比较严重，因此脚本里增加了：

- 候选池补位
- 失效页自动跳过
- 最终入选文件重命名与清理

## 5. 现代英语流行对比基准

`modern_english_pop` 已纳入主表，规模为 `120` 条。

选择逻辑：

- 使用现有英文流行候选池
- 去除被禁流派相关标签
- 保留时长 `>=30s`
- 控制规模到 `120`

这个域的作用是作为现代英语流行对比基准，而不是国家域。

## 6. Indonesia 推进结果

Indonesia 已做实际推进和本地下载，但目前**不进入主表**。

当前状态：

- 本地已落一个 `indonesia_probe`
- 条数为 `55`
- 全部 `>=30s`
- 来源为 `gamelan_music_dataset`

结论：

- 它适合作为 `probe domain`
- 不满足当前主表“每域 >100”的硬门槛
- 因此被保留为附加域，不并入 `metadata_v3_main.csv`

位置：

- `storage/public/research_dataset_v3/indonesia_probe/metadata.csv`

## 7. 本轮限制

### 7.1 China 仍偏传统

虽然中国域已经可用，但目前结构更接近：

- `jingju / traditional instrumental`

而不是：

- `mandarin_pop + traditional supplement`

如果后续能拿到更合适的现代中文流行长音频数据源，可以作为 V3.1 或 V4 继续补强。

### 7.2 Turkey / Modern English Pop 的 artist 元数据较弱

这两个域当前已经可用于训练和完整流程，但艺术家层 metadata 不如 FMA 国家域完整。

### 7.3 FMA 历史页面失效明显

脚本已经处理了这个问题，但后续如果要扩容 FMA 国家域，仍建议保留：

- 更深候选池
- 失败重试
- 链接失效自动补位

## 8. 可直接用于后续流程的文件

主表：

- `storage/public/research_dataset_v3/metadata_v3_main.csv`

汇总：

- `storage/public/research_dataset_v3/summary_v3_main.json`

去重报告：

- `storage/public/research_dataset_v3/metadata_v3_main.csv.merge_report.json`

各域 metadata：

- `storage/public/research_dataset_v3/india/metadata.csv`
- `storage/public/research_dataset_v3/turkey/metadata.csv`
- `storage/public/research_dataset_v3/china/metadata.csv`
- `storage/public/research_dataset_v3/modern_english_pop/metadata.csv`
- `storage/public/research_dataset_v3/germany/metadata.csv`
- `storage/public/research_dataset_v3/france/metadata.csv`
- `storage/public/research_dataset_v3/italy/metadata.csv`
- `storage/public/research_dataset_v3/great_britain/metadata.csv`
- `storage/public/research_dataset_v3/russia/metadata.csv`
- `storage/public/research_dataset_v3/indonesia_probe/metadata.csv`

## 9. 当前判断

这一版 `V3` 已经可以作为一个完整、可用、可实操推进全流程的研究数据集版本：

- 有 3 个非西方主域：`india / turkey / china`
- 有 5 个西方国家域：`germany / france / italy / great_britain / russia`
- 有 1 个现代英语流行对比基准：`modern_english_pop`
- 有 1 个暂不并入主表的额外探针域：`indonesia_probe`

如果后续继续推进，我建议优先做两件事：

1. 为 `china` 增补现代中文流行长音频主源。
2. 继续寻找可把 `indonesia_probe` 提升到 `>100` 条的第二来源。
