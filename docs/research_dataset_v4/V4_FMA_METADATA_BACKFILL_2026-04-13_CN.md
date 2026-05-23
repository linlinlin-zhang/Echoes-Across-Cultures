# V4 FMA Metadata Backfill

更新日期：2026-04-13

## 1. 本轮做了什么

本轮没有重抓音频，也没有替换任何现有样本。

做的是一件更稳的事：

- 把当前数据集中 `source_dataset = Free Music Archive` 的样本，回填了 FMA 官方 metadata 里的结构化字段
- 让原本只零散藏在 `notes` 或外链里的信息，变成可以直接统计和审计的列

新增脚本：

- [enrich_fma_metadata.py](E:/Desktop/Echo/dcas/scripts/enrich_fma_metadata.py)

## 2. 实际回填到了哪些文件

已回填：

- [metadata_release.csv](E:/Desktop/Echo/storage/public/research_dataset_v4/main/metadata_release.csv)
- [france/metadata.csv](E:/Desktop/Echo/storage/public/research_dataset_v3/france/metadata.csv)
- [germany/metadata.csv](E:/Desktop/Echo/storage/public/research_dataset_v3/germany/metadata.csv)
- [great_britain/metadata.csv](E:/Desktop/Echo/storage/public/research_dataset_v3/great_britain/metadata.csv)
- [italy/metadata.csv](E:/Desktop/Echo/storage/public/research_dataset_v3/italy/metadata.csv)
- [russia/metadata.csv](E:/Desktop/Echo/storage/public/research_dataset_v3/russia/metadata.csv)
- [indonesia/metadata.csv](E:/Desktop/Echo/storage/public/research_dataset_v3/indonesia/metadata.csv)

## 3. 回填了哪些字段

当前 V4 主发布表新增的 FMA 结构化字段包括：

- `fma_track_id`
- `fma_album_id`
- `fma_artist_id`
- `fma_album_title`
- `fma_artist_location`
- `fma_artist_latitude`
- `fma_artist_longitude`
- `fma_artist_website`
- `fma_track_genre_top`
- `fma_track_listens`
- `fma_track_favorites`
- `fma_track_language_code`
- `fma_track_date_recorded`
- `fma_track_url`
- `fma_album_url`
- `fma_artist_url`
- `fma_match_method`

## 4. V4 主数据集的回填结果

对应报告：

- [metadata_release.csv.fma_enrichment_report.json](E:/Desktop/Echo/storage/public/research_dataset_v4/main/metadata_release.csv.fma_enrichment_report.json)
- [metadata_release.csv.fma_enrichment_report.md](E:/Desktop/Echo/storage/public/research_dataset_v4/main/metadata_release.csv.fma_enrichment_report.md)

关键数字如下：

- V4 main 中 FMA 行数：`544`
- 成功匹配回 FMA 官方元数据：`544 / 544 = 100%`
- 唯一 FMA artist id 数：`220`
- 唯一 artist location 原始字符串数：`136`

字段覆盖率：

- `fma_artist_id`: `100.00%`
- `fma_artist_location`: `99.45%`
- `fma_artist_latitude`: `48.35%`
- `fma_artist_longitude`: `48.35%`
- `fma_artist_website`: `91.54%`
- `fma_track_genre_top`: `61.03%`
- `fma_track_listens`: `100.00%`
- `fma_track_favorites`: `100.00%`

## 5. 按文化桶看的结果

- `france`: `105` 条 FMA 样本，`48` 个 artist id，`26` 个 location 字符串，location 覆盖率 `100.00%`
- `germany`: `105` 条 FMA 样本，`38` 个 artist id，`19` 个 location 字符串，location 覆盖率 `97.14%`
- `great_britain`: `105` 条 FMA 样本，`77` 个 artist id，`57` 个 location 字符串，location 覆盖率 `100.00%`
- `italy`: `105` 条 FMA 样本，`22` 个 artist id，`14` 个 location 字符串，location 覆盖率 `100.00%`
- `russia`: `105` 条 FMA 样本，`28` 个 artist id，`15` 个 location 字符串，location 覆盖率 `100.00%`
- `indonesia`: `19` 条 FMA 样本，`7` 个 artist id，`5` 个 location 字符串，location 覆盖率 `100.00%`

## 6. 这件事对论文有什么用

这轮回填不能消灭 source confound，但能让我们更诚实也更有证据地说明：

1. 西方若干文化桶虽然主要来自同一个开放平台 FMA，但并不是少数作者的重复采样。
2. FMA 子集中存在显著的 artist-level 多样性。
3. FMA 子集中也存在 location-level 多样性，而且很多样本带有更细粒度的城市或地区字符串。
4. 因此，审稿时可以避免把 FMA 简化描述成“单一来源、单一作者、单一地点”的过强指控。

## 7. 仍然要诚实承认的限制

这轮回填解决的是：

- “缺少结构化来源内部多样性证据”

它没有完全解决的是：

- “多个欧洲文化桶依然主要来自同一平台 FMA”

所以论文里更合适的说法是：

- `source concentration remains a limitation`
- `but the FMA-backed subsets are internally heterogeneous at the artist and location levels`
