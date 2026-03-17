# Research Dataset V2 执行状态更新

更新时间：2026-03-15  
范围：记录当前已经实际跑通的导入 probe，以及 Germany / Japan / Turkey 的最新审计状态。

---

## 1. 已经实际跑通的导入 probe

当前已经在本地成功导入并生成了单域 `audio/ + metadata.csv + import_report.json` 的文化域有：

- `china`
- `india`
- `anglo_pop`
- `turkey`
- `germany`

对应目录：

- [china](E:/Desktop/Echo/storage/public/research_dataset_v2/china)
- [india](E:/Desktop/Echo/storage/public/research_dataset_v2/india)
- [anglo_pop](E:/Desktop/Echo/storage/public/research_dataset_v2/anglo_pop)
- [turkey](E:/Desktop/Echo/storage/public/research_dataset_v2/turkey)
- [germany](E:/Desktop/Echo/storage/public/research_dataset_v2/germany)

### China probe

来源：
- `ccmusic-database/CTIS`

结果：
- `limit=12`
- `imported=12`
- `skipped=0`

产物：
- [china/import_report.json](E:/Desktop/Echo/storage/public/research_dataset_v2/china/import_report.json)
- [china/metadata.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/china/metadata.csv)

### India probe

来源：
- `neerajaabhyankar/hindustani-raag-small`

结果：
- `limit=12`
- `imported=12`
- `skipped=0`

产物：
- [india/import_report.json](E:/Desktop/Echo/storage/public/research_dataset_v2/india/import_report.json)
- [india/metadata.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/india/metadata.csv)

### Anglo-pop probe

来源：
- `vtsouval/mtg_jamendo_autotagging`

结果：
- `limit=12`
- `imported=12`
- `skipped=0`

产物：
- [anglo_pop/import_report.json](E:/Desktop/Echo/storage/public/research_dataset_v2/anglo_pop/import_report.json)
- [anglo_pop/metadata.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/anglo_pop/metadata.csv)

说明：
- 这条 probe 很重要，因为它证明了我们现在不必卡在原始 MTG-Jamendo 官网下载上
- HF parquet 版本已经能直接接入现有导入脚本

### Turkey probe

来源：
- `bilal63/turkish_music_emotion_dataset`

结果：
- `limit=12`
- `imported=12`
- `skipped=0`

产物：
- [turkey/import_report.json](E:/Desktop/Echo/storage/public/research_dataset_v2/turkey/import_report.json)
- [turkey/metadata.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/turkey/metadata.csv)

说明：
- 工程上可导入
- 但 source license 仍需继续确认，因此当前 Turkey 仍保持 `provisional`

### Germany import

来源：
- Europeana `Westphalian Folk Song and Sound Archive`

结果：
- 通过 Europeana API 搜索与直接 mp3 下载组合方式
- `requested_limit=20`
- `scanned=84`
- `imported=20`
- `errors=0`

产物：
- [germany/import_report.json](E:/Desktop/Echo/storage/public/research_dataset_v2/germany/import_report.json)
- [germany/metadata.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/germany/metadata.csv)

说明：
- Germany 已经从“只有 rights 审计结果”推进到“实际导入成功”
- 当前这条线已经足以继续留在主域里推进

---

## 2. Germany 的最新状态

Germany 当前最关键的进展是：

- 已经完成 Europeana Westphalian collection 的 `50` 条 item-level API 抽样审计

报告文件：

- [germany_europeana_audit.json](E:/Desktop/Echo/reports/research_dataset_v2/germany_europeana_audit.json)
- [germany_europeana_audit.csv](E:/Desktop/Echo/reports/research_dataset_v2/germany_europeana_audit.csv)

抽样结果要点：

- `returned_rows = 50`
- `n_sound_type = 50`
- `n_has_audio_proxy = 50`
- `n_has_landing_page = 50`
- 抽样中的 rights 一致为：
  - `http://creativecommons.org/licenses/by-nc-sa/3.0/`

这意味着：

- Germany 这条线不再只是“概念候选”
- 它已经进入“有条目级 rights 证据、可继续推进”的状态

当前建议：

- 将 Germany 主来源状态进一步提升到：
  - `probe_import_ready`

也就是说，Germany 现在不只是保留在主域里，而是已经具备继续扩展导入的条件。

---

## 3. Japan 的最新状态

Japan 这条线目前最关键的新发现不是 probe 成功，而是：

- `tts-dataset/japanese-singing-voice` 的确可以作为一个规模型候选
- 但它是 gated dataset
- 当前未认证访问，因此**不能直接跑通自动导入**

因此当前 Japan 的实际状态是：

- 如果维持严格 `japanese_traditional`：仍偏弱
- 如果采用 `japanese_music_audio` 宽定义：概念上可行
- 但工程上仍需：
  - 获得该 gated 数据集访问
  - 或继续补另一个可直接开放导入的来源

所以 Japan 现在可以继续保留在主域，但**还没有进入“可直接导入”状态**。

---

## 4. 当前主域状态分层

### Ready / 已实际跑通 probe 或 seed import

- `china`
- `india`
- `anglo_pop`
- `germany`

### 工程已跑通但许可仍待确认

- `turkey`

### 定义已放宽，但来源接入仍待补

- `japan`

---

## 5. 当前最合理的下一步

当前最值钱的下一步顺序是：

1. 扩展 Germany 的 Europeana 导入样本量，逐步逼近 `100-200`  
2. 给 `anglo_pop` 增加 `pop` 和语言过滤策略  
3. 继续补 Japan 的可直接导入来源或 gated 访问  
4. 补 Turkey 的 license 证据

## 6. 当前 probe 级 merged metadata

当前已经合成出一版 5 域 probe 级 merged metadata：

- [metadata_probe_5domains.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_probe_5domains.csv)
- [metadata_probe_5domains.csv.merge_report.json](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_probe_5domains.csv.merge_report.json)

当前构成：

- `china = 12`
- `india = 12`
- `anglo_pop = 12`
- `turkey = 12`
- `germany = 20`

这说明：

- `research_dataset_v2` 已经不再只是概念方案
- 我们已经有了一版真实可用的 5 域 probe 数据底座

到这一步，`research_dataset_v2` 已经从纯规划阶段，进入了“部分域可实际开工”的状态。
