# Research Dataset V2 备选主版（Kazakhstan 替换 Turkey）

更新时间：2026-03-16

---

## 1. 目的

这一版本用于解决 Turkey 许可证据仍偏弱的问题。

策略是：

- 保留当前最稳的四个主域
  - `china`
  - `india`
  - `anglo_pop`
  - `germany`
- 用 `kazakhstan` 替换 `turkey`

从而形成一版许可证据更清楚、工程接入同样顺畅的备选主实验底座。

---

## 2. 备选主版文件

- [metadata_v2_alt_kazakhstan.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_v2_alt_kazakhstan.csv)

对应合并报告：

- [metadata_v2_alt_kazakhstan.csv.merge_report.json](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_v2_alt_kazakhstan.csv.merge_report.json)

Kazakhstan 本地域目录：

- [kazakhstan](E:/Desktop/Echo/storage/public/research_dataset_v2/kazakhstan)

---

## 3. 当前规模

- `china = 250`
- `india = 250`
- `anglo_pop = 250`
- `kazakhstan = 250`
- `germany = 253`

总计：

- `1253` 条音频记录

---

## 4. 为什么 Kazakhstan 适合作为 Turkey 备选

与当前 Turkey 路线相比，Kazakhstan 的优点是：

- 有公开可下载的真实音频
- Hugging Face 数据集卡和 README 都明确写有 `CC BY-NC 4.0`
- 仓库内自带 `metadata.jsonl`
- 本地已经成功导入 `250` 条

因此它非常适合作为：

- 论文公开版的更稳妥备选
- Turkey 许可证问题未解决时的主实验替换域

---

## 5. 推荐使用方式

如果后续 Turkey 许可证能补强：

- 继续使用标准主版 [metadata_v2_main.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_v2_main.csv)

如果 Turkey 许可证仍无法补强：

- 直接切换到这版 Kazakhstan 备选主版

这样可以避免在主实验阶段因为许可不稳而拖慢 Gemini embedding 和正式实验。
