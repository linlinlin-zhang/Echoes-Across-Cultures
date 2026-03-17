# Research Dataset V2 Main 构建记录

更新时间：2026-03-16

---

## 1. 构建目标

在不等待 Japan / Spain / France / Korea 进一步接入的情况下，先基于当前最稳的主实验域构建一版：

- 可运行
- 可统一 embedding
- 可接 DCAS

的 `v2-main` 数据底座。

---

## 2. 使用的主实验域

- `china`
- `india`
- `anglo_pop`
- `kazakhstan`
- `germany`

---

## 3. 实际构建动作

### 3.1 China

使用：

- `ccmusic-database/CTIS`

扩量到：

- `250` 条

### 3.2 India

使用：

- `neerajaabhyankar/hindustani-raag-small`

扩量到：

- `250` 条

### 3.3 Kazakhstan

使用：

- `rtrk/kazakh-traditional-audio`

扩量到：

- `250` 条

说明：

- 仓库自带 `metadata.jsonl`
- README 和数据集卡都明确写有 `CC BY-NC 4.0`
- 当前作为正式主版主域使用

### 3.4 Germany

使用：

- Europeana `Westphalian Folk Song and Sound Archive`

当前成功导入并去重合并：

- `253` 条

限制：

- 旧的 `start` 深分页在 `api2demo` 下会触发 `400 Bad Request`
- 当前已切换为 Europeana `DATA_PROVIDER` 字段检索加 `cursor` 分页，并连续完成两轮批量导入
- Germany 当前总量已经提升到 `253` 条
- 继续扩量仍然建议优先申请正式 Europeana API key，而不是长期依赖 `api2demo`

### 3.5 Anglo-pop

原始导入：

- `vtsouval/mtg_jamendo_autotagging`
- `1000` 条

再过滤为：

- `250` 条 `pop-like` 样本

过滤脚本：

- [filter_metadata_by_keywords.py](E:/Desktop/Echo/dcas/scripts/filter_metadata_by_keywords.py)

过滤结果目录：

- [anglo_pop_main](E:/Desktop/Echo/storage/public/research_dataset_v2/anglo_pop_main)

---

## 4. 主合并文件

合并结果：

- [metadata_v2_main.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_v2_main.csv)

当前总量：

- `1253` 条

---

## 5. 当前结论

`v2-main` 已可作为：

- Gemini embedding 构建输入
- DCAS 主实验输入

Germany 仍值得继续扩量，但 Turkey 已从主版移出，改为 legacy/internal 对照域，因此当前版本应被视为：

- **可运行主实验版**
- **不是最终 fully-polished 发布版**
