# Research Dataset V2 Main 状态说明

更新时间：2026-03-16  
适用范围：当前已落地的 `research_dataset_v2` 主实验域版本，用于后续统一 embedding 构建、DCAS 训练和论文数据说明。

---

## 1. 当前主实验域

当前 `v2-main` 收敛为 **5 个主实验域**：

- `china`
- `india`
- `anglo_pop`
- `kazakhstan`
- `germany`

本版不包含：

- `japan`
- `spain`
- `france`
- `korea`

这些域仍保留在来源与访问策略层，但尚未进入当前这版主实验数据底座。

说明：

- 原 Turkey 主版已备份为 [metadata_v2_main_turkey_legacy.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_v2_main_turkey_legacy.csv)

---

## 2. 当前已落地的数据规模

主实验合并文件：

- [metadata_v2_main.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_v2_main.csv)

当前规模：

- `china = 250`
- `india = 250`
- `anglo_pop = 250`
- `kazakhstan = 250`
- `germany = 253`

总计：

- `1253` 条音频记录

说明：

- `china / india / kazakhstan` 通过公开音频数据集直接扩量到 `250`
- `anglo_pop` 先导入 `1000` 条原始候选，再基于 `pop-like` 标签过滤出 `250`
- `germany` 当前已通过 Europeana `DATA_PROVIDER` 字段检索加 `cursor` 分页显著扩量；在旧版 `48` 条基础上分两轮新增导入并去重后达到 `253` 条可用音频
- `turkey` 已从当前主版移出，转为内部 legacy 对照版本以避免许可与主实验叙事混淆

---

## 3. 当前版本为什么仍然成立

虽然 Germany 还没有扩到和其他域同规模，但这版 `v2-main` 已经达到更适合方法实验的规模，原因是：

- 已经形成统一 schema
- 已经有 5 个真实文化域
- 已经具备进入统一 embedding 构建的最小数据底座
- 已经可以支持 Gemini embedding 的正式构建与 pilot 训练

因此，这一版可被视为：

- **V2-main paper-ready pilot**

而不是最终 submission-scale full release。

---

## 4. Anglo-pop 的当前定义

当前 `anglo_pop` 并不是严格意义上的“已知英语歌词流行曲库”，而是：

- 基于 `vtsouval/mtg_jamendo_autotagging`
- 从原始候选池中筛出带有下列 `pop-like` 标签的条目：
  - `pop`
  - `poprock`
  - `popfolk`
  - `electropop`
  - `synthpop`
  - `instrumentalpop`

当前过滤产物目录：

- [anglo_pop_main](E:/Desktop/Echo/storage/public/research_dataset_v2/anglo_pop_main)

这意味着当前 Anglo 域应被解释为：

- **pop-tagged anglophone-friendly anchor domain**

而不是已经严格验证语言字段后的“纯英语流行域”。

---

## 5. 每个主域的状态判断

### 5.1 China

- 状态：`ready`
- 说明：来源、许可、音频与规模都稳定

### 5.2 India

- 状态：`ready`
- 说明：来源清楚、规模稳定、标签一致

### 5.3 Anglo-pop

- 状态：`ready_with_proxy_caveat`
- 说明：已完成 pop-like 过滤，但仍缺显式语言字段

### 5.4 Kazakhstan

- 状态：`ready`
- 说明：音频、metadata 和 README 级别许可证据都已落地，当前作为正式主版主域使用

### 5.5 Germany

- 状态：`balanced_ready`
- 说明：已完成条目级 rights 审计，并通过 Europeana `DATA_PROVIDER` 字段检索、`cursor` 分页和去重合并将 Germany 域提升到 `253` 条，已经和其他主域达到同量级，可稳定进入正式 embedding 构建

---

## 6. 这版数据最适合做什么

当前最适合：

- Gemini embedding 构建
- 统一向量格式与 manifest 管线验证
- DCAS 的 pilot 训练与基础主实验
- 主实验域定义冻结后的方法开发

当前还不建议直接把这版当成：

- 最终论文 full-scale 数据集
- 强 benchmark 版本

---

## 7. 当前版后续建议

### 立即可做

- 在 `metadata_v2_main.csv` 基础上构建 Gemini embeddings
- 生成统一 `tracks.npz`
- 跑一版 DCAS 与基础 baseline

### 后续应继续补

- Germany 继续扩量或申请正式 Europeana API key
- Turkey 作为 legacy/internal 对照域保留
- Japan / Spain / France / Korea 的第二阶段扩展

---

## 8. 结论

当前 `v2-main` 已经从“probe 级拼接”推进成一版超过 `1000` 条的主实验数据底座。  
它已经足以支持下一步统一 Gemini embedding 构建，但仍应被诚实地定位为：

- **完整可运行的主实验版**
- **而非最终大规模发布版**
