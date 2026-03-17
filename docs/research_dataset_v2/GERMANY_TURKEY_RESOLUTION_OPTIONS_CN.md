# Germany / Turkey 问题处理建议

更新时间：2026-03-16

---

## 1. Germany 当前是否还卡在扩量上

结论：**已经部分解决，不再是主阻塞项。**

当前 Germany 主源仍然是：

- Europeana `Westphalian Folk Song and Sound Archive`

这轮的关键进展有两点：

1. 旧问题并不是 Europeana 没有足够条目，而是我们早先使用的深分页方式在 `api2demo` 下会被卡住。
2. 改成：
   - `DATA_PROVIDER:"Westphalian Folk Song and Sound Archive"` 字段检索
   - `cursor` 分页

   之后，导入脚本已经连续完成两轮批量导入。

与旧版 Germany 主 metadata 去重合并后：

- Germany 从 `48` 条提升到 `253` 条

因此，Germany 现在的真实状态应理解为：

- **已经达到主实验同量级**
- **仍值得继续扩量**
- **但扩量方式应转为正式 API key + cursor workflow**

### Germany 后续建议

优先级从高到低：

1. 申请 Europeana 正式 API key，不再长期依赖 `api2demo`
2. 继续沿用 `DATA_PROVIDER + cursor` 的主检索方式
3. 对少量 `404` 失效 mp3 重新扫尾补抓
4. 如有需要，再增加少量合法补充源，但当前已经不需要为了“凑够量”而更换主源

---

## 2. Turkey 当前问题到底是什么

结论：**Turkey 的问题不是音频拿不到，而是公开许可证据不够硬。**

当前 Turkey 主源：

- `bilal63/turkish_music_emotion_dataset`

目前已经确认的事实：

- Hugging Face API 能确认该数据集存在、为 `audiofolder`、模态是 `audio`
- 公开文件树能看到 `400` 条左右音频文件
- 但公开 README 基本为空
- HF API 返回中也没有明确的 license tag

这意味着：

- **工程上可用**
- **研究内部试验可用**
- **但还不能把它当成“许可已完全坐实的公开可重分发主源”**

### Turkey 后续建议

优先级从高到低：

1. 联系数据集维护者，请求书面或页面级 license 说明
2. 如果能拿到明确许可，则可把 Turkey 保留为内部对照域或附加实验域
3. 如果拿不到明确许可，则：
   - 内部实验可继续保留
   - 论文公开版和对外发布版应避免把原始音频视作可再分发资源
4. 如果需要一个更干净的公开替代域，优先考虑 Kazakhstan

---

## 3. 最强的相对容易替代域

当前最推荐的替代域是：

- `kazakhstan`

原因：

- 来源：`rtrk/kazakh-traditional-audio`
- Hugging Face API 明确显示：
  - `license:cc-by-nc-4.0`
  - `size_categories:1K<n<10K`
- 有真实音频
- 文化域定义清楚
- 本地已经成功导入 `250` 条
- 工程接入难度低

因此，Kazakhstan 已经成为当前最稳的正式主域替代方案。

---

## 4. 当前推荐决策

如果以“尽快进入 Gemini embedding 正式构建”为目标：

- **Germany：保留并继续扩量**
- **Kazakhstan：已正式提升为当前主版主域**
- **Turkey：降为 legacy/internal 对照域**

这条路线能同时兼顾：

- 主实验域数量稳定
- 文化覆盖仍然足够
- 数据治理风险可控
