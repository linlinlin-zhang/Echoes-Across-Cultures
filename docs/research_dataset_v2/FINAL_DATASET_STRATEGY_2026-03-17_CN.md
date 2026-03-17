# Final Dataset Strategy (2026-03-17)

本文档用于回答一个当前最关键的问题：

在 `DCAS + Gemini/CultureMERT + benchmark + PAL platform` 已经基本跑通的前提下，
我们应该把哪一版数据集冻结成后续真人 PAL 和论文主实验的最终底座。

## 1. 当前现实起点

当前正式主版：
- [metadata_v2_main.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_v2_main.csv)

当前规模：
- `china = 250`
- `india = 250`
- `anglo_pop = 250`
- `kazakhstan = 250`
- `germany = 253`
- 总计 `1253`

当前主版已经具备：
- 统一 schema
- Gemini embeddings
- CultureMERT embeddings
- benchmark 对比
- ablation
- PAL 平台

因此它已经不是 early probe，而是一版 **paper-ready pilot main set**。

## 2. 我们现在真正要解决的不是“有没有数据”，而是“要不要继续升级主版”

这个决策要看三件事：

1. 新增域是否真的能提高论文说服力
2. 新增域是否会拖慢真人 PAL 和写作
3. 新增域是否足够稳定，不会在后面再次返工

## 3. 当前最合理的三种最终版本

### 方案 A：冻结当前 5 域，直接进入真人 PAL

构成：
- `china`
- `india`
- `anglo_pop`
- `kazakhstan`
- `germany`

优点：
- 最稳
- 所有主实验链路都已打通
- 不阻塞真人 PAL
- 最快进入论文写作

缺点：
- 跨文化广度已经不错，但仍然只是 `5` 域
- 如果 reviewer 特别在意文化覆盖，还可以再强一点

适用：
- 如果目标是最大化投稿稳定性

### 方案 B：加 1 个低风险域，再冻结

最推荐新增：
- `norway`

形成：
- `china`
- `india`
- `anglo_pop`
- `kazakhstan`
- `germany`
- `norway`

优点：
- Norway 已完成真实导入 probe
- 公开、许可清楚、接入成本低
- 可以把主版从 `5` 域提升到 `6` 域
- 对现有主线干扰最小

缺点：
- Norway 规模只有 `119`
- 文化说服力的提升是有的，但不如加入 Arab-Andalusian / Korea 那么强

适用：
- 如果目标是“小幅升级主版，但不想重新拖慢项目”

### 方案 C：加 1 个高价值域，再冻结

最推荐新增：
- `arab_andalusian`

形成：
- `china`
- `india`
- `anglo_pop`
- `kazakhstan`
- `germany`
- `arab_andalusian`

优点：
- 学术味道最强
- 对“跨文化推荐”叙事加分明显
- 文化跨度显著增加

缺点：
- 需要 Dunya API token
- 还没做本地导入 probe
- 比 Norway 更容易拖慢主线

适用：
- 如果愿意花一点时间换来更强的论文文化说服力

## 4. 当前不建议作为“主版升级第一优先”的域

### Korea

原因：
- 来源很好
- 但当前环境连 `data.go.kr` 都不稳定
- 在真正拿到 service key 并完成 sample audit 前，不应该拿它来阻塞主线

结论：
- 适合做 **高优先级扩展域**
- 不适合现在就当作“马上加入最终版”的域

### Georgia

原因：
- 文化域很强
- 但 Zenodo 已明确显示 files restricted

结论：
- 适合作为 **restricted-access 候选**
- 不适合现在直接当公开新增主域

## 5. 我最推荐的最终数据策略

### 如果以“尽快形成高质量最终版并进入真人 PAL”为目标

我推荐：

1. 先把当前 `5` 域主版当作稳定底座
2. 在此基础上只做一次可控升级
3. 升级后立即冻结

### 我的首选

**首选：方案 B**

也就是：
- 保留当前 `5` 域
- 增加 `norway`
- 然后冻结为最终 PAL 主版

原因：
- 升级幅度可见
- 风险最小
- 不会显著拖慢进度
- 可以把主版升级成 `6` 域，增强“跨文化系统”说服力

### 我的次选

**次选：方案 C**

也就是：
- 保留当前 `5` 域
- 增加 `arab_andalusian`
- 冻结

原因：
- 学术收益更高
- 但技术/访问成本也更高

## 6. 什么时候不该继续扩

一旦满足以下条件，就应该冻结数据集，不再扩：

- 主域数达到 `6`
- 总量超过 `1300`
- backbone 对比已跑通
- 真人 PAL 标注者已就位

达到这一步后，继续扩域的边际收益会明显下降，
而 PAL、回灌、写作的收益会更高。

## 7. 结论

当前阶段最关键的不是“还能不能继续搜到更多域”，
而是“要不要让数据集继续变化”。

我的建议是：

- **如果要升级，就只再加 1 个域**
- **优先加 Norway**
- **加完就冻结**

这样最适合把项目推进到：
- 真人 PAL
- 回灌重跑
- 论文写作
