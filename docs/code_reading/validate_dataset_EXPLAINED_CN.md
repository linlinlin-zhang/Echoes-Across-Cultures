# `validate_dataset.py` 代码说明

对应源码：[E:/Desktop/Echo/dcas/scripts/validate_dataset.py](E:/Desktop/Echo/dcas/scripts/validate_dataset.py)

## 1. 这个文件在数据链路里负责什么
这个脚本负责对最终可用于训练或评测的数据资产做“可用性验证”。

和 `audit_dataset_v4.py` 相比，它更靠近模型输入层，重点关注：

- `tracks.npz` 本身是否健康
- embedding 是否有非有限值或零向量
- 各文化域是否过度失衡
- `interactions.csv` 是否引用了不存在的 track
- 权重是否合法

所以它更像“训练前验收脚本”。

## 2. 顶部阈值类 `ValidationThresholds`
这个 dataclass 把验证用的阈值集中起来，包括：

- `min_tracks_per_culture`
- `max_culture_imbalance_ratio`
- `max_unknown_track_ratio`
- `max_duplicate_user_track_ratio`
- `max_zero_norm_ratio`
- `min_interactions_per_user`

它和审计脚本的阈值类似，但这里多了和 embedding 质量直接相关的：

- `max_zero_norm_ratio`

## 3. 辅助函数流程说明
这个文件的辅助函数可以按“基础数值工具 -> tracks 验证 -> interactions 验证 -> 报告输出”来记忆：

1. `_safe_float(...)`、`_round(...)`
   - 提供最基础的数值清洗与统一输出格式
2. `_norm_stats(...)`
   - 专门服务于 embedding 范数分布统计
3. `_validate_tracks(...)`
   - 先验证 `tracks.npz`
4. `_validate_interactions(...)`
   - 再验证 `interactions.csv`
5. `_to_markdown(...)`
   - 最后把验证结果变成人可读报告

所以这些辅助函数并不是散开的，它们构成了一条非常顺的验证链：

- 先准备数值工具
- 再验证表征资产
- 再验证监督资产
- 最后统一输出报告

## 4. 辅助函数说明
### `_safe_float(v)`
- 安全转浮点，非有限值返回 `None`

### `_round(v, ndigits=6)`
- 统一保留小数位，保证报告输出稳定

### `_norm_stats(emb)`
作用：

- 计算 embedding 向量范数的统计信息：
  - `min / p25 / p50 / p75 / max / mean / std`

意义：

- 可以快速发现 embedding 是否整体异常，例如全都很接近 0

### `_to_markdown(report)`
作用：

- 把 JSON 报告转成 Markdown
- 适合直接人读

## 5. `_validate_tracks(...)` 在检查什么
这是验证 `tracks.npz` 的核心函数。

### 4.1 读取 `tracks.npz`
使用 `load_tracks(...)` 读取：

- `track_id`
- `culture`
- `embedding`
- 可选的 `affect_label`

### 4.2 检查 embedding 是否为有限值
通过 `np.isfinite(emb)` 计算：

- `finite_embedding_ratio`

如果不是 1.0，就报 error。

### 4.3 检查 zero norm ratio
先计算每条 embedding 的 L2 norm，再统计有多少向量接近 0。

如果 `zero_norm_ratio > max_zero_norm_ratio`，就报 warn。

这个检查很实用，因为全零向量或近零向量通常意味着前面的特征提取出了问题。

### 4.4 检查重复 `track_id`
如果 `track_id` 重复，会报 error。

### 4.5 检查文化分布
它会统计：

- 每个文化域的条数和比例
- `culture_imbalance_ratio`

如果某文化域样本数太少或整体失衡太严重，就报 warn。

### 4.6 检查 `affect_label`
如果没有 `affect_label`：

- 不是错误
- 会报一个 info，说明这限制了 affect 相关评估

这体现了一个很好的设计：

- 区分真正错误和能力缺失

## 6. `_validate_interactions(...)` 在检查什么
这是交互层的验证函数。

它检查的点包括：

- 文件是否存在
- 是否有 `user_id / track_id`
- 是否为空
- 是否有缺失主键
- 是否有 unknown track
- 是否有无效或非正权重
- 是否有重复 `(user, track)` 对
- 用户交互数是否过低
- track 覆盖率是否足够

### 5.1 unknown track 为什么是关键指标
因为一旦 interactions 中引用了 `tracks.npz` 不存在的 `track_id`：

- 模型训练或评测就会出现不一致

所以这个指标比很多表面统计都更关键。

### 5.2 weight 检查的意义
它区分两类问题：

- `invalid_weight`
  - 不是有限数
- `non_positive_weight`
  - 小于等于 0

这很重要，因为推荐模型通常默认权重应为正。

## 7. 核心函数 `validate_dataset(...)`
### 参数解释
`tracks_path`

- `tracks.npz` 路径

`interactions_path`

- 可选的交互文件路径

`thresholds`

- 验证阈值配置

### 它的主流程
1. 检查 `tracks.npz` 是否存在
2. 调用 `_validate_tracks(...)`
3. 如有交互文件，再调用 `_validate_interactions(...)`
4. 汇总所有 issue
5. 根据 `error / warn` 数量给出总状态：
   - `fail`
   - `warn`
   - `pass`
6. 返回统一报告对象

## 8. `main()` 在做什么
`main()` 提供了命令行接口，支持：

- 指定 `--tracks`
- 可选传入 `--interactions`
- 可选输出 `--out_json`
- 可选输出 `--out_md`
- 开启 `--strict` 后，在状态为 `fail` 时退出码为 2

`--strict` 很适合用于自动化流水线或 CI。

## 9. 它和 `audit_dataset_v4.py` 的区别
这是面试里非常容易被追问的一点。

### `audit_dataset_v4.py`
更偏 metadata 治理与数据集审计：

- 字段覆盖率
- 缺失音频
- source confound
- schema 完整性

### `validate_dataset.py`
更偏最终可用资产验证：

- `tracks.npz` embedding 健康度
- interactions 是否和 tracks 对齐
- 权重是否合法
- 是否可以进入训练/评测

你可以把它们理解成：

- `audit` 更偏“数据集画像和风险分析”
- `validate` 更偏“交付前验收”

## 10. 设计优点
### 优点 1：把 tracks 和 interactions 一起验证
这能发现很多单独看任一文件时发现不了的问题。

### 优点 2：错误、警告、信息分级清楚
不会把所有异常混成一类。

### 优点 3：同时支持 JSON 和 Markdown 报告
既方便程序消费，也方便人工阅读。

## 11. 局限与注意点
### 局限 1：不检查更深层语义正确性
它能发现结构错误和数值异常，但不能证明 embedding 语义质量一定好。

### 局限 2：阈值仍然需要结合任务理解
例如文化失衡阈值在不同研究里可能不同。

## 12. 面试时怎么讲
可以这样说：

“`validate_dataset.py` 是项目最终数据资产的验收脚本。它从模型输入层出发，检查 `tracks.npz` 的 embedding 是否有限、是否存在零向量、是否有重复 track_id，以及文化域分布是否失衡；同时还验证 `interactions.csv` 是否引用了未知 track、是否有非法权重和重复 `(user, track)` 对。与偏治理视角的 `audit_dataset_v4.py` 不同，它更像训练和评测前的最终质量门禁。”

## 13. 建议重点看哪里
优先看：

1. `ValidationThresholds`
2. `_validate_tracks(...)`
3. `_validate_interactions(...)`
4. `status = fail / warn / pass` 的判定逻辑

把这几部分掌握住，你就能把“最终验收层”讲得比较完整。
