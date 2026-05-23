# 数据集阶段超详细执行说明（本地版，不入库）

文档日期：2026-03-01  
适用项目：Echo / DCAS  
文档用途：指导你从“当前原型数据集状态”升级到“可支撑 ISMIR 投稿证据链”的数据集阶段执行。  
说明：本文件为本地工作说明，不要求提交或推送。

---

## 0. 当前项目数据现状（基于仓库实况）

### 0.1 已有资产
- 你已经有完整的公开数据接入链路与脚本：
  - `dcas/scripts/import_hf_audio_dataset.py`
  - `dcas/scripts/merge_metadata.py`
  - `dcas/scripts/build_tracks_from_audio.py`
  - `dcas/scripts/synthesize_interactions.py`
  - `dcas/scripts/validate_dataset.py`
  - `dcas/scripts/make_splits.py`
- 你已经有四文化数据版本：`storage/public/routeA_phase2_cn`
- 该版本当前已知统计（从已有 report 提取）：
  - `n_tracks=640`
  - `n_cultures=4`（west/india/turkey/china）
  - `n_interactions=2682`
  - `n_users=80`
  - `validate status=warn`（主要因 `affect_label` 缺失与部分用户交互过低）
  - `split status=pass`

### 0.2 当前短板（数据阶段必须补）
- 短板 A：`affect_label` 覆盖不足，限制情感相关分析与辅助监督。
- 短板 B：交互日志多数为合成弱监督，论文中不可将其当成真实用户行为证据。
- 短板 C：跨文化数据域仍偏“可运行闭环”，距离“高可信 benchmark 级别”还有数据治理与标注质量差距。
- 短板 D：缺少严格的数据版本治理（每轮数据改动到结果变化的可追踪证据链）。

---

## 1. 数据集阶段总目标与成功标准

## 1.1 总目标
把当前“可运行数据闭环”升级为“可发表证据链数据闭环”，使你后续模型结果具备：
- 可复现性
- 可审计性
- 可比较性
- 可解释性

## 1.2 成功标准（建议作为阶段 Gate）
- Gate-1（可用）：数据可稳定构建、训练、评测，且无致命数据错误。
- Gate-2（可信）：分布报告、泄漏检测、重复样本控制、标签一致性检查齐备。
- Gate-3（可发表）：有明确数据卡（Data Card）、版本号、构建日志、协议说明、统计显著性方案。

## 1.3 为什么先做数据阶段
- 你的模型路线已进入“指标优化和结构比较”阶段，如果数据证据链薄弱，后续所有提升都可能被审稿人质疑为数据偏差或实验偶然性。
- 数据质量决定模型上限；在跨文化音乐任务中，数据治理重要性不低于模型结构创新。

---

## 2. 全阶段执行总览（建议顺序）

建议按 D0-D10 执行，不要并行跳步。

- D0：论文主张反推数据需求（先定义“要证明什么”）
- D1：数据源清单冻结与合规审查
- D2：导入与标准化（metadata schema 统一）
- D3：音频质量控制与重复样本治理
- D4：标签层治理（culture/label/affect）
- D5：交互层治理（真实日志/弱监督策略与边界）
- D6：数据门禁与切分门禁
- D7：嵌入构建与可追溯缓存
- D8：训练数据版本封版（dataset release candidate）
- D9：论文级实验矩阵与统计计划绑定
- D10：Data Card + Appendix 证据包

---

## 3. D0：论文主张反推数据需求

## 3.1 目标
先把“你想在论文里说什么”转成“数据必须满足什么”。

## 3.2 要做什么
- 列出最终主张（例如：跨文化推荐有效性、公平性、解纠缠可解释性、风格迁移可控性）。
- 为每个主张定义最小数据需求。

## 3.3 怎么做
在本地建一份 claim-data 对齐表（建议 CSV 或 md 表）：
- 列 1：主张
- 列 2：所需字段
- 列 3：所需样本量
- 列 4：必须的对照组
- 列 5：必须的统计检验

## 3.4 为什么
很多项目失败不是模型不行，而是数据字段在实验后期才发现不够，导致回炉重做。

## 3.5 原理
这是“反向设计（backward design）”：先定证据要求，再定数据工程。

## 3.6 验收标准
- 每个论文主张都有对应数据要求，不存在“无证据主张”。

## 3.7 风险与回退
- 风险：主张写太强，数据不可满足。
- 回退：把主张降级为“prototype evidence / preliminary evidence”。

---

## 4. D1：数据源清单冻结与合规审查

## 4.1 目标
形成“可复现的数据源 manifest”，并明确许可证与使用边界。

## 4.2 要做什么
- 冻结数据源（数据集名、版本、split、样本上限、下载日期）。
- 记录 license、是否允许研究用途、是否允许重分发音频。
- 明确文化映射策略（原标签 -> 文化域）。

## 4.3 怎么做
- 为每个数据源准备一条导入命令（建议统一收敛到 runbook）。
- 在每个源目录保留 `import_report.json`。
- 新建 `storage/public/dataset_manifest_v1.json`（本地即可），字段建议：
  - dataset_id
  - config/split
  - import_limit
  - culture_mode/culture_value/culture_map
  - label_column/affect_column
  - import_time
  - license_note

## 4.4 为什么
审稿人最常问：
- 你到底用了哪些数据？
- 不同人能否复现？
- 这些数据是否允许该用途？

## 4.5 原理
“数据可追溯性”是可复现科学基本要求：样本来源必须可回放。

## 4.6 验收标准
- 每个源有明确记录，不依赖口头记忆。
- 任意样本都能追溯回源数据集与源索引。

## 4.7 风险与回退
- 风险：部分公开集 license 不清晰。
- 回退：保留特征级统计、不分发原音频，论文中显式写明限制。

---

## 5. D2：导入与 metadata 标准化

## 5.1 目标
把多源异构数据统一到一个 schema，消除字段语义混乱。

## 5.2 要做什么
- 使用 `import_hf_audio_dataset.py` 导入每个来源。
- 用 `merge_metadata.py` 合并后形成统一 `metadata_merged.csv`。
- 定义强制字段与可选字段。

## 5.3 怎么做（建议 schema）
强制字段：
- `track_id`
- `culture`
- `audio_path`

推荐字段：
- `label`
- `affect_label`
- `source_dataset`
- `source_split`
- `source_index`
- `duration_sec`（可后处理补）
- `sample_rate`（可后处理补）

## 5.4 为什么
跨文化任务里最怕“同名字段异义”（例如 label 在不同源代表不同层级）。

## 5.5 原理
统一 schema 是“结构对齐（structural alignment）”；没有结构对齐就无法做可靠统计比较。

## 5.6 验收标准
- `metadata_merged.csv` 能被后续全部脚本直接消费。
- 不存在重复 `track_id`，不存在空 `audio_path`。

## 5.7 风险与回退
- 风险：不同数据源 `label` 粒度差太大。
- 回退：定义层级标签（coarse/fine），先用 coarse 做主实验。

---

## 6. D3：音频质量控制与重复样本治理

## 6.1 目标
把“数据可用”提升为“数据干净可分析”。

## 6.2 要做什么
- 检查音频损坏、时长异常、采样率异常、静音样本。
- 做重复样本检测（同源重复、跨源重复、近重复）。

## 6.3 怎么做
- 第一层：文件级校验
  - 能否读取
  - 时长是否在合理区间（例如 3s-120s）
- 第二层：信号级规则
  - RMS/峰值/静音占比
- 第三层：嵌入级近重复
  - 用 CultureMERT embedding 做近邻相似度（例如 cosine > 阈值）
  - 标记“重复簇”，保留一个代表样本

## 6.4 为什么
重复样本会抬高看似指标，尤其在 retrieval/recommendation 与 disentanglement probe 中会造成假提升。

## 6.5 原理
重复样本导致训练/测试分布泄漏（information leakage），使泛化估计偏乐观。

## 6.6 验收标准
- 重复比例有报告。
- 去重策略明确可复现（阈值、保留规则、日志）。

## 6.7 风险与回退
- 风险：阈值过严导致数据量骤降。
- 回退：保留“原始版”和“去重版”双版本，主表用去重版，附录给敏感性分析。

---

## 7. D4：标签层治理（culture/label/affect）

## 7.1 目标
提升标签可靠性，让 disentanglement 与解释性实验有可信监督信号。

## 7.2 要做什么
- 文化标签一致性校验：`culture` 是否与源映射一致。
- `label` 语义归一：建立 label mapping 字典（跨源统一）。
- `affect_label` 补全策略：
  - 可自动映射则映射
  - 不可映射则设缺失并记录

## 7.3 怎么做
- 建 `label_mapping_v1.json`：
  - 源标签 -> 统一标签
- 建 `affect_mapping_v1.json`：
  - 源情感标签 -> 整数类
- 对每轮 mapping 输出覆盖率统计：
  - 覆盖率
  - 冲突率
  - OOD 比例

## 7.4 为什么
你后续 MIG/DCI/SAP 与线性 probe 强依赖标签质量；标签噪声会直接污染“解纠缠证据”。

## 7.5 原理
监督评测是“标签-表示一致性”估计，标签噪声会导致互信息与可分性低估或失真。

## 7.6 验收标准
- `culture` 100% 非空、可解释。
- `label` 和 `affect_label` 覆盖率达预设阈值（建议 >70% 才做主分析）。

## 7.7 风险与回退
- 风险：不同文化域标签体系不可直接对齐。
- 回退：主实验只用 culture + 一组跨域可对齐共享因子（你已有 shared factors 管线）。

---

## 8. D5：交互层治理（真实日志 vs 弱监督）

## 8.1 目标
把推荐实验的证据边界写清楚，避免“弱监督日志被误解成真实用户行为”。

## 8.2 要做什么
- 明确交互来源类型：
  - Type-A：真实日志
  - Type-B：规则合成弱监督（当前多数是这类）
- 对 Type-B 进行多策略生成，减少单一规则偏置。

## 8.3 怎么做
- 你已有脚本：`synthesize_interactions.py`
- 建议扩展多策略版本（后续可加）：
  - S1：文化偏好 + 标签偏好
  - S2：仅文化偏好
  - S3：文化 + 长尾曝光偏置
- 每个策略独立输出 `interactions_{strategy}.csv`

## 8.4 为什么
单一规则交互会把实验变成“验证你写的规则”，而不是验证模型泛化。

## 8.5 原理
推荐评测受用户行为模型分布影响；单分布评测会高估特定 inductive bias。

## 8.6 验收标准
- 至少 2 套交互生成策略用于鲁棒性报告。
- 论文中明确“弱监督交互”的限制与用途。

## 8.7 风险与回退
- 风险：多策略会增加实验组合爆炸。
- 回退：主表保留 1 套，附录给策略敏感性。

---

## 9. D6：数据门禁与切分门禁（你已有脚本，但要升级使用方式）

## 9.1 目标
让每次数据构建都必须通过自动门禁后才允许训练。

## 9.2 要做什么
- 固化 `validate_dataset.py` 阈值与 `strict` 模式策略。
- 固化 `make_splits.py` 生成与泄漏检查。

## 9.3 怎么做
- 必跑：
  - `python -m dcas.scripts.validate_dataset ... --strict`
  - `python -m dcas.scripts.make_splits ... --strict`
- 推荐新增 CI 风格脚本：
  - `scripts/run_dataset_gate.ps1`（你可后续让我补）

## 9.4 为什么
没有“硬门禁”，数据质量会随实验迭代悄悄漂移，最后很难回溯问题来源。

## 9.5 原理
把软约束变成硬约束（quality as code），通过自动化保证实验输入稳定性。

## 9.6 验收标准
- 任一 fail 直接阻断训练。
- 所有 warning 有解释和处理记录。

## 9.7 风险与回退
- 风险：阈值设过严导致流程频繁阻断。
- 回退：先双阈值（hard fail + soft warn），逐步收紧。

---

## 10. D7：嵌入构建与缓存治理

## 10.1 目标
保证 embedding 构建可复现、可缓存、可重跑。

## 10.2 要做什么
- 固化 CultureMERT 参数：
  - `model_id`
  - `pooling`
  - `max_seconds`
  - `device`
- 保存 manifest（已由脚本支持）并纳入版本。

## 10.3 怎么做
- 使用：`build_tracks_from_audio.py`
- 每次构建都保存：
  - `tracks.npz`
  - `tracks.npz.manifest.json`
- 建议增加哈希：
  - metadata hash
  - audio file list hash

## 10.4 为什么
同一数据在不同 embedding 参数下会形成不同任务难度，不可混为同一“数据版本”。

## 10.5 原理
表示学习阶段等价于“特征提取器定义了观测空间”，观测空间变化即任务变化。

## 10.6 验收标准
- 任意实验可追溯到唯一 embedding manifest。
- 不存在“同名 tracks.npz 不同内容”情况。

## 10.7 风险与回退
- 风险：GPU/CPU 或库版本差异导致微小漂移。
- 回退：锁定依赖版本 + 记录环境指纹。

---

## 11. D8：数据版本封版（Dataset Release Candidate）

## 11.1 目标
形成可用于主实验的“封版数据包”。

## 11.2 要做什么
每个封版至少包含：
- `metadata_merged.csv`
- `interactions.csv`（或多策略）
- `tracks.npz`
- `tracks.npz.manifest.json`
- `dataset_profile.json/md`
- `split_report.json/md`
- `split_track_ids.json`
- `dataset_manifest_vX.json`

## 11.3 怎么做
建议目录：
- `storage/public/routeA_dataset_rc1/`
- `reports/routeA_dataset_rc1/`

命名规则建议：
- `rc1`, `rc2`, `final`
- 禁止覆盖历史版本

## 11.4 为什么
没有“封版”，结果不可被准确复现，论文数字可能在你自己机器上都漂移。

## 11.5 原理
实验科学需要固定输入对象；封版就是固定实验输入分布。

## 11.6 验收标准
- 模型主实验全部只使用同一封版数据。
- 变更数据时必须升版本，不得静默替换。

## 11.7 风险与回退
- 风险：封版过晚，导致之前实验不可追溯。
- 回退：立即从当前状态开始 `rc` 流程，即使后续继续迭代。

---

## 12. D9：与训练实验绑定（避免数据-模型错配）

## 12.1 目标
让每个训练 run 与数据版本一一对应。

## 12.2 要做什么
- 训练配置中强制写入：
  - dataset_version
  - tracks_path
  - split_id
  - interactions_strategy

## 12.3 怎么做
- 对每次训练输出增加 `run_manifest.json`（可后续让我补自动化）。
- `reports/*` 中保留数据版本字段。

## 12.4 为什么
后期会跑大量 ablation 与 baseline，没有绑定关系就无法解释差异来源。

## 12.5 原理
控制变量法：只有明确固定输入变量，模型差异解释才成立。

## 12.6 验收标准
- 任一结果表格都能追溯到数据版本。

## 12.7 风险与回退
- 风险：历史结果缺版本字段。
- 回退：做一次 retroactive mapping（补历史映射表）。

---

## 13. D10：论文证据包（Data Card + Appendix）

## 13.1 目标
把“数据做了什么”从工程记录变成审稿可读材料。

## 13.2 要做什么
写 Data Card，至少包含：
- 数据来源与许可
- 文化覆盖
- 标签定义
- 采样与筛选规则
- 去重规则
- 交互构建规则
- 偏差与限制
- 不可用声明（例如不能用于真实用户行为推断）

## 13.3 怎么做
在 `reports/` 生成：
- `dataset_card_rc1.md`
- `dataset_bias_and_limitations.md`

## 13.4 为什么
ISMIR 审稿越来越重视数据透明性和可复现实证路径。

## 13.5 原理
提高研究可信度：可解释的数据谱系比堆指标更有说服力。

## 13.6 验收标准
- 不依赖口头补充，评审仅看材料即可理解数据构建与限制。

## 13.7 风险与回退
- 风险：文档晚写导致细节遗失。
- 回退：每个 D 阶段结束即更新 Data Card 草稿。

---

## 14. 你现在可以直接执行的“第一轮数据阶段任务单”

建议按优先级执行（P0 -> P2）：

## 14.1 P0（本周必须完成）
- 任务 1：冻结 `routeA_phase2_cn` 为 `dataset_rc1`（仅复制，不改内容）。
- 任务 2：补 `dataset_manifest_rc1.json`（记录来源、参数、版本）。
- 任务 3：重跑 `validate_dataset --strict` 与 `make_splits --strict`。
- 任务 4：补齐 `affect_label` 可用性报告（至少覆盖率分析）。

## 14.2 P1（下周）
- 任务 5：做一次“去重敏感性”实验（原始 vs 去重版）。
- 任务 6：做两套交互策略（当前策略 + 对照策略）。
- 任务 7：重新跑 phase2/phase3 关键结果，比较稳定性。

## 14.3 P2（投稿前）
- 任务 8：形成 Data Card + 数据限制声明。
- 任务 9：将主表中所有数字绑定数据版本号。
- 任务 10：补充“弱监督交互边界”附录。

---

## 15. 每一步建议产物清单（执行即打勾）

- [ ] 数据源冻结清单（含许可证备注）
- [ ] 合并后 metadata 与 merge_report
- [ ] 质量门禁报告（profile）
- [ ] 切分报告（split + leakage）
- [ ] embedding manifest
- [ ] dataset version manifest
- [ ] interactions 策略说明
- [ ] 去重与标签覆盖率报告
- [ ] Data Card 草稿
- [ ] 数据版本-实验结果映射表

---

## 16. ISMIR 视角下最容易被质疑的问题（提前规避）

- Q1：交互是合成的，如何证明真实有效？
  - 回答策略：明确定位为“方法验证环境”，不夸大真实用户结论；补多策略鲁棒性。
- Q2：文化标签是否过于粗糙？
  - 回答策略：给出 coarse-grained 定义边界，并在附录中说明 finer taxonomy 规划。
- Q3：数据泄漏是否存在？
  - 回答策略：提供 split leakage report 与重复样本治理流程。
- Q4：结果是否依赖某一数据源偏置？
  - 回答策略：提供 per-source / leave-one-source-out 分析（后续可加）。

---

## 17. 原理总结：为什么这个阶段必须这么“重”

- 在跨文化音乐建模中，数据不是中立容器，而是模型行为的“隐式先验”。
- 你后续所有模型提升（TC/HSIC、三因子必要性、PAL 增益）都建立在数据分布稳定且可解释的前提上。
- 数据阶段做得越扎实，后面你在论文里就越能把“提升来自模型机制”而不是“偶然数据偏置”讲清楚。

---

## 18. 结合你当前仓库的最终建议（非常具体）

从今天开始，建议你就按下面这条主线推进：

1. 以 `routeA_phase2_cn` 复制出 `dataset_rc1`，并冻结。
2. 先不扩新模型，先补数据治理：
   - affect 覆盖率
   - 交互策略对照
   - 去重敏感性
3. 在 `dataset_rc1` 上重跑你已经做好的：
   - phase2 sensitivity
   - phase3 baseline comparison
   - eval suite
4. 只要数据版本固定，后面的任何改模结果都可直接进入论文证据链。

如果你愿意，下一步我可以直接帮你做 D1-D2 的落地文件：
- 生成 `dataset_manifest_rc1.json` 模板
- 生成一键门禁脚本（PowerShell）
- 生成 `dataset_card_rc1.md` 初稿

