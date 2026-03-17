# Echo / DCAS 项目简历摘要

更新时间：2026-03-15

## 一句话版本

设计并实现了一个面向跨文化音乐推荐的研究原型系统，完成了从公开音频数据接入、统一元数据治理、音乐 foundation embedding 构建、解纠缠推荐模型训练，到评测、消融、PAL 反馈回灌的完整研究链路。

## 简历项目描述（精简版）

### 版本 A

- 独立设计并实现跨文化音乐推荐研究原型 `DCAS`，构建了从公开音频数据导入、统一 metadata、foundation audio embedding 提取、推荐建模到评测与消融分析的完整实验流程。
- 基于 `CultureMERT` / 统一 embedding 底座，设计三因子解纠缠表示学习架构（内容 / 风格文化 / 情感功能），并结合领域对抗、最优传输和参与式主动学习（PAL）进行跨文化推荐。
- 搭建多文化域数据治理与实验基础设施，完成来源清单、许可审计、单域导入 probe、统一 schema 设计、item-level rights audit 和后续 Gemini embedding 迁移准备。

### 版本 B

- 负责跨文化音乐推荐项目从 0 到 1 的研究工程建设，完成公开音频数据接入、向量化表示构建、推荐模型训练、可复现评测与数据版本治理。
- 设计并实现 `DCAS` 推荐框架，在音乐 foundation embeddings 之上引入三因子解纠缠、领域对抗、OT 分布对齐和 PAL 反馈回灌，用于探索跨文化音乐推荐与文化域迁移问题。
- 主导 `research_dataset_v1 / v2` 数据路线，完成多文化域候选来源筛查、开源许可审计、Germany 条目级 rights 抽样审计、以及 China / India / Anglo-pop / Turkey 的实际导入 probe。

## 简历项目描述（偏技术版）

- 设计并实现 `DCAS`（Disentangled Cross-cultural Alignment System）研究原型：使用三因子 latent 表示 (`zc/zs/za`) 对音乐 embedding 进行结构化分解，并结合 domain adversarial learning、optimal transport ranking 和 pairwise constraint feedback 实现跨文化推荐。
- 构建了统一的公开音频数据处理流水线：支持 Hugging Face 音频数据导入、metadata 标准化合并、弱监督 interactions 合成、数据质检、切分和 embedding 构建。
- 基于 `CultureMERT` 完成音频 embedding 数据集生成，并进一步规划 / 搭建面向 `Gemini Embedding 2` 的统一向量数据库迁移方案。
- 实现推荐评测与研究脚本，包括 serendipity、cultural calibration、minority exposure、per-target-culture breakdown、bootstrap 置信区间、显著性比较和多种 ablation / baseline 实验。
- 推进 Germany / Japan / Turkey 等文化域的数据来源审计，结合 Europeana、Zenodo、Hugging Face 等平台完成来源规模、音频可得性和 license 合规性判断。

## 可直接放进简历的关键词

- 跨文化音乐推荐
- 音乐信息检索（MIR）
- 多文化音频数据治理
- Foundation audio embeddings
- CultureMERT
- Representation learning
- Disentangled learning
- Domain adversarial learning
- Optimal transport
- Active learning / PAL
- Dataset curation
- License audit
- Reproducible research pipeline

## 面试 / 申请时可重点展开的贡献点

- 从“想法”推进到“完整可运行研究链路”的工程能力
- 把复杂研究问题拆成数据、表示、推荐、反馈、评测五层
- 不只做模型，还同时做数据治理、来源审计和实验复现
- 能把研究原型进一步推进到可投稿的系统化实验平台
