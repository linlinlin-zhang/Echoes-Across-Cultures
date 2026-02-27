# 第2点-子项1开始文档：CultureMERT 接入

时间：2026-02-27
负责人：Codex

## 目标
在当前 DCAS 原型中落地可执行的 CultureMERT 特征提取能力，使项目可从真实音频数据构建 `tracks.npz`，而非仅依赖 toy 生成数据。

## 范围
1. 新增 CultureMERT embedding 抽取模块
2. 新增数据集构建脚本（metadata.csv + audio 文件 -> tracks.npz）
3. 增加最小命令行用法与错误处理（模型下载失败、音频采样率不匹配等）

## 输入输出设计
输入：
- metadata.csv：`track_id,culture,audio_path[,affect_label]`

输出：
- tracks.npz：`track_id,culture,embedding,affect_label(optional)`

## 验收标准
- 模块可通过 `--help` 和导入测试
- 支持 `ntua-slp/CultureMERT-95M` 作为默认模型 ID（可覆盖）
- 对音频进行自动重采样并输出定长向量 embedding
