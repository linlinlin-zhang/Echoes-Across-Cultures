# Gemini Embedding 2 处理平台说明

更新时间：2026-03-16

---

## 1. 目标

这条平台用于把 `research_dataset_v2` 的原始音频统一转换成：

- `tracks.npz`
- `tracks.npz.manifest.json`

从而直接接入 DCAS 的后续训练与评测流程。

当前默认推荐输入版本：

- [metadata_v2_main.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_v2_main.csv)

---

## 2. 新增代码

### 2.1 Gemini embedder

- [gemini_embedding2.py](E:/Desktop/Echo/dcas/embeddings/gemini_embedding2.py)

作用：

- 读取本地音频
- 混为 mono
- 按需要裁剪到 `max_seconds`
- 重采样到 `target_sample_rate`
- 编码为内联 `audio/wav`
- 调用 `gemini-embedding-2-preview:embedContent`
- 返回 `numpy.float32` embedding

### 2.2 构建脚本

- [build_tracks_with_gemini.py](E:/Desktop/Echo/dcas/scripts/build_tracks_with_gemini.py)

作用：

- 读取标准 `metadata.csv`
- 按 `track_id` 逐条生成 embedding
- 支持本地缓存与断点续跑
- 生成 `tracks.npz` 和 manifest
- 支持 `dry_run`

### 2.3 配置驱动入口

- [run_gemini_embedding_build.py](E:/Desktop/Echo/dcas/scripts/run_gemini_embedding_build.py)

### 2.4 配置样例

- [gemini_embedding2_v2_main.example.json](E:/Desktop/Echo/configs/embedding/gemini_embedding2_v2_main.example.json)

---

## 3. 当前实现选择

### 3.1 为什么先做单请求而不是批请求

当前平台优先追求：

- 稳定
- 可恢复
- 便于排查失败样本

因此当前实现使用逐条 `embedContent`，并通过本地缓存减少重复调用。

这意味着：

- 第一次全量构建会较慢
- 但失败后重跑成本很低
- 更适合当前研究型数据构建流程

后续如果 API 表现稳定，可以再加 `batchEmbedContents` 优化。

### 3.2 为什么统一转成 `audio/wav`

虽然官方示例直接展示了 `audio/mpeg` 内联嵌入，但我们这里统一转成 `audio/wav`，好处是：

- 不依赖原始文件编码差异
- 便于控制裁剪长度
- 便于控制采样率
- 便于后续缓存与稳定复现

---

## 4. 运行前准备

### 4.1 API Key 配置方式

当前支持三种方式，优先级从高到低：

1. 在 JSON 配置里直接写 `api_key`
2. 在 JSON 配置里写 `api_key_file`
3. 使用环境变量 `GEMINI_API_KEY`

更推荐的本地方式是：

- 复制 [gemini_embedding2_v2_main.local.example.json](E:/Desktop/Echo/configs/embedding/gemini_embedding2_v2_main.local.example.json)
- 重命名为你自己的 `.local.json`
- 再把 key 写进去，或者指向一个本地文本文件

例如：

```json
{
  "api_key": "your_key_here"
}
```

或者：

```json
{
  "api_key_file": "E:/Desktop/Echo/configs/embedding/gemini_api_key.local.txt"
}
```

这两类 `.local.*` 文件已经加入 `.gitignore`，默认不会被提交。

### 4.2 当前机器状态

本机已确认：

- `requests` 可用
- `numpy` 可用
- `torchaudio` 可用

当前机器之前缺的是 live key，本平台现在已经支持直接从配置文件读取，不再强依赖环境变量。

---

## 5. 推荐运行方式

### 5.1 先做 dry-run

```powershell
python E:\Desktop\Echo\dcas\scripts\build_tracks_with_gemini.py `
  --metadata E:\Desktop\Echo\storage\public\research_dataset_v2\metadata_v2_main.csv `
  --out E:\Desktop\Echo\storage\public\research_dataset_v2\tracks_gemini_embedding2_main.npz `
  --limit 5 `
  --dry_run `
  --skip_errors
```

### 5.2 用 JSON 配置运行

```powershell
python E:\Desktop\Echo\dcas\scripts\run_gemini_embedding_build.py `
  --config E:\Desktop\Echo\configs\embedding\gemini_embedding2_v2_main.example.json
```

如果你想把 key 放在本地文件里，更推荐这样：

```powershell
Copy-Item `
  E:\Desktop\Echo\configs\embedding\gemini_embedding2_v2_main.local.example.json `
  E:\Desktop\Echo\configs\embedding\gemini_embedding2_v2_main.local.json
```

然后把 `api_key` 或 `api_key_file` 改好，再运行：

```powershell
python E:\Desktop\Echo\dcas\scripts\run_gemini_embedding_build.py `
  --config E:\Desktop\Echo\configs\embedding\gemini_embedding2_v2_main.local.json
```

---

## 6. 关键参数建议

### `output_dimensionality`

当前建议：

- `768`

原因：

- 与现有 CultureMERT 版 `tracks.npz` 维度一致
- 后续替换 backbone 时更容易比较

### `max_seconds`

当前建议：

- `30.0`

原因：

- 足够保留较多音乐信息
- 同时能较稳地控制内联请求体大小

### `target_sample_rate`

当前建议：

- `16000`

原因：

- 与 Gemini 音频理解文档保持一致的保守做法
- 便于控制请求体积

---

## 7. 当前最推荐的数据版本

优先推荐：

- [metadata_v2_main.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_v2_main.csv)

原因：

- 当前主版已经改为 Kazakhstan 版本
- 总量 `1253` 条
- Kazakhstan 许可证据更清楚
- 更适合作为正式 Gemini embedding 构建输入

---

## 8. 官方依据

Google 官方博客说明：

- Gemini Embedding 2 是第一个原生多模态 embedding 模型
- 支持 text、image、video、audio、documents
- 推荐维度包括 `3072 / 1536 / 768`

来源：

- <https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/>

Gemini API 官方 embedding 文档说明：

- `gemini-embedding-2-preview` 支持直接嵌入音频
- 可使用 `embedContent` 或 `batchEmbedContents`
- 支持内联音频或 Files API

来源：

- <https://ai.google.dev/gemini-api/docs/embeddings>

Gemini 音频文档说明：

- 当总请求体超过 `20 MB` 时应使用 Files API
- 内联音频适合较小文件

来源：

- <https://ai.google.dev/gemini-api/docs/audio>

---

## 9. 当前建议

立即执行顺序：

1. 用当前主版跑 `dry_run`
2. 在本地 `.local.json` 或 `api_key_file` 中配置 key
3. 先跑 `limit=10` 的 live smoke test
4. 通过后再开始全量 embedding 构建
