# Domain Expansion Execution Audit (2026-03-17)

本文档记录对四个高优先级扩展域的“执行层”尝试结果，而不只是搜索结论。

本轮重点推进的四个域：
- `korea`
- `arab_andalusian`
- `georgia`
- `norway`

目标不是立刻把四个域全部导入，而是回答四个更实际的问题：
- 能不能真实访问到音频或音频清单
- 许可条件是否足够清楚
- 当前环境里最大的阻塞点是什么
- 哪些域最值得下一步正式接入

## 1. Norway

来源：
- <https://huggingface.co/datasets/Bots4M/HF2-Hardanger-fiddle-dataset>

执行结果：
- 已直接命中 HF dataset API，确认：
  - `license: cc-by-4.0`
  - `format: audiofolder`
  - `modality: audio`
  - `119` audio-MIDI pairs
- 已下载并解析：
  - `data/manifests/manifest.csv`
- 已完成真实小样本导入 probe：
  - 输出目录：[norway_probe](E:/Desktop/Echo/storage/public/research_dataset_v2/norway_probe)
  - 元数据：[metadata.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/norway_probe/metadata.csv)
  - 报告：[import_report.json](E:/Desktop/Echo/storage/public/research_dataset_v2/norway_probe/import_report.json)

结论：
- `norway` 是这四个域里最容易直接接进现有流水线的
- 不是候选概念，而是已经完成了真实导入 probe
- 如果我们需要一个低摩擦新增域，它是最稳的选择之一

## 2. Arab-Andalusian

来源：
- <https://compmusic.upf.edu/corpora>
- <https://dunya.compmusic.upf.edu/developers/>
- <https://dunya.compmusic.upf.edu/andalusian/info>
- <https://dunya.compmusic.upf.edu/docs/andalusian.html>

执行结果：
- 已确认 Dunya developers 页面中的关键说明：
  - `Arab Andalusian` 行显示 `Metadata=open, Audio=open`
  - 同页说明：`open` 项需要 **API token**
  - `restricted` 项则需要额外请求
- 已用匿名请求直接测试 API：
  - `https://dunya.compmusic.upf.edu/api/andalusian/recording`
  - 返回 `401 Authentication credentials were not provided.`
- 已确认 API 文档中存在 `download_mp3(recordingid, location)` 接口说明

结论：
- 这条线不是“完全匿名公开下载”
- 它的真实状态是：
  - **音频开放给有 API token 的普通用户**
  - **不需要额外学术审批**
- 因此它依然是非常强的候选，但下一步必须先创建 Dunya API token

## 3. Georgia

来源：
- <https://zenodo.org/records/6900390>
- <https://transactions.ismir.net/articles/10.5334/tismir.44>
- <https://doc.dezrann.net/status>

执行结果：
- 已从 Zenodo API 取回公开 record metadata
- 已确认描述文本明确写到：
  - 传统 Georgian vocal music
  - 已获 Tbilisi State Conservatoire 许可发布录音
  - 许可为 `CC-BY-NC-4.0`
- 但进一步检查 `export/json` 发现：
  - `"status": "restricted"`
  - `"files": "restricted"`

结论：
- `georgia` 不是“没数据”
- 也不是“版权不清楚”
- 它的关键问题是：
  - **文件访问是受限的**
- 所以当前应把它视为一个 **高价值 restricted-access candidate**
- 不应再把它误认为“和 Norway 一样能直接公开批量下载”的域

## 4. Korea

来源：
- <https://www.data.go.kr/data/15098241/fileData.do>
- <https://www.data.go.kr/data/15098387/openapi.do>
- <https://www.data.go.kr/data/15094324/openapi.do>
- <https://www.data.go.kr/data/15142755/fileData.do>
- <https://www.data.go.kr/data/15142756/fileData.do>
- <https://www.data.go.kr/data/15142758/fileData.do>

执行结果：
- 已多次尝试：
  - Python `requests`
  - PowerShell `Invoke-WebRequest`
  - 浏览器自动化
  - `curl.exe`
- 当前环境下统一结果是：
  - `SSL/TLS handshake failed`
  - 或 `ERR_CONNECTION_CLOSED`

结论：
- Korea 目前最主要的阻塞，不是我们没有候选源
- 而是：
  - **当前执行环境无法稳定访问 `data.go.kr`**
- 因此 Korea 的下一步不是继续盲试脚本，而是：
  - 在能正常访问 `data.go.kr` 的网络环境下
  - 申请 `service key`
  - 再做首次 sample audit

## 5. 当前优先级重排

如果按“离真实接入还有多远”排序，这四个域现在应当这样看：

1. `norway`
   - 已完成真实导入 probe
   - 随时可以扩成正式域
2. `arab_andalusian`
   - 访问路径清楚
   - 只差创建 Dunya API token
3. `korea`
   - 来源强
   - 但当前环境网络阻塞
4. `georgia`
   - 数据和许可有研究价值
   - 但文件访问本身受限

## 6. 建议

如果下一步只想推进最值钱的动作：

1. 先把 `norway` 作为低摩擦新增域
2. 同时为 `arab_andalusian` 创建 Dunya API token 并做 10-track probe
3. `korea` 留到能正常访问 `data.go.kr` 的环境再继续
4. `georgia` 保留为 restricted-access 候选，不作为当前优先接入域
