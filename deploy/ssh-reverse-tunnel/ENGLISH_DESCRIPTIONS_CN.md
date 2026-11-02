# 英文音乐描述

中文原文保留在 `storage/public/merged/metadata_merged.csv`。
DeepSeek Flash 的完整英文翻译保存在独立字典
`storage/public/merged/descriptions_en.json`，结构如下：

```json
{
  "schema_version": 1,
  "prompt_version": 1,
  "model": "deepseek-flash",
  "updated_at": "...",
  "translations": {"中文原文": "English translation"}
}
```

按原文查找可在修改中文描述后避免误用旧译文，也可复用重复描述。
翻译覆盖 `description` 与非空的 `album_description`；不修改歌曲名称和中文字段。
曲库、推荐及收藏接口返回 `description_en` / `album_description_en`。
音乐页、收藏页切换到英语时优先显示英文；未翻译条目沿用既有回退行为。
服务器每次字典更新后会自动读取新版本，无需逐批重启服务。

## 私密配置

将 `configs/deepseek.env.example` 复制到 `configs/deepseek.env` 并填入 key。
私密配置已被 Git 忽略。key 只用于离线翻译任务，不发送到浏览器。
默认模型和 API 地址来自 [DeepSeek 官方文档](https://api-docs.deepseek.com/en/)。

## 翻译、续跑与检查

在仓库根目录执行：

```bash
python3 scripts/translate_descriptions_deepseek.py --dry-run
python3 scripts/translate_descriptions_deepseek.py --limit 12 --workers 1
python3 scripts/translate_descriptions_deepseek.py --workers 24 --batch-size 12
```

Windows 可将 `python3` 替换为 `.\.venv-gpu\Scripts\python.exe`。
全量任务共 30,009 条不同原文：30,000 条歌曲描述，以及 9 条专辑描述。
执行同一命令即可续跑，已保存的译文不会重新请求 API。
缓存不兼容的模型或提示词版本需要指定新的 `--output` 路径。

每批返回后校验 JSON、ID 完整性、英文非空和中文残留，失败则重试。
结果立即追加至 `descriptions_en.checkpoint.jsonl` 并刷新磁盘；
字典约每 15 秒原子更新一次，进程正常结束时再次保存。
多个进程不能同时写同一个字典。网络错误或余额错误导致任务退出后，
修复问题并重跑即可；中断时尚未收到的 API 请求可能已经计费。

本次运行日志：`tmp/description-translation/run.log`。
检查 `--dry-run` 的 `pending` 是否为 0，才表示当前原文已全部覆盖。
脚本只翻译输入文本，不调用网页应用的 AI 对话接口或推荐模型。

## 完成记录

2026-10-31 已生成并核对 30,009 条译文：30,000 首歌曲和 9 条专辑描述，
`--dry-run` 确认 `pending=0`，所有译文均非空且没有残留汉字。
覆盖率及字典 SHA-256 记录在 `tmp/description-translation/verification.json`。
原有中文 CSV 未被翻译脚本修改。刷新网页并切换到 English 即可使用。
线上曲目接口、20 条收藏及 3 条 GPU 推荐结果均已核对，英文描述与字典一致。
