# 数据集步骤-结束文档（第3阶段：公开数据接入与训练落地）

时间：2026-02-27  
负责人：Codex

## 已完成内容
1. 新增详细说明文档：
   - `docs/PUBLIC_DATASET_TRAINING_GUIDE.md`

2. 新增公开数据导入脚本（HuggingFace 音频数据）：
   - `dcas/scripts/import_hf_audio_dataset.py`
   - 产物：`audio/` + `metadata.csv` + `import_report.json`

3. 新增弱监督交互构建脚本：
   - `dcas/scripts/synthesize_interactions.py`
   - 产物：`interactions.csv`

4. 修复真实数据场景兼容性问题：
   - `dcas/embeddings/culturemert.py`
   - 修复点：当 `attention_mask` 与 `hidden_state` 长度不一致时，自动回退到无 mask 平均池化，避免 embedding 构建失败。

5. README 增补公开数据引入与训练范式说明。

## 本地实跑验证（公开数据小样本）

数据来源：
- HF：`sanchit-gandhi/gtzan`
- 导入规模：12 首（`--streaming --limit 12`）

执行结果：
1. 导入公开数据：
- imported: 12
- skipped: 0
- 输出：`storage/public/gtzan_demo/metadata.csv`

2. 生成弱交互：
- `n_rows=20`, `n_users=4`
- 输出：`storage/public/gtzan_demo/interactions.csv`

3. 构建 tracks（CultureMERT）：
- `n_tracks=12`, `dim=768`
- 输出：`storage/public/gtzan_demo/tracks.npz`

4. 数据检查与切分：
- `validate_dataset`: `status=warn`（单文化+小样本导致 warning，非错误）
- `make_splits`: `status=pass`

5. 训练：
- 命令：`python -m dcas.cli.train ... --epochs 2`
- 日志：`epoch=0 loss=0.4610`, `epoch=1 loss=0.2687`
- 模型输出：`storage/public/gtzan_demo/model.pt`

6. 推荐链路冒烟：
- 命令：`python -m dcas.cli.recommend ...`
- 成功返回推荐列表与指标。

## 结论
- “公开数据接入 -> 标准化 -> embedding 构建 -> 训练 -> 推荐”链路已可执行。
- 当前仍属于“冻结基座 embedding + 下游 DCAS 训练”，不是基座全量微调。
- 下一步应扩展到多文化域公开/半公开数据与 PAL 标注，以满足 ISMIR 级跨文化结论需求。
