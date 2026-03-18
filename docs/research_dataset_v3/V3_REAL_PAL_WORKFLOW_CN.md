# Research Dataset V3 真人 PAL 操作手册

本文档对应 V3 主数据集：
- 主表：[metadata_v3_main.csv](E:/Desktop/Echo/storage/public/research_dataset_v3/metadata_v3_main.csv)
- 汇总：[summary_v3_main.json](E:/Desktop/Echo/storage/public/research_dataset_v3/summary_v3_main.json)

## 1. 本地音乐文件在哪里

V3 的音频已经在本地，主目录是：

`E:\Desktop\Echo\storage\public\research_dataset_v3`

每个文化域各自有一个 `audio` 子目录，例如：

- `E:\Desktop\Echo\storage\public\research_dataset_v3\china\audio`
- `E:\Desktop\Echo\storage\public\research_dataset_v3\india\audio`
- `E:\Desktop\Echo\storage\public\research_dataset_v3\turkey\audio`
- `E:\Desktop\Echo\storage\public\research_dataset_v3\indonesia\audio`
- `E:\Desktop\Echo\storage\public\research_dataset_v3\modern_english_pop\audio`
- `E:\Desktop\Echo\storage\public\research_dataset_v3\germany\audio`
- `E:\Desktop\Echo\storage\public\research_dataset_v3\france\audio`
- `E:\Desktop\Echo\storage\public\research_dataset_v3\italy\audio`
- `E:\Desktop\Echo\storage\public\research_dataset_v3\great_britain\audio`
- `E:\Desktop\Echo\storage\public\research_dataset_v3\russia\audio`

最准确的定位方式不是手动翻文件夹，而是直接看主表里的 `audio_path` 列。每一行都写了绝对路径。

## 2. 真人 PAL 前需要准备什么

最小必需件有三样：

1. `tracks.npz`
2. baseline model
3. PAL annotation sheet

其中：

- `tracks.npz` 由 embedding 构建得到
- baseline model 用 `tracks.npz` 训练得到
- PAL annotation sheet 由 baseline model 挑出高不确定样本后导出

如果你只是要收集真人 PAL 标注，不急着比较推荐指标，那么 `interactions_v3_main.csv` 不是前置必需。
如果你想在 PAL 前后做推荐评测对比，再补 `interactions_v3_main.csv`。

## 3. 第一步：构建 V3 embedding

已准备好的 V3 Gemini 配置模板：

- [gemini_embedding2_v3_main.example.json](E:/Desktop/Echo/configs/embedding/gemini_embedding2_v3_main.example.json)
- [gemini_embedding2_v3_main.local.example.json](E:/Desktop/Echo/configs/embedding/gemini_embedding2_v3_main.local.example.json)

建议先在本地复制一份 `local` 配置，再运行：

```powershell
Copy-Item `
  E:\Desktop\Echo\configs\embedding\gemini_embedding2_v3_main.local.example.json `
  E:\Desktop\Echo\configs\embedding\gemini_embedding2_v3_main.local.json
```

如果你更喜欢把 key 放在文件里，把 `api_key` 设为 `null`，并保留：

`"api_key_file": "E:/Desktop/Echo/configs/embedding/gemini_api_key.local.txt"`

运行 embedding：

```powershell
python E:\Desktop\Echo\dcas\scripts\run_gemini_embedding_build.py `
  --config E:\Desktop\Echo\configs\embedding\gemini_embedding2_v3_main.local.json
```

输出文件会是：

`E:\Desktop\Echo\storage\public\research_dataset_v3\tracks_gemini_embedding2_main.npz`

## 4. 第二步：训练 baseline model

有了 `tracks.npz` 之后，先训练一个 baseline model。PAL 不是随机抽样，它依赖模型先判断“自己最不确定哪些样本”。

```powershell
python -m dcas.cli.train `
  --data E:\Desktop\Echo\storage\public\research_dataset_v3\tracks_gemini_embedding2_main.npz `
  --out E:\Desktop\Echo\storage\models\dcas_full_v3_main_gemini.pt `
  --epochs 10 `
  --batch_size 128 `
  --lr 0.002
```

## 5. 第三步：如果需要推荐评测，再生成模拟交互

如果你只想做真人 PAL 采样，这一步可以先跳过。

如果你想在 PAL 前后比较推荐结果，可以先生成占位交互：

```powershell
python -m dcas.scripts.synthesize_interactions `
  --metadata E:\Desktop\Echo\storage\public\research_dataset_v3\metadata_v3_main.csv `
  --out E:\Desktop\Echo\storage\public\research_dataset_v3\interactions_v3_main.csv `
  --users_per_culture 20 `
  --tracks_per_user 50 `
  --genre_column label `
  --seed 42
```

## 6. 第四步：生成 PAL 候选池

已准备好的 V3 PAL 候选配置：

- [pal_v3_main_gemini_tasks.example.json](E:/Desktop/Echo/configs/pal/pal_v3_main_gemini_tasks.example.json)

建议本地复制为 `run` 配置后执行：

```powershell
Copy-Item `
  E:\Desktop\Echo\configs\pal\pal_v3_main_gemini_tasks.example.json `
  E:\Desktop\Echo\configs\pal\pal_v3_main_gemini_tasks.run.json
```

```powershell
python -m dcas.scripts.run_pal_platform `
  --config E:\Desktop\Echo\configs\pal\pal_v3_main_gemini_tasks.run.json
```

这个配置默认会生成 `1000` 对候选样本，输出到：

- `E:\Desktop\Echo\storage\pal\v3_main_gemini\candidates_1000.jsonl`
- `E:\Desktop\Echo\storage\pal\v3_main_gemini\candidates_1000_annotation.csv`

这里的 `1000` 对不是最终给人标的规模，而是候选池。

## 7. 第五步：从候选池中均衡抽样 200 对

不要直接用前 200 对，也不建议纯随机。推荐做法是：

- 先让模型按不确定性给出 1000 对候选
- 再按文化域均衡抽样到 200 对

已准备好的脚本：

- [select_pal_tasks_stratified.py](E:/Desktop/Echo/dcas/scripts/select_pal_tasks_stratified.py)

运行：

```powershell
python -m dcas.scripts.select_pal_tasks_stratified `
  --tasks E:\Desktop\Echo\storage\pal\v3_main_gemini\candidates_1000.jsonl `
  --out E:\Desktop\Echo\storage\pal\v3_main_gemini\tasks_200.jsonl `
  --n_total 200 `
  --group_field culture `
  --pool_multiplier 3 `
  --seed 42
```

这样得到的 `200` 对会在文化域之间尽量均衡，而不是被某一类样本包圆。

## 8. 第六步：导出最终真人标注表

```powershell
python -m dcas.scripts.export_pal_annotation_sheet `
  --tasks E:\Desktop\Echo\storage\pal\v3_main_gemini\tasks_200.jsonl `
  --metadata E:\Desktop\Echo\storage\public\research_dataset_v3\metadata_v3_main.csv `
  --out E:\Desktop\Echo\storage\pal\v3_main_gemini\tasks_200_annotation.csv
```

这个 CSV 就是给真人标注员用的主表。

## 9. 标注员具体怎么填

标注表关键列包括：

- `track_id_a`
- `track_id_b`
- `culture_a`
- `culture_b`
- `title_a`
- `title_b`
- `audio_path_a`
- `audio_path_b`
- `question`
- `similar`
- `rationale`
- `annotator`
- `notes`

推荐的标注流程：

1. 打开 `audio_path_a` 和 `audio_path_b`
2. 各听一次，如果有必要再回放一次
3. 只判断“情感功能 / 聆听意图是否相似”，不要按国家名直接猜
4. 在 `similar` 中填：
   - `yes` / `no`
   - 或 `1` / `0`
5. 在 `rationale` 中写一句短理由
6. 在 `annotator` 中写标注员代号
7. 若难以判断，可把 `similar` 留空，并在 `notes` 里写明原因

推荐的听音规范：

- 使用耳机
- 尽量在安静环境
- 音量固定，不要一对一对地乱调
- 优先比较“整体情绪功能”和“使用场景感”
- 不要因为语言、器乐名字或国家标签而直接下结论

## 10. 第七步：把真人标注转成约束

标注完成后，把文件另存为：

`E:\Desktop\Echo\storage\pal\v3_main_gemini\tasks_200_annotation_filled.csv`

然后执行：

```powershell
python -m dcas.scripts.build_pal_constraints_from_annotations `
  --annotations E:\Desktop\Echo\storage\pal\v3_main_gemini\tasks_200_annotation_filled.csv `
  --out E:\Desktop\Echo\storage\pal\v3_main_gemini\tasks_200_constraints.jsonl
```

## 11. 第八步：跑真人 PAL 回灌

已准备好的 V3 真人 PAL 配置：

- [pal_v3_main_gemini_real.example.json](E:/Desktop/Echo/configs/pal/pal_v3_main_gemini_real.example.json)

建议本地复制为 `run` 配置：

```powershell
Copy-Item `
  E:\Desktop\Echo\configs\pal\pal_v3_main_gemini_real.example.json `
  E:\Desktop\Echo\configs\pal\pal_v3_main_gemini_real.run.json
```

然后把 `annotations_csv` 改成你填好的文件路径，再运行：

```powershell
python -m dcas.scripts.run_pal_platform `
  --config E:\Desktop\Echo\configs\pal\pal_v3_main_gemini_real.run.json
```

这个阶段会做三件事：

1. 把真人标注转成 pairwise constraints
2. 训练带 PAL 约束的新模型
3. 如果 `interactions_v3_main.csv` 已准备好，再做 PAL 前后推荐效果对比

## 12. 最实用的建议

第一次做真人 PAL 时，不要一上来就发 200 对给标注员。

更稳的流程是：

1. 先抽 20 对做 pilot
2. 看标注员是否理解“情感功能相似”这个任务
3. 修一下说明文字
4. 再发完整 200 对

这样会比直接大规模开标更稳。
