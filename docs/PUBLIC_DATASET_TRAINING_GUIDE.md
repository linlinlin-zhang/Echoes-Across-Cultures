# DCAS 公开数据集与训练说明（详细版）

更新日期：2026-02-27

## 1) 先回答你的核心问题

### 这个项目现在是“微调训练”吗？
严格来说，**当前主流程不是“基座模型微调”**，而是两阶段训练：
- 阶段 A：用 CultureMERT（或其他音频基础模型）提取 `embedding`
- 阶段 B：在这些 embedding 上训练 DCAS（解纠缠 + 对抗 + OT）

这属于“**下游模型训练 / 任务适配**”，不是 end-to-end 微调整个基础模型。

### 如果要做“真正微调”，要不要选基座模型？
要。若你要微调基础模型，必须先选 backbone：
- 推荐首选：`ntua-slp/CultureMERT-95M`
- 对照基线：`m-a-p/MERT-v1-95M`

## 2) 建议的训练路线（面向 ISMIR）

### 路线 A（当前可稳定执行，建议先做）
- 冻结基座模型，只抽 embedding
- 训练 DCAS 主模型
- 用公开集做可复现 baseline + 消融

优点：工程风险低、实验迭代快、可快速建立可复现实验表。

### 路线 B（论文增强阶段）
- 在路线 A 稳定后，做基座模型轻量微调（例如 LoRA 或部分层解冻）
- 比较“冻结 vs 微调”在跨文化指标上的差异

优点：更接近你 PDF 中“文化适应基础模型”的主张。  
风险：计算开销大，对数据规模和标注质量敏感。

## 3) 公开数据集策略（结合你当前项目）

建议采用“**公开基准 + 定向补充**”：
- 公开可复现集：先接入 HF 音频公开集（本仓库已新增导入脚本）
- 文化增强集：后续补充 CultureMERT 论文涉及的数据域（如 Turkish/Carnatic/Hindustani 等）与 PAL 标注

原因：
- 仅用单一公开集很难支持“跨文化”强主张；
- 仅做自建数据又缺乏和社区工作的可比性。

## 4) 现已新增的数据接入能力

新增脚本：`dcas/scripts/import_hf_audio_dataset.py`

作用：从 HuggingFace 音频数据集导入本地音频，并生成标准 `metadata.csv`：
- 必备列：`track_id,culture,audio_path`
- 可选列：`label,affect_label,...`

新增脚本：`dcas/scripts/synthesize_interactions.py`

作用：当公开数据集没有用户日志时，生成弱监督 `interactions.csv`，用于推荐链路联调与早期实验（非最终论文主实验）。

## 5) 可直接执行的完整流程（命令）

### Step 1. 导入公开数据（示例：GTZAN on HF）
```bash
python -m dcas.scripts.import_hf_audio_dataset \
  --dataset sanchit-gandhi/gtzan \
  --split train \
  --streaming \
  --limit 200 \
  --out_dir ./storage/public/gtzan_raw \
  --culture_mode constant \
  --culture_value west \
  --label_column genre
```

### Step 2. 生成 embedding 格式训练集
```bash
python -m dcas.scripts.build_tracks_from_audio \
  --metadata ./storage/public/gtzan_raw/metadata.csv \
  --out ./storage/public/gtzan_raw/tracks.npz \
  --model_id ntua-slp/CultureMERT-95M \
  --pooling mean \
  --max_seconds 30
```

### Step 3. 构建弱交互（可选，但推荐）
```bash
python -m dcas.scripts.synthesize_interactions \
  --metadata ./storage/public/gtzan_raw/metadata.csv \
  --out ./storage/public/gtzan_raw/interactions.csv \
  --users_per_culture 20 \
  --tracks_per_user 50
```

### Step 4. 数据门禁检查
```bash
python -m dcas.scripts.validate_dataset \
  --tracks ./storage/public/gtzan_raw/tracks.npz \
  --interactions ./storage/public/gtzan_raw/interactions.csv \
  --out_json ./reports/dataset_profile_gtzan.json \
  --out_md ./reports/dataset_profile_gtzan.md
```

### Step 5. 切分与泄漏检测
```bash
python -m dcas.scripts.make_splits \
  --tracks ./storage/public/gtzan_raw/tracks.npz \
  --interactions ./storage/public/gtzan_raw/interactions.csv \
  --out_dir ./reports/splits_gtzan \
  --seed 42 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1
```

### Step 6. 训练 DCAS（当前主流程）
```bash
python -m dcas.cli.train \
  --data ./storage/public/gtzan_raw/tracks.npz \
  --out ./storage/public/gtzan_raw/model.pt \
  --epochs 20 \
  --batch_size 128 \
  --lr 2e-3
```

## 6) 论文层面的边界（必须明确）

- 你现在这条流程是“**基础模型特征抽取 + DCAS 下游训练**”，不是完整基座微调。
- 如果论文要强调“foundation model adaptation”，需要补做“冻结 vs 微调”的系统对照实验。
- 弱交互数据可用于工程联调和初步实验，不应作为最终用户行为结论的唯一证据。

## 7) 下一阶段建议（可直接排期）

1. 在 1-2 个公开集上先跑完整基线（冻结 embedding）。
2. 加入至少 2 个非西方文化域数据，完成跨文化实验最小闭环。
3. 再进入基座微调实验，做清晰的 ablation（冻结/部分解冻/LoRA）。
