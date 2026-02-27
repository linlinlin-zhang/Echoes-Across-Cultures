# DCAS：深度文化对齐音乐推荐（可运行原型）

该原型把你描述的系统拆成四条可落地的“工程主线”：

- 表征学习：把曲目嵌入解纠缠为内容/风格/情感三因子（z_c, z_s, z_a）
- 跨文化对齐：用领域对抗让 z_a 去文化化，并在 z_a 上做 OT（Sinkhorn）对齐推荐
- 参与式主动学习：用不确定性采样挑样本给专家，并把专家的成对约束回灌训练
- 评估：用 Serendipity（意外性×相关性）与文化公平性指标评估，而不只看准确率

## 快速开始（玩具数据）

1) 安装依赖

```bash
python -m pip install -r requirements.txt
```

2) 生成玩具数据

```bash
python -m dcas.scripts.make_toy_data --out ./toy
```

3) 训练解纠缠模型

```bash
python -m dcas.cli.train --data ./toy/tracks.npz --out ./toy/model.pt
```

4) 做跨文化推荐（示例：用户 u0，从 culture=west 推荐到 culture=india）

```bash
python -m dcas.cli.recommend --model ./toy/model.pt --tracks ./toy/tracks.npz --interactions ./toy/interactions.csv --user u0 --target_culture india --k 10
```

5) 运行 PAL 选样（把高不确定样本输出到 JSONL，供专家标注/给出成对约束）

```bash
python -m dcas.cli.pal_loop --model ./toy/model.pt --tracks ./toy/tracks.npz --out ./toy/pal_tasks.jsonl --n 50
```

## 全栈控制台（现代前端 + API）

一键脚本（Windows PowerShell）：

```powershell
.\build.ps1
.\dev.ps1
```

1) 安装后端依赖

```bash
python -m pip install -r requirements.txt
```

2) 安装前端依赖

```bash
cd web
npm install
```

3) 启动后端（FastAPI）

```bash
python -m dcas_server
```

4) 启动前端（Vite）

```bash
cd web
npm run dev -- --host 0.0.0.0 --port 5173
```

5) 打开控制台

- http://localhost:5173/
- API 文档：http://localhost:8000/docs

## 数据格式

tracks.npz 至少包含：

- track_id: (N,) 字符串
- culture: (N,) 字符串（域标签，用于对抗去文化化）
- embedding: (N, D) float32（可来自任何音频基础模型，如 MERT/CultureMERT/CLAP 等）
- affect_label: (N,) int（可选，用于评估/训练一个轻量情感头；真实系统可来自 GlobalMood 或弱监督）

interactions.csv：

- user_id,track_id,weight

## 新增能力（2026-02-27）

1) CultureMERT 接入（真实音频 -> tracks.npz）

```bash
python -m dcas.scripts.build_tracks_from_audio \
  --metadata ./storage/uploads/metadata.csv \
  --out ./storage/datasets/tracks.npz \
  --model_id ntua-slp/CultureMERT-95M
```

说明：
- `metadata.csv` 必需列：`track_id,culture,audio_path`，可选列：`affect_label`
- `audio_path` 可为绝对路径，或相对 `metadata.csv` 的路径
- 同目录会生成 `tracks.npz.manifest.json`，记录模型、参数和错误清单，便于复现实验

数据集质量门禁（推荐在训练前执行）：

```bash
python -m dcas.scripts.validate_dataset \
  --tracks ./storage/datasets/tracks.npz \
  --interactions ./storage/datasets/interactions.csv \
  --out_json ./reports/dataset_profile.json \
  --out_md ./reports/dataset_profile.md
```

数据集切分（分层 + 防泄漏检查）：

```bash
python -m dcas.scripts.make_splits \
  --tracks ./storage/datasets/tracks.npz \
  --interactions ./storage/datasets/interactions.csv \
  --out_dir ./reports/splits \
  --seed 42 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1
```

2) 生成式风格迁移（embedding 级反事实）

```bash
python -m dcas.cli.style_transfer \
  --model ./toy/model.pt \
  --tracks ./toy/tracks.npz \
  --source_track t00001 \
  --style_track t00002 \
  --out ./toy/style_transfer.npz \
  --k 10
```

说明：
- 机制：`zc(source) + zs(style) + za(source)` 解码生成 embedding
- 输出包含最近邻候选，可用于解释推荐

3) 动态本体（概念/关系/标注）

```bash
python -m dcas.cli.ontology --state ./storage/ontology/state.json add-concept --name Han --description "Korean culturally grounded sorrow"
python -m dcas.cli.ontology --state ./storage/ontology/state.json suggest --query "sorrow grief"
python -m dcas.cli.ontology --state ./storage/ontology/state.json export-constraints --out ./storage/ontology/constraints.jsonl
```

API 同步提供：
- `GET /api/ontology/state`
- `POST /api/ontology/concepts`
- `POST /api/ontology/relations`
- `POST /api/ontology/annotations`
- `POST /api/ontology/suggest`
- `POST /api/ontology/export_constraints`

## 路线图（接入真实音频）

- 把 embedding 替换为 CultureMERT 的帧级/段级表示
- 增加内容一致性：同曲目增广（pitch shift/EQ）后 z_c 对比学习
- 生成模块：用 z_c/z_s 做风格迁移（此原型仅保留接口，便于后续接入扩散/编解码器）

## 论文主张边界（重要）

为避免“论文主张超出实现证据”的风险，本仓库提供主张对齐矩阵：

- [docs/PAPER_CLAIM_ALIGNMENT.md](docs/PAPER_CLAIM_ALIGNMENT.md)

当前状态摘要：

- 已实现：解纠缠三因子、领域对抗、OT 推荐、PAL 选样、pairwise constraints 回灌、全栈原型闭环
- 部分实现：解纠缠正则（当前是 KL + 协方差去相关近似，尚非完整 TC/FactorVAE 级别）
- 计划中：CultureMERT 持续预训练接入、风格迁移生成模块、MIG/DCI/SAP 等标准解纠缠评测


## Public Dataset Bootstrap (2026-02-27)

Detailed guide:
- [docs/PUBLIC_DATASET_TRAINING_GUIDE.md](docs/PUBLIC_DATASET_TRAINING_GUIDE.md)

Import public audio dataset from HuggingFace to local metadata/audio:
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

Build weak interactions when user logs are not available:
```bash
python -m dcas.scripts.synthesize_interactions \
  --metadata ./storage/public/gtzan_raw/metadata.csv \
  --out ./storage/public/gtzan_raw/interactions.csv
```

Training paradigm note:
- Current default pipeline is **not full backbone fine-tuning**.
- It is `foundation embedding extraction + DCAS downstream training`.
- If backbone fine-tuning is required, select a base model first (recommended: `ntua-slp/CultureMERT-95M`).
## Route A Pilot (Phase 1)

Runbook:
- [docs/ROUTE_A_PHASE1_RUNBOOK.md](docs/ROUTE_A_PHASE1_RUNBOOK.md)

New scripts:
- `python -m dcas.scripts.merge_metadata --help`
- `python -m dcas.scripts.evaluate_recommender --help`
## Route A Phase 2

- [docs/ROUTE_A_PHASE2_RUNBOOK.md](docs/ROUTE_A_PHASE2_RUNBOOK.md)
- [docs/technical_notes/routeA_phase2_end.md](docs/technical_notes/routeA_phase2_end.md)
## Waveform Style Transfer (Baseline)

Prereq check:
```bash
python -m dcas.scripts.check_waveform_generation_prereqs \
  --metadata ./storage/public/routeA_phase2_cn/metadata_merged.csv \
  --out_json ./reports/waveform_prereq_check_cn.json
```

Generate waveform from metadata track ids:
```bash
python -m dcas.cli.style_transfer_wave \
  --metadata ./storage/public/routeA_phase2_cn/metadata_merged.csv \
  --source_track sanchit-gandhi_gtzan_00000000 \
  --style_track ccmusic-database_erhu_playing_tech_00000000 \
  --out_wav ./storage/style/wave_transfer_demo_from_metadata.wav \
  --alpha 0.7 \
  --target_sr 24000 \
  --max_seconds 12 \
  --report_json ./reports/wave_transfer_demo_from_metadata.json
```

Notes:
- This is a waveform-level spectral statistics transfer baseline.
- It is not yet a diffusion/codec-trained high-fidelity generator.

## Disentanglement Upgrade (Shared Factors + Multi-Seed)

Build cross-cultural shared acoustic factors (reduce culture-label leakage):
```bash
python -m dcas.scripts.build_shared_acoustic_factors \
  --metadata ./storage/public/routeA_phase2_cn/metadata_merged.csv \
  --out_csv ./reports/routeA_shared_factors_cn.csv \
  --report_json ./reports/routeA_shared_factors_cn_report.json \
  --target_sr 16000 \
  --max_seconds 10 \
  --n_bins 3
```

Evaluate MIG/DCI/SAP with multi-seed protocol:
```bash
python -m dcas.scripts.evaluate_disentanglement \
  --model ./storage/public/routeA_phase2_cn/model.pt \
  --tracks ./storage/public/routeA_phase2_cn/tracks.npz \
  --metadata ./reports/routeA_shared_factors_cn.csv \
  --factors factor_energy,factor_brightness,factor_texture,factor_dynamics,factor_percussiveness \
  --seeds 42,43,44,45,46 \
  --out_json ./reports/routeA_disentanglement_phase2_cn_sharedfactors.json \
  --out_md ./reports/routeA_disentanglement_phase2_cn_sharedfactors.md
```

## Disentanglement Model Upgrade (TC + HSIC)

Train upgraded model:
```bash
python -m dcas.cli.train \
  --data ./storage/public/routeA_phase2_cn/tracks.npz \
  --out ./storage/public/routeA_phase4_disentangle_cn/model_tc_hsic.pt \
  --epochs 10 \
  --batch_size 64 \
  --lr 2e-3 \
  --lambda_domain 0.5 \
  --lambda_contrast 0.2 \
  --lambda_cov 0.05 \
  --lambda_tc 0.1 \
  --lambda_hsic 0.05 \
  --regularizer_warmup_epochs 3
```

Evaluate (shared factors):
```bash
python -m dcas.scripts.evaluate_disentanglement \
  --model ./storage/public/routeA_phase4_disentangle_cn/model_tc_hsic.pt \
  --tracks ./storage/public/routeA_phase2_cn/tracks.npz \
  --metadata ./reports/routeA_shared_factors_cn.csv \
  --factors factor_energy,factor_brightness,factor_texture,factor_dynamics,factor_percussiveness \
  --seeds 42,43,44,45,46 \
  --out_json ./reports/routeA_disentanglement_phase4_tc_hsic_cn_sharedfactors.json \
  --out_md ./reports/routeA_disentanglement_phase4_tc_hsic_cn_sharedfactors.md
```

## Unified Eval Suite

Run recommender/disentanglement/significance in one command:
```bash
python -m dcas.cli.eval \
  --model ./storage/public/routeA_phase4_disentangle_cn/model_tc_hsic.pt \
  --baseline_model ./storage/public/routeA_phase2_cn/model.pt \
  --tracks ./storage/public/routeA_phase2_cn/tracks.npz \
  --interactions ./storage/public/routeA_phase2_cn/interactions.csv \
  --metadata ./reports/routeA_shared_factors_cn.csv \
  --factors factor_energy,factor_brightness,factor_texture,factor_dynamics,factor_percussiveness \
  --seeds 42,43,44,45,46 \
  --out_dir ./reports/eval_suite_phase4_v2 \
  --method ot \
  --k 10 \
  --epsilon 0.1 \
  --iters 200 \
  --bootstrap_samples 5000 \
  --permutation_samples 5000
```

See detailed guide: `docs/EVAL_SUITE_GUIDE.md`.

Calibration metric note:
- `cultural_calibration_kl` now uses a smoothed target-culture prior in style latent (`zs`) space, avoiding the previous fixed-value degeneration under single-target recommendation.
- `cultural_calibration_kl_legacy` is kept for backward compatibility with historical reports.
