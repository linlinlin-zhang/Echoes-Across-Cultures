# 波形级风格迁移前置要求与使用说明

更新时间：2026-02-27

## 1. 当前能力边界
- 已实现：波形级风格迁移基线（频谱统计迁移），可直接输出 `.wav`
- 已实现：CLI + API 接口
- 未实现：扩散/codec/vocoder 训练式高保真生成、主观听测协议

## 2. 前置要求
1. 依赖环境
- `torch`、`torchaudio` 已安装（见 `requirements.txt`）

2. 数据要求
- 元数据 CSV 需包含：`track_id`, `audio_path`
- `audio_path` 指向本地可读音频文件（wav/mp3/flac）

3. 质量建议
- 每个文化至少若干分钟音频素材
- 采样率建议可统一到 24kHz 或 44.1kHz
- 单条音频长度建议 >= 8 秒（短于 3 秒效果不稳定）

## 3. 前置检查命令

```bash
python -m dcas.scripts.check_waveform_generation_prereqs \
  --metadata ./storage/public/routeA_phase2_cn/metadata_merged.csv \
  --track_id_col track_id \
  --audio_col audio_path \
  --sample_size 30 \
  --out_json ./reports/waveform_prereq_check_cn.json
```

## 4. CLI 用法

方式 A：直接给音频路径

```bash
python -m dcas.cli.style_transfer_wave \
  --source_audio ./storage/public/routeA_phase2/gtzan/audio/sanchit-gandhi_gtzan_00000000.wav \
  --style_audio ./storage/public/routeA_phase2_cn/china/audio/ccmusic-database_erhu_playing_tech_00000000.wav \
  --out_wav ./storage/style/wave_transfer_demo.wav \
  --alpha 0.7 \
  --target_sr 24000 \
  --max_seconds 12 \
  --report_json ./reports/wave_transfer_demo.json
```

方式 B：给 metadata + track_id

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

## 5. API 用法

- 路由：`POST /api/style/transfer_waveform`
- 请求体：
  - `source_audio_path`
  - `style_audio_path`
  - `out_name`
  - `alpha`
  - `target_sr`
  - `n_fft` / `hop_length` / `win_length`
  - `max_seconds`
  - `peak_norm`

## 6. 指标解释
输出报告内包含：
- `spectral_centroid_src_hz`
- `spectral_centroid_style_hz`
- `spectral_centroid_out_hz`
- `style_alignment_gain_hz`（`src->style`距离缩短量，越大通常越好）

## 7. 下一步（论文级）
- 升级为可训练波形生成器（Diffusion/Codec LM）
- 增加客观指标（FAD/KL）和主观听测（MUSHRA/ABX）
- 建立跨文化风格迁移专用评测集与统计检验
