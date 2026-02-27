# 第2点-子项1结束文档：CultureMERT 接入已完成

时间：2026-02-27
负责人：Codex

## 已完成内容
1. 新增 CultureMERT embedding 模块：
   - `dcas/embeddings/culturemert.py`
   - 默认模型 ID：`ntua-slp/CultureMERT-95M`
   - 支持自动重采样、单声道混合、时长截断、mean/cls 池化

2. 新增 embedding 接口导出：
   - `dcas/embeddings/__init__.py`

3. 新增真实数据构建脚本：
   - `dcas/scripts/build_tracks_from_audio.py`
   - 输入：`metadata.csv(track_id,culture,audio_path[,affect_label])`
   - 输出：`tracks.npz`

4. 后端 API 增强：
   - 新增 schema：`DatasetBuildRequest`
   - 新增接口：`POST /api/dataset/build_from_audio`
   - 对应实现：`dcas/pipelines.py` + `dcas_server/app.py`

5. 依赖更新：
   - `requirements.txt` 增加 `torchaudio`、`transformers`

## 验证记录
- `python -m dcas.scripts.build_tracks_from_audio --help` 通过
- `python -c "from dcas.embeddings import CultureMERTConfig, CultureMERTEmbedder; print('ok')"` 通过
- `python -c "from dcas_server.app import create_app; app=create_app(); print('routes', len(app.routes))"` 通过

## 输出能力
现在项目支持“上传 metadata + 音频路径 -> 生成真实 `tracks.npz`”，可替代 toy 数据构建流程。

## 残余风险
- 首次运行会下载 HuggingFace 模型权重，耗时受网络影响。
- 大规模音频建议离线批处理并缓存 embedding。
