# 深度解纠缠模型增强结束文档（TC + HSIC + Warmup）

时间：2026-02-27  
负责人：Codex

## 本轮实现
1. 损失函数增强
- 文件：`dcas/models/losses.py`
- 新增：
  - `gaussian_total_correlation`
  - `hsic_rbf`

2. 模型配置增强
- 文件：`dcas/models/dcas_vae.py`
- 新增超参：
  - `lambda_tc`
  - `lambda_hsic`
- 前向接口支持正则缩放：`reg_scales`

3. 训练入口增强
- 文件：`dcas/cli/train.py`, `dcas/pipelines.py`
- 新增参数：
  - `--lambda_tc`
  - `--lambda_hsic`
  - `--regularizer_warmup_epochs`

4. API 参数同步
- 文件：`dcas_server/schemas.py`, `dcas_server/app.py`
- `TrainRequest` 与 `/api/train` 已支持新超参

## 实验设置
- 数据：`storage/public/routeA_phase2_cn/tracks.npz`
- 新模型：`storage/public/routeA_phase4_disentangle_cn/model_tc_hsic.pt`
- 训练参数：
  - epochs=10, batch_size=64, lr=2e-3
  - lambda_domain=0.5
  - lambda_contrast=0.2
  - lambda_cov=0.05
  - lambda_tc=0.1
  - lambda_hsic=0.05
  - warmup_epochs=3

## 评测结果
1. 推荐指标
- `phase2 serendipity = 0.8109801431`
- `phase4_tc_hsic serendipity = 0.8726107828`
- 变化：`+0.0616306397`
- 文件：`reports/routeA_recommender_compare_phase4_cn.md`

2. 解纠缠（shared factors, multi-seed）
- 文件：
  - `reports/routeA_disentanglement_phase4_tc_hsic_cn_sharedfactors.json`
  - `reports/routeA_disentanglement_sharedfactors_compare_phase4_cn.md`

3. 关键观察
- `za` 的 `DCI_disentanglement` 与 `DCI_completeness` 相对 phase2 提升
- 指标仍存在 trade-off（并非所有空间/指标同时提升）
- 说明模型已改善，但仍未达到“强解纠缠全面领先”可宣称状态

## 结论
- 本轮改造有效提升了模型可用性与部分解纠缠性质。
- 下一步应继续：
  1. 将 TC/HSIC 权重做系统化网格搜索
  2. 对 MIG/DCI/SAP 增加 bootstrap 置信区间
  3. 增加显著性检验后再形成论文主表结论
