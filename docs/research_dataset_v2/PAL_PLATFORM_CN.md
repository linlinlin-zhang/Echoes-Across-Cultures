# PAL Platform

这套 PAL 平台支持三种工作模式：

- `tasks_only`
  - 从当前模型导出高不确定样本对
  - 同时生成可直接发给人工标注的 CSV 表格
- `simulate_rounds`
  - 用 metadata label 自动生成模拟专家反馈
  - 连续跑两轮 `PAL -> constraints -> retrain -> evaluate`
- `real_round`
  - 把人工标注好的 CSV 转成约束
  - 回灌训练并与 baseline 模型比较

## 核心脚本

- 统一入口：
  - `E:/Desktop/Echo/dcas/scripts/run_pal_platform.py`
- 导出标注表：
  - `E:/Desktop/Echo/dcas/scripts/export_pal_annotation_sheet.py`
- 从人工标注构建约束：
  - `E:/Desktop/Echo/dcas/scripts/build_pal_constraints_from_annotations.py`
- 两轮模拟 PAL：
  - `E:/Desktop/Echo/dcas/scripts/run_phase3_pal.py`

## 任务选样

当前默认建议使用：

- `uncertainty_method = culture_centroid_entropy`

原因：

- 当前 `v2_main` 没有 affect labels
- 原来单纯依赖 `affect_head` 的 entropy 不够稳
- 新方法改为在 `za` 空间里看“文化质心分布熵”，更适合当前数据

可选值：

- `auto`
- `culture_centroid_entropy`
- `affect_entropy`
- `hybrid`

## 配置模板

- 任务导出：
  - `E:/Desktop/Echo/configs/pal/pal_v2_main_gemini_tasks.example.json`
- 模拟两轮：
  - `E:/Desktop/Echo/configs/pal/pal_v2_main_gemini_simulated.example.json`
- 真实标注回灌：
  - `E:/Desktop/Echo/configs/pal/pal_v2_main_gemini_real.example.json`

## 使用方式

### 1. 导出人工标注任务

```powershell
python -m dcas.scripts.run_pal_platform `
  --config E:\Desktop\Echo\configs\pal\pal_v2_main_gemini_tasks.example.json
```

输出：

- `tasks_round1.jsonl`
- `tasks_round1_annotation.csv`

其中 CSV 里会包含：

- `track_id_a / track_id_b`
- `culture_a / culture_b`
- `audio_path_a / audio_path_b`
- `uncertainty`
- 留给人工填写的：
  - `similar`
  - `rationale`
  - `annotator`
  - `notes`

### 2. 跑模拟 PAL

```powershell
python -m dcas.scripts.run_pal_platform `
  --config E:\Desktop\Echo\configs\pal\pal_v2_main_gemini_simulated.example.json
```

输出：

- `phase3_pal_summary.json`
- `phase3_pal_summary.md`
- 每轮的 `tasks / constraints / model / eval / compare`

### 3. 跑真实 PAL

先把 `tasks_round1_annotation.csv` 人工填好，保存成：

- `tasks_round1_annotation_filled.csv`

然后运行：

```powershell
python -m dcas.scripts.run_pal_platform `
  --config E:\Desktop\Echo\configs\pal\pal_v2_main_gemini_real.example.json
```

这一步会：

1. 把人工填写的 `similar / rationale` 转成 `constraints.jsonl`
2. 用这些约束重新训练模型
3. 自动评估 `baseline vs real PAL`
4. 输出比较报告

## 人工标注约定

建议在 `similar` 列只填写以下值之一：

- `yes`
- `no`

也支持：

- `true / false`
- `1 / 0`
- `similar / dissimilar`

留空会被跳过。

## 当前最推荐的工作流

1. 先跑 `simulate_rounds`
   - 验证 PAL 机制链路是否跑通
2. 再跑 `tasks_only`
   - 导出真正给专家/同学标注的任务表
3. 最后跑 `real_round`
   - 用真实标注做回灌训练和比较

## 备注

- 这套平台当前已经面向 `v2_main + Gemini + DCAS` 收敛
- 也可以切到 `CultureMERT`，只要换：
  - `tracks`
  - `baseline_model`

## Windows 备注

- 推荐使用 `python -m ...` 形式运行脚本，这样模块路径更稳。
- 标注表 CSV 以 `utf-8-sig` 写出；如果终端里中文显示乱码，优先以 Excel 或编辑器打开 CSV 检查实际内容。
