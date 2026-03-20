# V4 真人 PAL 准备与执行手册

这份手册对应当前最值得投入真人 PAL 的主线：

- 数据集：`V4 main`
- backbone：`CultureMERT mw3`
- baseline model：`dcas_full_v4_main_culturemert_stage3.pt`

当前目标不是马上回灌训练，而是先把真人 PAL 前的准备工作完全做实，让标注员一到就能开工。

## 1. 现在还需要做什么

在等待真人 PAL 的这段时间里，最值得做的不是继续盲目加模型，而是把以下内容准备好：

1. 生成高不确定性候选池
2. 从候选池里均衡抽出 pilot 标注集
3. 再准备正式首轮标注集
4. 固定真人 PAL 回灌配置，避免标注回来后临时拼流程

这些工作现在都已经有自动化入口。

## 2. 一键准备脚本

新增脚本：

- `dcas/scripts/prepare_real_pal_bundle.py`

对应配置：

- `configs/pal/pal_v4_main_culturemert_prepare.example.json`
- `configs/pal/pal_v4_main_culturemert_prepare.run.json`

运行：

```powershell
E:\Desktop\Echo\.venv-gpu\Scripts\python.exe -m dcas.scripts.prepare_real_pal_bundle `
  --config E:\Desktop\Echo\configs\pal\pal_v4_main_culturemert_prepare.run.json
```

默认会在下面这个目录生成整套真人 PAL 准备包：

- `E:\Desktop\Echo\storage\pal\v4_main_culturemert_real`

## 3. 准备包里会有什么

准备包默认包含：

- `candidates_1000.jsonl`
- `candidates_1000_annotation.csv`
- `pilot_tasks_20.jsonl`
- `pilot_tasks_20_annotation.csv`
- `tasks_round1_200.jsonl`
- `tasks_round1_200_annotation.csv`
- `bundle_manifest.json`
- `README.md`

含义分别是：

- `candidates_1000`：模型最不确定的 1000 对候选样本
- `pilot_tasks_20`：建议先给标注员试做的 20 对
- `tasks_round1_200`：正式第一轮标注任务

## 4. 为什么要先做 pilot

第一次真人 PAL 最容易出问题的，不是模型，而是人对任务理解不一致。

所以更稳的流程是：

1. 先发 `pilot_tasks_20_annotation.csv`
2. 回收后检查标注理由是否一致
3. 如果发现大家把“文化相似”误当成“国家相同”或“语言相同”，先修说明
4. 再发 `tasks_round1_200_annotation.csv`

## 5. 标注员到底要判断什么

问题不是：

- 这两首歌是不是来自同一国家
- 是不是同一种乐器
- 标签名字是不是一样

问题是：

- 这两首歌在“情绪功能”或“聆听意图”上是否相似

更接近下面这些判断：

- 都适合安静沉浸聆听
- 都像庆典/群体场景
- 都像舞动/律动驱动
- 都像抒情、内省、安抚

建议标注员填写：

- `similar`：`yes` / `no`
- `rationale`：一句短理由
- `annotator`：标注员代号
- `notes`：难例补充说明

## 6. 标注完成后怎么回灌

当首轮标注表填完后，把文件保存为：

- `E:\Desktop\Echo\storage\pal\v4_main_culturemert_real\tasks_round1_200_annotation_filled.csv`

然后运行：

```powershell
E:\Desktop\Echo\.venv-gpu\Scripts\python.exe -m dcas.scripts.run_pal_platform `
  --config E:\Desktop\Echo\configs\pal\pal_v4_main_culturemert_real.run.json
```

这个阶段会自动完成：

1. 把真人标注转成 pairwise constraints
2. 训练带真人 PAL 约束的新模型
3. 生成 baseline vs real PAL 的评测对比

## 7. 当前建议

如果只选一条主线做人类标注，优先级建议是：

1. `V4 main + CultureMERT`
2. 先做 `pilot 20`
3. 再做 `round1 200`
4. Gemini 暂时只作为交叉参考，不作为第一批真人 PAL 主线

原因很简单：

- 当前 `CultureMERT` 在主数据集上更稳
- 这条线更适合作为论文里的真人 PAL 主证据
- Gemini 现在更像“迁移性补充证据”
