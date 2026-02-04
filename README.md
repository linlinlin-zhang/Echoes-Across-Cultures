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

## 路线图（接入真实音频）

- 把 embedding 替换为 CultureMERT 的帧级/段级表示
- 增加内容一致性：同曲目增广（pitch shift/EQ）后 z_c 对比学习
- 生成模块：用 z_c/z_s 做风格迁移（此原型仅保留接口，便于后续接入扩散/编解码器）

