# V3 数据集 Gemini Embedding 2 / CultureMERT 预 PAL 对比实验报告

日期：2026-03-18

## 1. 目标

在最终版 `V3` 主数据集上完成一轮真人 PAL 之前的完整预实验，比较两条 embedding 主线：

- `Gemini Embedding 2`
- `CultureMERT-95M`

并在每条 embedding 主线上运行：

- 弱基线：`popularity`
- 内容基线：`cosine`
- 邻域基线：`knn`
- 浅层学习排序：`shallow_mlp`
- 混合推荐：`hybrid_content_popularity_diversity`
- DCAS：`dcas_full_ot`
- DCAS：`dcas_full_knn`

同时补做模型结构 ablation，对比：

- `three_factor_dcas`
- `vae`
- `beta_vae`
- `factorvae`

重点观察：

- 跨文化相关度 / 目标文化贴合度
- 惊喜度（`serendipity`）
- 文化校准误差（`cultural_calibration_kl`，越低越好）
- 少数项目曝光（`minority_exposure_at_k`）

## 2. 数据与环境

### 2.1 数据集版本

- 主表：[metadata_v3_main.csv](E:/Desktop/Echo/storage/public/research_dataset_v3/metadata_v3_main.csv)
- 汇总：[summary_v3_main.json](E:/Desktop/Echo/storage/public/research_dataset_v3/summary_v3_main.json)

当前 `V3` 主数据集包含 `1122` 首音乐，`10` 个文化域：

- `china`
- `india`
- `turkey`
- `indonesia`
- `germany`
- `france`
- `italy`
- `great_britain`
- `russia`
- `modern_english_pop`

其中中国域已是最终混合版：

- `jingju_acappella`: `30`
- `traditional_instrumental`: `65`
- `mandarin_pop_singing` (`OpenCpop`): `50`

### 2.2 交互数据

由于当前没有真人听众日志，本轮 benchmark 使用弱监督合成交互：

- 文件：[interactions_v3_main.csv](E:/Desktop/Echo/storage/public/research_dataset_v3/interactions_v3_main.csv)
- 规模：`1880` 条交互
- 用户数：`240`
- 文化域数：`10`

生成命令：

```powershell
python -m dcas.scripts.synthesize_interactions `
  --metadata E:\Desktop\Echo\storage\public\research_dataset_v3\metadata_v3_main.csv `
  --out E:\Desktop\Echo\storage\public\research_dataset_v3\interactions_v3_main.csv `
  --users_per_culture 24 `
  --tracks_per_user 40 `
  --min_weight 0.5 `
  --max_weight 2.0 `
  --genre_column label `
  --seed 42
```

### 2.3 计算环境

系统侧显卡可见：

- `NVIDIA GeForce RTX 4060 Laptop GPU`
- 驱动：`591.86`
- `nvidia-smi` 显示 CUDA 运行时：`13.1`

为了避免继续占用 `C:`，额外创建了一个 GPU 实验环境：

- Python：`E:\venvs\echo-gpu\Scripts\python.exe`
- PyTorch：`2.8.0+cu128`
- `torch.cuda.is_available() == True`

说明：

- `Gemini Embedding 2` 的 embedding 构建使用 API，因此不依赖本地 GPU。
- `CultureMERT` embedding、CultureMERT 训练和 CultureMERT benchmark 使用了这个 `E:` 盘 GPU venv。

## 3. 流程记录

### 3.1 Gemini Embedding 2 轨道

Embedding 产物：

- [tracks_gemini_embedding2_main.npz](E:/Desktop/Echo/storage/public/research_dataset_v3/tracks_gemini_embedding2_main.npz)
- [tracks_gemini_embedding2_main.npz.manifest.json](E:/Desktop/Echo/storage/public/research_dataset_v3/tracks_gemini_embedding2_main.npz.manifest.json)

结果：

- `n_tracks = 1122`
- `dim = 768`
- `n_cache_hits = 1117`
- 最终 `errors = []`

训练命令：

```powershell
python -m dcas.cli.train `
  --data E:\Desktop\Echo\storage\public\research_dataset_v3\tracks_gemini_embedding2_main.npz `
  --out E:\Desktop\Echo\storage\models\dcas_full_v3_main_gemini.pt `
  --epochs 10 `
  --batch_size 128 `
  --lr 0.002 `
  --lambda_domain 0.5 `
  --lambda_contrast 0.2 `
  --lambda_cov 0.05 `
  --lambda_tc 0.05 `
  --lambda_hsic 0.02 `
  --beta_kl 1.0
```

训练损失：

- `epoch 0: 2.1766`
- `epoch 1: 1.9597`
- `epoch 2: 1.7241`
- `epoch 3: 1.6318`
- `epoch 4: 1.5953`
- `epoch 5: 1.5667`
- `epoch 6: 1.5541`
- `epoch 7: 1.5390`
- `epoch 8: 1.5366`
- `epoch 9: 1.5293`

模型产物：

- [dcas_full_v3_main_gemini.pt](E:/Desktop/Echo/storage/models/dcas_full_v3_main_gemini.pt)
- [shallow_ranker_v3_main_gemini.pt](E:/Desktop/Echo/storage/models/shallow_ranker_v3_main_gemini.pt)

Benchmark 输出目录：

- [v3_main_gemini_embedding2](E:/Desktop/Echo/reports/benchmarks/v3_main_gemini_embedding2)

Ablation 输出目录：

- [v3_main_gemini](E:/Desktop/Echo/reports/baseline_comparison/v3_main_gemini)

### 3.2 CultureMERT 轨道

先做了 1 条 GPU 烟测：

- [tracks_culturemert_v3_smoke_gpu.npz](E:/Desktop/Echo/storage/public/research_dataset_v3/tracks_culturemert_v3_smoke_gpu.npz)

全量 embedding 命令：

```powershell
E:\venvs\echo-gpu\Scripts\python.exe -m dcas.scripts.build_tracks_from_audio `
  --metadata E:\Desktop\Echo\storage\public\research_dataset_v3\metadata_v3_main.csv `
  --out E:\Desktop\Echo\storage\public\research_dataset_v3\tracks_culturemert_v3_main.npz `
  --model_id ntua-slp/CultureMERT-95M `
  --pooling mean `
  --max_seconds 30.0 `
  --skip_errors
```

Embedding 产物：

- [tracks_culturemert_v3_main.npz](E:/Desktop/Echo/storage/public/research_dataset_v3/tracks_culturemert_v3_main.npz)
- [tracks_culturemert_v3_main.npz.manifest.json](E:/Desktop/Echo/storage/public/research_dataset_v3/tracks_culturemert_v3_main.npz.manifest.json)

结果：

- `n_tracks = 1122`
- `dim = 768`
- `n_errors = 0`

训练命令：

```powershell
E:\venvs\echo-gpu\Scripts\python.exe -m dcas.cli.train `
  --data E:\Desktop\Echo\storage\public\research_dataset_v3\tracks_culturemert_v3_main.npz `
  --out E:\Desktop\Echo\storage\models\dcas_full_v3_main_culturemert.pt `
  --epochs 10 `
  --batch_size 128 `
  --lr 0.002 `
  --lambda_domain 0.5 `
  --lambda_contrast 0.2 `
  --lambda_cov 0.05 `
  --lambda_tc 0.05 `
  --lambda_hsic 0.02 `
  --beta_kl 1.0 `
  --prefer_cuda
```

训练损失：

- `epoch 0: 3.0443`
- `epoch 1: 2.3465`
- `epoch 2: 2.0116`
- `epoch 3: 1.8408`
- `epoch 4: 1.7178`
- `epoch 5: 1.6568`
- `epoch 6: 1.6032`
- `epoch 7: 1.5768`
- `epoch 8: 1.5526`
- `epoch 9: 1.5426`

模型产物：

- [dcas_full_v3_main_culturemert.pt](E:/Desktop/Echo/storage/models/dcas_full_v3_main_culturemert.pt)
- [shallow_ranker_v3_main_culturemert.pt](E:/Desktop/Echo/storage/models/shallow_ranker_v3_main_culturemert.pt)

Benchmark 输出目录：

- [v3_main_culturemert](E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert)

Ablation 输出目录：

- [v3_main_culturemert](E:/Desktop/Echo/reports/baseline_comparison/v3_main_culturemert)

## 4. Benchmark 结果

### 4.1 Gemini Embedding 2

总表源文件：

- [benchmark_summary.json](E:/Desktop/Echo/reports/benchmarks/v3_main_gemini_embedding2/benchmark_summary.json)
- [benchmark_table.md](E:/Desktop/Echo/reports/benchmarks/v3_main_gemini_embedding2/benchmark_table.md)

| 方法 | Serendipity | Calibration KL | Minority Exposure | Target Culture Prob |
|---|---:|---:|---:|---:|
| popularity | 0.805657 | 2.328180 | 0.000000 | 0.109648 |
| cosine | 0.905367 | 2.333291 | 0.391271 | 0.108737 |
| knn | 0.906172 | 2.333325 | 0.390667 | 0.108730 |
| shallow_mlp | 0.832572 | 2.335223 | 0.404104 | 0.108333 |
| hybrid_content_popularity_diversity | 0.855144 | 2.332274 | 0.175667 | 0.108926 |
| dcas_full_ot | 0.789060 | 2.327942 | 0.383604 | 0.109751 |
| dcas_full_knn | 0.788865 | 2.327995 | 0.382208 | 0.109741 |

Gemini 线的关键观察：

- `dcas_full_ot` 是这组里 `Calibration KL` 最低的方法，`target_culture_prob` 也是最高。
- 但在 `serendipity` 上，`cosine / knn / hybrid` 都明显高于 `dcas_full_ot`。
- `minority_exposure_at_k` 最好的是 `shallow_mlp`，其次是 `cosine / knn`；`dcas_full_ot` 不是最优。
- `popularity` 几乎没有长尾暴露。

也就是说，Gemini 线上的 DCAS 更像“文化校准更强”，但“惊喜度并没有赢”。

### 4.2 CultureMERT

总表源文件：

- [benchmark_summary.json](E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert/benchmark_summary.json)
- [benchmark_table.md](E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert/benchmark_table.md)

| 方法 | Serendipity | Calibration KL | Minority Exposure | Target Culture Prob |
|---|---:|---:|---:|---:|
| popularity | 0.643510 | 2.040498 | 0.000000 | 0.185936 |
| cosine | 0.744366 | 2.185802 | 0.410896 | 0.160909 |
| knn | 0.768502 | 2.187656 | 0.409104 | 0.160005 |
| shallow_mlp | 0.689171 | 2.187280 | 0.401646 | 0.159106 |
| hybrid_content_popularity_diversity | 0.705608 | 2.165751 | 0.141771 | 0.164265 |
| dcas_full_ot | 0.813677 | 2.058421 | 0.402729 | 0.185575 |
| dcas_full_knn | 0.813285 | 2.058401 | 0.402313 | 0.185634 |

CultureMERT 线的关键观察：

- `dcas_full_ot` 在 `serendipity` 上直接超过所有 raw / hybrid baseline。
- `dcas_full_ot` 的 `Calibration KL` 明显优于 `cosine / knn / shallow_mlp / hybrid`，但仍略差于 `popularity`。
- `minority_exposure_at_k` 上，`cosine / knn` 略高于 `dcas_full_ot`，但差距很小。
- `target_culture_prob` 上，`popularity` 和 `dcas_full_ot` 基本持平，都显著高于内容基线。

这说明 `CultureMERT + DCAS` 已经出现了更均衡的优势：不仅文化目标贴合度强，而且惊喜度也能一起抬起来。

## 5. DCAS 与最佳传统基线比较

### 5.1 Gemini

以 `dcas_full_ot` 对比对应最强 raw / hybrid 对手：

- 相对最佳惊喜度基线 `knn`：
  - `serendipity`: `-0.117112`
- 相对最佳长尾曝光基线 `shallow_mlp`：
  - `minority_exposure_at_k`: `-0.020500`
- 相对最佳文化校准基线 `popularity`：
  - `Calibration KL`: `-0.000239`

Gemini 线的结论很明确：

- DCAS 只在文化校准上有优势。
- 惊喜度和少数项目曝光没有赢过最强传统基线。

### 5.2 CultureMERT

- 相对最佳惊喜度基线 `knn`：
  - `serendipity`: `+0.045175`
- 相对最佳长尾曝光基线 `cosine`：
  - `minority_exposure_at_k`: `-0.008167`
- 相对最佳文化校准基线 `popularity`：
  - `Calibration KL`: `+0.017923`

CultureMERT 线的结论：

- DCAS 在惊喜度上确实赢了。
- 长尾曝光与最强内容基线接近，但没有超过。
- 纯 `popularity` 的校准 KL 更低，不过这是以牺牲惊喜度和长尾暴露为代价换来的。

如果把可用性放在一起看，`CultureMERT + DCAS` 更接近“实用系统”的平衡点。

## 6. 结构性 Ablation

### 6.1 Gemini Ablation

源文件：

- [baseline_comparison_summary.json](E:/Desktop/Echo/reports/baseline_comparison/v3_main_gemini/baseline_comparison_summary.json)

三因子模型均值：

| 变体 | Serendipity | Calibration KL | Minority Exposure |
|---|---:|---:|---:|
| three_factor_dcas | 0.822584 | 2.376040 | 0.390681 |
| vae | 0.837985 | 2.376171 | 0.374472 |
| beta_vae | 0.838226 | 2.376171 | 0.374431 |
| factorvae | 0.832523 | 2.376196 | 0.382417 |

解读：

- `three_factor_dcas` 在 `Calibration KL` 上稳定优于 `vae / beta_vae / factorvae`。
- 在 `minority_exposure_at_k` 上通常优于 `vae / beta_vae`，对 `factorvae` 的优势较弱。
- 在 `serendipity` 上反而低于 `vae / beta_vae`，对 `factorvae` 也没有形成稳定优势。

因此：

- `three_factor_necessity_checks = false`
- 不能说三因子结构在 Gemini 线上“全面必要”
- 但可以说它明确提升了文化校准，并在部分长尾曝光上更好

### 6.2 CultureMERT Ablation

源文件：

- [baseline_comparison_summary.json](E:/Desktop/Echo/reports/baseline_comparison/v3_main_culturemert/baseline_comparison_summary.json)

三因子模型均值：

| 变体 | Serendipity | Calibration KL | Minority Exposure |
|---|---:|---:|---:|
| three_factor_dcas | 0.831674 | 2.376078 | 0.424250 |
| vae | 0.854679 | 2.376154 | 0.406111 |
| beta_vae | 0.854026 | 2.376156 | 0.406375 |
| factorvae | 0.846564 | 2.376187 | 0.423208 |

解读：

- `three_factor_dcas` 同样在 `Calibration KL` 上稳定优于所有 VAE 系对照。
- 对 `vae / beta_vae`，`minority_exposure_at_k` 也有稳定提升。
- 对 `factorvae`，长尾曝光优势很小。
- 在 `serendipity` 上仍然低于 `vae / beta_vae`。

因此：

- `three_factor_necessity_checks = false`
- 三因子结构的最稳收益仍然是文化校准，而不是惊喜度

这个结果很重要：  
`benchmark 里 DCAS 对传统推荐方法的优势，和 ablation 里三因子结构相对普通 VAE 的优势，并不是同一件事。`

更准确地说：

- `DCAS 作为推荐系统` 在 `CultureMERT` 上很强
- 但 `三因子潜变量结构` 本身的主收益更偏校准，不是单纯拉高惊喜度

## 7. Gemini 与 CultureMERT 的 DCAS 直接比较

以 `dcas_full_ot` 为准：

| 组合 | Serendipity | Calibration KL | Minority Exposure | Target Culture Prob |
|---|---:|---:|---:|---:|
| Gemini + DCAS | 0.789060 | 2.327942 | 0.383604 | 0.109751 |
| CultureMERT + DCAS | 0.813677 | 2.058421 | 0.402729 | 0.185575 |
| CultureMERT - Gemini | +0.024617 | -0.269521 | +0.019125 | +0.075824 |

结论：

- `CultureMERT + DCAS` 在四个核心维度里有三个明显更好：
  - `serendipity` 更高
  - `Calibration KL` 更低
  - `minority exposure` 更高
  - `target culture probability` 明显更高

这意味着在当前 `V3 + 合成交互` 设定下，`CultureMERT` 是更强的预 PAL 主线。

## 8. 每文化域观察

以 `dcas_full_ot` 为例：

### 8.1 Gemini 线

惊喜度最高的文化域：

- `india`: `ser=0.833445`, `kl=2.255854`, `target_prob=0.124081`
- `china`: `ser=0.815641`, `kl=2.339854`, `target_prob=0.107285`
- `indonesia`: `ser=0.814306`, `kl=2.323694`, `target_prob=0.110560`

校准最好（KL 最低）的文化域：

- `india`: `2.255854`
- `turkey`: `2.312003`
- `indonesia`: `2.323694`

### 8.2 CultureMERT 线

惊喜度最高的文化域：

- `modern_english_pop`: `ser=0.839720`, `kl=1.698509`, `target_prob=0.274988`
- `turkey`: `ser=0.836323`, `kl=1.878045`, `target_prob=0.234997`
- `france`: `ser=0.820060`, `kl=2.248591`, `target_prob=0.143762`

校准最好（KL 最低）的文化域：

- `modern_english_pop`: `1.698509`
- `india`: `1.853419`
- `turkey`: `1.878045`

含义：

- `CultureMERT` 对现代流行域和土耳其现代域的文化识别非常强。
- 它对中国、印度、印尼等非西方文化域的目标文化概率也比 Gemini 高得多。
- 从跨文化推荐的角度，这对后续 PAL 很重要，因为模型的不确定点会更集中在真正难的边界上，而不是 embedding 自身表达不足。

## 9. 对后续真人 PAL 的建议

建议把这轮实验后的主线定为：

1. 主 baseline：`CultureMERT + dcas_full_ot`
2. 对照 baseline：`Gemini Embedding 2 + dcas_full_ot`
3. 辅助传统对照：
   - `knn`
   - `hybrid_content_popularity_diversity`
   - `popularity`

原因：

- `CultureMERT + DCAS` 是目前最平衡、最像“可进入真人 PAL 阶段”的组合。
- `Gemini + DCAS` 仍保留研究价值，因为它代表外部闭源 embedding 路线。
- `knn / hybrid / popularity` 足够覆盖弱到强的传统推荐基线。

真人 PAL 时，建议优先基于 `CultureMERT + DCAS` 生成候选对，再用 `Gemini + DCAS` 做交叉验证。

## 10. 限制

- 当前交互数据是合成的，不是真实用户日志。
- 当前 hybrid baseline 仍是启发式混合，不是工业平台那种大规模协同过滤 + 多塔排序系统。
- benchmark 使用 `k=20`，ablation 使用 `k=10`，两者不要直接混读。
- 本轮还没有引入真人 PAL 约束，所以结果应视为 `pre-PAL baseline`。

## 11. 结论

这轮 `V3` 预 PAL 实验可以下一个清晰判断：

- 如果只看 `Gemini`，DCAS 更像“文化校准增强器”，还不是“全面优于传统推荐”的方案。
- 如果换成 `CultureMERT`，DCAS 才真正表现出更完整的优势，尤其是在惊喜度和跨文化贴合度的平衡上。
- 因此，后续真人 PAL 最值得投入的主路线应是：`CultureMERT + DCAS`。

原始结果请直接查看：

- [Gemini benchmark](E:/Desktop/Echo/reports/benchmarks/v3_main_gemini_embedding2/benchmark_summary.json)
- [Gemini ablation](E:/Desktop/Echo/reports/baseline_comparison/v3_main_gemini/baseline_comparison_summary.json)
- [CultureMERT benchmark](E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert/benchmark_summary.json)
- [CultureMERT ablation](E:/Desktop/Echo/reports/baseline_comparison/v3_main_culturemert/baseline_comparison_summary.json)
