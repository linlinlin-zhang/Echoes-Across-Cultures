# 跨文化推荐盲听 Pilot 研究者说明

这个文件夹是今晚可用的真人盲听小实验包。
目标是为 ISMIR 论文补一个 small blind listener sanity check，缓解 synthetic interactions 的弱点。

## 文件夹结构

- `volunteer_site/`: 发给志愿者的网页文件夹。
- `volunteer_site/index.html`: 志愿者双击打开这个文件即可开始。
- `volunteer_site/audio/`: 已复制并截取好的 36 个音频片段。
- `volunteer_site/tasks.js`: 网页任务数据，不包含方法名。
- `volunteer_site/task_manifest_public.csv`: 公开任务清单，不包含方法名。
- `researcher_private/method_key_private.csv`: 研究者私有方法映射，不要发给志愿者。

## 推荐怎么发给志愿者

现在推荐使用反平衡后的 10 个个人包：

```text
human_pilot_package/participant_versions/P01_volunteer_site.zip
human_pilot_package/participant_versions/P02_volunteer_site.zip
...
human_pilot_package/participant_versions/P10_volunteer_site.zip
```

给 10 个志愿者分别发 P01-P10 对应的 zip。
这些包使用同一组 12 个核心任务，但题目顺序不同，A/B 左右位置也做了反平衡。
这样比十套完全不同的音乐更适合统计，因为同一任务会被多人评价，同时又能减少 A/B 位置偏差。

如果临时不想区分志愿者，也可以使用共用包：

```text
human_pilot_package/volunteer_site_for_participants.zip
```

但更推荐 P01-P10 个人包。

不要把整个 `human_pilot_package/` 都发出去，因为里面有 `researcher_private/`。
研究者私有 key 包含 A/B 分别对应 baseline 还是 PAL，给志愿者看到会破坏盲测。

## 志愿者怎么做

1. 解压分配给自己的 zip，例如 `P01_volunteer_site.zip`。
2. 双击打开 `index.html`。
3. 页面会自动填入匿名编号；如果没有自动填，可以手动输入对应编号，例如 `P01`。
4. 每题先听参考音乐，再听候选 A/B。
5. 完成 12 题后点击 `导出 CSV`。
6. 把导出的 CSV 发回给你。

## 实验内容

当前任务共 12 组。

每组包含：

- Seed/context track: 用户历史里的参考音乐。
- Candidate A: baseline 或 PAL-balanced 推荐，顺序随机。
- Candidate B: 另一个方法的推荐，顺序随机。

使用的方法：

- Baseline: `BPR listwise hybrid`，作为今晚可直接运行的强基线替代。
- Experimental: `PAL OT calibrated P3 balanced`。

说明：

原计划用论文主表中的 `BPR+LambdaMART hybrid`，但当前本机环境缺少 `lightgbm`，无法加载 LambdaMART pickle。
为了今晚能稳定执行，任务包使用同一 benchmark 系列中的 PyTorch `BPR listwise hybrid` 作为强基线。
论文正式写作时要诚实表述为 small sanity check，不要说它严格复现主 benchmark 的 strongest hybrid baseline。

## 收回 CSV 后如何统计

志愿者导出的 CSV 不含方法名。
如果使用 P01-P10 个人包，统计时默认使用：

```text
researcher_private/method_key_private_counterbalanced_all.csv
```

如果使用共用包，统计时才使用：

```text
researcher_private/method_key_private.csv
```

把每个 task 的 `candidate_a_method` / `candidate_b_method` 合并进去，然后计算 PAL 胜率。

三个主要指标：

- `compatible_choice`: 情绪/风格兼容性。
- `discovery_choice`: 跨文化新鲜感且仍可听。
- `overall_choice`: 总体更想继续听。

推荐统计方式：

```text
PAL win rate = PAL wins / (PAL wins + baseline wins)
Tie 单独报告，或按 0.5 计入敏感性分析。
```

运行：

```powershell
cd E:\Desktop\Echo\human_pilot_package\researcher_private
python analyze_responses.py
```

论文可写成：

```text
We additionally conducted a small blind listener sanity check with 10 volunteers and
12 anonymized A/B tasks each. The pilot compared a BPR-listwise hybrid baseline with
the PAL-balanced calibrated operating point. We report this only as pilot evidence,
not as a population-level user study.
```

## 风险控制

- 不收集真实姓名。
- 不收集手机号、IP、账号等敏感信息。
- 不要公开发布音频包。
- 只将本实验称为 `small blind listener sanity check`。
- 如果结果一般，也可以写进论文：它说明跨文化发现性提升更明显，但兼容性仍需要更多 PAL 数据。
