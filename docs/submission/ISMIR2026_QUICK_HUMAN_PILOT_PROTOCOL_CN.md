# ISMIR 2026 快速真人听众 Pilot 方案

目的：用一个小而干净的 blind listener sanity check，补强论文中 synthetic interactions 和小规模 PAL 的可信度。
本方案不是正式大规模用户研究，不应被写成 population-level user study。

## 结论建议

建议做。

它对录用的帮助不是“直接证明系统真实推荐效果非常强”，而是：

- 给评审一个真实听众层面的 sanity check，缓解主 benchmark 依赖 synthetic interactions 的弱点。
- 支撑论文的核心叙事：DCAS/PAL 推荐不是只在自定义指标上好看，而是在人耳判断中也更容易被认为“情绪/风格可连接、文化上更新颖但仍可听”。
- 给 Discussion 或 Appendix 增加一个人类评价小节，让论文从纯离线系统变成 human-grounded prototype。

如果时间有限，只做 10 人也值得。但必须严格写成 pilot 或 sanity check。

## 不要这么做

- 不要说“用户研究证明系统显著优于 baseline”。
- 不要说“10 人结果代表真实用户偏好”。
- 不要让志愿者知道哪个结果来自我们的方法，哪个来自 baseline。
- 不要让志愿者只听完整歌曲，时间会爆炸。
- 不要用 AI agents 替代这次真人 pilot。

## 推荐实验设计

人数：8-12 人，目标 10 人。

任务数：每人 12 组，最多 15 组。不要超过 20 组，否则疲劳会影响质量。

每组材料：

- 一个 seed/context track，可以给 20-30 秒片段或标题/艺术家信息。
- Candidate A：baseline 推荐。
- Candidate B：DCAS 或 PAL 推荐。
- A/B 顺序随机，隐藏方法名。

每组问题：

1. 情绪/风格兼容性：哪一个候选更像是“虽然不一定同文化，但听感、情绪或功能上更接得上 seed/context”？
2. 跨文化发现性：哪一个候选更像是“文化上更新鲜，但仍然可听、不是乱推”？
3. 总体偏好：如果要推荐给你继续听，你更愿意点哪一个？
4. 可选一句话理由：为什么？

问题 1 和 2 是论文最有用的。
问题 3 可以作为直觉补充。
问题 4 用于 qualitative examples。

## 表单推荐字段

建议用腾讯问卷、问卷星、Google Forms 或飞书表格。每一行对应一个志愿者对一个任务的回答。

字段如下：

| 字段名 | 含义 |
|---|---|
| participant_id | 匿名编号，例如 P01 |
| task_id | 任务编号，例如 T01 |
| seed_track | seed/context track ID 或匿名名 |
| candidate_a_id | A 候选 ID |
| candidate_b_id | B 候选 ID |
| a_method_hidden | A 对应方法，仅研究者保存，不给志愿者看 |
| b_method_hidden | B 对应方法，仅研究者保存，不给志愿者看 |
| compatible_choice | A / B / Tie |
| discovery_choice | A / B / Tie |
| overall_choice | A / B / Tie |
| confidence | 1-5 |
| comment | 一句话理由，可空 |

给志愿者看的表单不要出现 `a_method_hidden` 和 `b_method_hidden`。
研究者自己保留一个 key 文件即可。

## 快速任务选择

优先选择能体现论文主张的任务，而不是随机全抽。

推荐选 12-15 个 seed/context：

- 4-5 个 CultureMERT 主结果中 DCAS/PAL 明显改善 minority exposure 的任务。
- 4-5 个 baseline 和 DCAS 都看起来合理、但 DCAS 更跨文化的边界任务。
- 2-3 个系统可能失败或比较难判断的任务。

不要只选“我们稳赢”的任务。
最好保留少量失败/边界任务，这样论文写起来更诚实。

## 快速统计方式

每个问题分别统计 DCAS/PAL 胜率：

```text
win_rate = DCAS_or_PAL_wins / (DCAS_or_PAL_wins + baseline_wins)
```

Tie 可以单独报告，也可以按 0.5 计入。

建议同时报告：

- compatibility win rate
- discovery win rate
- overall preference win rate
- number of participants
- number of judgments
- tie rate
- optional binomial sign test p-value

示例表述：

```text
In a small blind listener sanity check with 10 participants and 120 pairwise judgments,
participants preferred the DCAS/PAL candidate over the baseline in X% of compatibility
judgments and Y% of cross-cultural discovery judgments. We report this result only as
pilot evidence because the study is small and not population-representative.
```

## 论文里应该怎么写

推荐正文或 appendix 文字：

```text
To complement the synthetic interaction protocol, we conducted a small blind listener
sanity check with 10 volunteers. Each participant compared anonymized recommendation
pairs from the strongest hybrid baseline and the calibrated DCAS/PAL operating point.
The questionnaire asked which candidate was more affectively/style-compatible with the
seed context and which was more culturally novel while remaining listenable. We treat
this study as pilot evidence only, not as a population-level user evaluation.
```

如果结果好，可以写：

```text
The pilot supports the offline trend: listeners more often selected the calibrated
DCAS/PAL candidate for cross-cultural discovery while maintaining comparable or better
style-affective compatibility.
```

如果结果一般，也可以写：

```text
The pilot suggests that the calibrated recommendations are perceived as more culturally
novel, while compatibility judgments are mixed. This supports the need for larger PAL
rounds and more user-grounded calibration.
```

## 明天摘要是否要写进去

如果今晚能收完并且结果方向明显，可以在摘要里谨慎加一句：

```text
A small blind listener sanity check further supports that the calibrated operating
point improves perceived cross-cultural discovery without collapsing style-affective
compatibility.
```

如果今晚还没收完，不要写进摘要。可以只保留目前摘要里的 200-pair PAL pilot。

## 最快执行清单

1. 今晚先确定 12 个任务。
2. 每个任务准备 seed、baseline candidate、DCAS/PAL candidate。
3. 随机 A/B 顺序，做一个隐藏 method key。
4. 发给 10 个志愿者。
5. 每人控制在 10-15 分钟内完成。
6. 收回 CSV 后统计三个胜率和 tie rate。
7. 只在论文中写成 small blind listener sanity check。

## 风险控制

- 如果音乐片段涉及版权，不要公开发布音频包；只用于私下学术问卷 pilot。
- 不要收集真实姓名，使用 P01-P10。
- 不收集敏感个人信息。
- 如果问卷平台要求登录，导出前去掉账号、IP、手机号等信息。
- 如果志愿者不是音乐专业也没关系；这次评价的是 listener-facing recommendation plausibility，不是专家音乐学判断。
