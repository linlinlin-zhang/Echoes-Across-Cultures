# CCMUSIC “流行歌曲 Midi-Wav 双向数据库” 审核

更新日期: 2026-03-18

## 1. 问题

在已经确认 `CTIS` 主要是几秒到十多秒器乐片段的前提下, 本报告回答:

- CCMUSIC 站点里的“流行歌曲 Midi-Wav 双向数据库”是否更适合补充现代中国音乐
- 它能否进入我们当前研究数据集主线
- 如果能用, 最合适的定位是什么

## 2. 官方页面给出的信息

根据 CCMUSIC 官方 `cpop` 页面, 这个数据库:

- 包含“`hundreds of Chinese pop songs`”
- 每首歌包含对应的:
  - `midi`
  - `audio`
  - `lyric`
- 音频中把:
  - `vocal`
  - `accompaniment`
  独立录制
- 还标了与人声对应的歌唱技巧信息, 例如:
  - `breath sound`
  - `falsetto`
  - `breathing`
  - `vibrato`
  - `mute`
  - `slide`
- 页面示例歌曲 `Yueding` 给出的 demo 文件包括:
  - `Yueding xml-01.wav`
  - `Yueding vocal.wav`
  - `Yueding accompaniment.wav`
  - `Yueding.musicxml`
  - `Yueding.mid`

这说明它和 `CTIS` 很不一样:

- `CTIS` 更像短时中国传统器乐音色/乐器片段库
- 这个库更像现代流行歌曲级别的结构化研究库

## 3. 它的优势

如果目标是补“现代中国流行音乐”而不是“中国传统器乐”, 这个库的方向明显比 `CTIS` 更合适。

主要优势:

1. 它是完整歌曲级别的资源, 不是只有几秒音色片段。
2. 它明确是 `Chinese pop songs`, 语义上比之前那个 `music_genre` 数据集更贴近“中国流行音乐”。
3. 它不仅有音频, 还有:
   - MIDI
   - lyric
   - vocal / accompaniment 分离
   - singing technique 标注
4. 对很多 MIR/跨模态任务都很有吸引力, 例如:
   - 对齐
   - 转录
   - 歌声分析
   - 结构建模
   - 人声/伴奏相关任务

所以如果你要补的是:

- `cn_pop`
- `mandarin_pop`
- 现代中文流行子域

那它比 `CTIS` 更对路。

## 4. 最大问题: 不是开放直下数据

这个库最关键的问题不是内容, 而是访问和权利边界。

### 4.1 官方页面的限制

`cpop` 页面明确写到:

- `Since this database involves a copyright agreement with a music company, it is only available to the university that signed the agreement, and will be opened in succession.`

这句话的含义非常重:

- 它不是一个像 `CTIS` 那样公开可直接拉取的开放数据集
- 它涉及音乐公司版权协议
- 当前能否获得完整库, 取决于机构是否符合授权条件

### 4.2 下载页的限制

官方下载页写得更具体:

- 平台数据是 `partially free`
- 需要下载并提交 `Application Form`
- 管理方会人工评估是否提供完整数据库
- 评估通过后才会邮件发送完整库的网盘链接

也就是说, 这不是“公开下载”, 而是“申请制受控访问”。

### 4.3 申请表的限制

申请表中写明:

1. 只能用于 `academic research`
2. 不得用于任何商业用途
3. 只能由申请人及其所在部门/研究机构成员使用
4. 公开成果时必须明确引用该数据库
5. 最终解释权归数据库主管方

这进一步说明:

- 它更像“受控研究数据”
- 不是我们可以默认纳入公开可复现实验主线的开放语料

另外还有一个小风险:

- 申请表里列出的数据库名称并没有直接写成 `Midi-Wav Bi-directional Database of Pop Music`
- 更接近的条目是 `Structure Annotation Database of Songs`

这意味着在正式申请前, 最好额外确认:

- 申请表中的哪个条目实际对应这个 `cpop` 子库
- 是否需要单独说明我们申请的是 `Midi-Wav` 流行歌曲库

## 5. 还有一个需要注意的矛盾

这里存在一个需要谨慎对待的信号不一致:

### 5.1 Zenodo 页面

CCMUSIC 的 Zenodo 记录写到:

- 三个数据库都 “available for free use by computational musicology researchers”
- 提供的音频和标注 “have no commodity copyright problem”
- 公开文件里只有一个 `ccmusic-database-demo.zip` (`302.0 MB`)

### 5.2 但 `cpop` 页面又说

- 它涉及与音乐公司的版权协议
- 只有签约大学可以使用

### 5.3 我的判断

更稳妥的解释是:

- Zenodo/概览页描述的是平台层面的宣传性总说明
- 但 `cpop` 子库的实际完整访问条件, 以 `cpop` 页面和申请流程为准

因此不能因为 Zenodo 页面写了“free use”就把这个库当作完全开放数据来处理。

## 6. 它适不适合补中国音乐

### 6.1 适合补哪一类中国音乐

它适合补的是:

- 现代中文流行音乐
- 歌曲级别的人声主导数据
- 结构化流行歌曲研究

它不适合直接替代:

- `CTIS` 这类中国传统器乐域
- 中国民族器乐/地方戏曲/传统乐种域

所以如果使用它, 最合理的建模方式不是把它直接并入当前 `china` 域, 而是单独定义成:

- `cn_pop`
- `mandarin_pop`
- `modern_chinese_pop`

这样的子域。

### 6.2 它比 `music_genre` 数据集更好吗

是的, 明显更好。

和 `CCMusic music_genre` 相比:

- 它明确是 `Chinese pop songs`
- 它有真实音频
- 它有 MIDI / lyric / vocal-accompaniment 分离 / singing-technique 标注
- 它不是一个“多数英文歌”的泛 genre 分类频谱图集合

所以如果二选一:

- `music_genre` 更适合做 genre classification benchmark
- `Midi-Wav 双向数据库` 更适合做现代中文流行研究候选源

## 7. 它适不适合进入我们当前主线

我的结论是:

### 7.1 作为开放主线数据: 不建议直接纳入

原因:

- 访问受控
- 权利边界不够开放
- 完整规模没有精确公开, 只有 “hundreds”
- 目前不保证任何研究者都能复现拿到同样的完整文件

如果我们当前主线强调:

- 稳定获取
- 可复现
- 后续能比较放心地发布 metadata / embedding / splits

那么它不适合直接成为开放主线核心源。

### 7.2 作为受限扩展数据: 很值得追

如果你能通过机构渠道拿到授权, 它非常值得做成:

- `china_restricted_pop`
- `mandarin_pop_restricted`

这样的受限扩展子域。

这是因为它在内容层面确实很强:

- 歌曲级
- 中文流行
- 多模态对齐
- 有额外歌唱标注

## 8. 对当前项目最现实的建议

### 方案 A: 保持当前开放主线

- 继续把 `CTIS` 作为开放 `china` 域核心
- 不把这个 `cpop` 库放进当前公开主线

适合场景:

- 你希望主线数据尽量公开可复现
- 你不想被机构授权卡住进度

### 方案 B: 增设一个受限现代中文流行子域

- 保留 `CTIS -> china_traditional`
- 如果申请通过, 再新增:
  - `cn_pop`
  - 或 `mandarin_pop`

适合场景:

- 你愿意接受“部分主线开放 + 部分扩展受限”
- 你想让中国域不只有传统器乐, 还包括现代流行

### 方案 C: 只做研究性内部增强, 不进公开发布版本

- 内部训练/分析使用这个库
- 对外公开版本只保留开放来源

这通常是最稳的折中方案。

## 9. 审核结论

一句话结论:

这个“流行歌曲 Midi-Wav 双向数据库”是一个内容上很有价值的现代中文流行研究候选源, 但它不是开放直下数据, 更适合作为“受限访问的 `mandarin_pop` 扩展子域”, 而不适合作为当前公开主线里可自由复现的核心中国数据源。

如果只看“补中国音乐是否比 CTIS 更适合现代歌曲”:

- 是, 明显更适合现代歌曲。

如果看“是否适合马上并入当前开放主线”:

- 否, 不建议直接并入。

## 10. 下一步建议

最值得立刻确认的四件事:

1. 你所在机构是否符合该库的授权条件。
2. 授权后是否能获得完整 song-level metadata。
3. 完整库里歌曲总数到底是多少, 是否足够支撑一个稳定 `cn_pop` 子域。
4. 授权条款是否允许我们发布:
   - 元数据整理结果
   - embedding
   - split
   - 非可逆统计特征

如果这四点都过关, 我建议把它作为:

- `mandarin_pop` 受限子域

来推进, 而不是替代 `CTIS`。

## 11. Sources

- 官方数据库页: <https://ccmusic-database.github.io/en/database/cpop.html>
- 官方下载页: <https://ccmusic-database.github.io/en/download.html>
- 官方申请表: <https://ccmusic-database.github.io/en/files/Application%20Form.pdf>
- CCMUSIC Zenodo 记录: <https://zenodo.org/records/5676893>
