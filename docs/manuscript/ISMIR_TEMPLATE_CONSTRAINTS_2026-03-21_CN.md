# ISMIR Template Constraints (2026-03-21)

## 1. 模板来源

- style file: [ismir.sty](E:/Desktop/Echo/paper/ismir.sty)
- current draft: [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex)

## 2. 章节与结构约束

- 模板本身没有强制 `IMRaD` 顺序。
- 模板只定义了 `section / subsection / subsubsection` 的排版方式，没有硬编码“必须 Introduction/Method/Results/Discussion”的顺序。
- `abstract` 环境存在，但 `ismir.sty` 中没有写死 abstract 字数上限。
- 本地模板中也没有编码页数上限，因此页数限制需要以投稿通知或 CFP 为准，而不能从 `ismir.sty` 直接推出。
- `submission` 选项会自动匿名并打开行号。

代码锚点：

- [ismir.sty:95-103](E:/Desktop/Echo/paper/ismir.sty#L95)
- [ismir.sty:329-330](E:/Desktop/Echo/paper/ismir.sty#L329)
- [ismir.sty:334-401](E:/Desktop/Echo/paper/ismir.sty#L334)

## 3. 图表规范

- 模板为双栏版式：`\twocolumn`
- `textwidth = 172mm`
- `columnsep = 8mm`
- 因此单栏可用宽度约为 `(172 - 8) / 2 = 82mm`
- `figure` 与 `table` 标题格式由 `\@makecaption` 统一控制
- 图题前缀固定为 `Figure n`，表题前缀固定为 `Table n`
- 正文字号默认约 `10pt`，`\small` 约 `9pt`，`\footnotesize` 约 `8pt`

这意味着：

- 主结果表若列很多，优先考虑 `table*`
- 复杂流程图优先用 `figure*`
- 单栏图尽量控制在约 `82mm` 宽
- 图中文字不要低于 `8pt` 左右，否则和模板正文不协调

代码锚点：

- [ismir.sty:133-159](E:/Desktop/Echo/paper/ismir.sty#L133)
- [ismir.sty:417-429](E:/Desktop/Echo/paper/ismir.sty#L417)

## 4. 引用格式

- bibliography style 固定为 `IEEEtran`
- 文内通过 `cite.sty` 走数值型引用
- 因而推荐采用 numerical citations，例如 `[3], [5], [7]`
- 本地模板没有显式禁止 arXiv，但也没有给 arXiv 特殊格式说明

因此更稳妥的投稿口径是：

- 优先引用正式发表版本
- 若只有 arXiv，可保留，但在 Related Work 中尽量控制其承担核心论证的比例

代码锚点：

- [ismir.sty:401-415](E:/Desktop/Echo/paper/ismir.sty#L401)
- [cite.sty](E:/Desktop/Echo/paper/cite.sty)

## 5. 当前写作上的直接约束

- 现稿 [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex) 已使用匿名提交模式，符合 submission 形态。
- 模板没有替你限制摘要字数，所以摘要必须手工压缩。
- 模板没有替你限制图表数量，所以真正的约束将来自页数预算，而不是 style 文件本身。

## 6. 对本项目的具体建议

- 章节顺序采用稳健版：
  - Introduction
  - Related Work
  - Method
  - Experimental Setup
  - Results
  - Discussion and Limitations
  - Conclusion
- 主文图表优先保留：
  - pipeline overview
  - main benchmark table
  - calibration sweep figure
  - one PAL workflow figure
- 大量扩展表格放附录或 supplementary，不要在主文堆积过多多栏表。

