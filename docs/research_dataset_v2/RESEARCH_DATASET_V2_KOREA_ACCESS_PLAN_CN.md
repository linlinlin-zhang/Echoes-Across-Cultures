# Research Dataset V2 Korea 接入执行方案

更新时间：2026-03-17

## 1. 目标

将 `korea` 从当前的“强候选扩展域”推进到：

- 可做 sample audit
- 可做 metadata 导入
- 具备后续升级成正式实验域的条件

本方案默认 Korea 的文化域定义为：

- `Korean traditional music / gugak audio`

---

## 2. 当前最现实的三层接入路线

### 路线 A：`digitaleum` 公开数字音源站

链接：

- <https://www.gugak.go.kr/digitaleum>

优势：

- 直接是数字音源站
- 已有公开报道说明存在 `wav / mp3` 高音质音源
- 如果 item 页面里就有稳定下载对象，这会成为最省事的 Korea 接入路线

风险：

- 可能存在动态加载
- 可能需要浏览器端解析
- 可能存在反爬或复杂会话机制

当前建议：

- **作为 Korea 第一优先样本审计入口**

---

### 路线 B：National Gugak Center digital audio OpenAPI

链接：

- <https://www.data.go.kr/data/15097515/openapi.do>

优势：

- 来自官方公共数据平台
- 页面显示自动审批
- 提供明确 API 形态

风险：

- 需要 service key
- 还未验证返回字段里是否直接含音频 URL 或可解析 playback object

当前建议：

- **作为 Korea 第一优先程序化导入入口**

---

### 路线 C：辅助 metadata / archive 线

链接：

- <https://www.data.go.kr/en/data/3062269/openapi.do>
- <https://www.data.go.kr/en/data/15083294/fileData.do>

作用：

- 补文化说明
- 补 archive linkage
- 在主音源线不足时提供 record-to-audio mapping 的辅助信息

当前建议：

- 只作为 **supporting sources**

---

## 3. 当前最推荐的推进顺序

### 第一步：先拿到一个 Korea 样本访问能力

二选一：

- `A1`：直接人工查看 `digitaleum` 的若干单条页面
- `B1`：在 `data.go.kr` 申请 `15097515` 的 service key

如果都能做，优先级是：

1. `digitaleum` 样本审计
2. `15097515` OpenAPI service key

我已经先把 Korea 这条线的 OpenAPI 审计脚本搭好了：

- [audit_data_go_openapi.py](E:/Desktop/Echo/dcas/scripts/audit_data_go_openapi.py)

它的作用不是直接导入 Korea，而是：

- 在拿到 `service key` 后
- 先批量保存几页原始返回
- 让我们快速确认字段结构、音频对象字段和分页方式

---

### 第二步：做 sample audit

目标数量：

- `20-30` 条

重点检查：

- 是否存在真实音频对象
- 是否能稳定拿到播放或下载链接
- 每条是否有足够 metadata
- 是否有较明确的版权/出处标识

达标条件：

- 至少 `20` 条中有 `15+` 条可稳定访问音频对象

如果走 OpenAPI 路线，建议先这样做：

```powershell
python E:\Desktop\Echo\dcas\scripts\audit_data_go_openapi.py `
  --service_url "PUT_REAL_SERVICE_URL_HERE" `
  --service_key "PUT_SERVICE_KEY_HERE" `
  --out_dir E:\Desktop\Echo\reports\research_dataset_v2\korea_openapi_audit `
  --pages 3 `
  --page_size 10
```

先拿到原始响应后，再去写最终的 Korea importer，会比现在猜字段稳得多。

---

### 第三步：做小规模导入 probe

目标规模：

- `50-100` 条

产物：

- `storage/public/research_dataset_v2/korea/audio/`
- `storage/public/research_dataset_v2/korea/metadata.csv`
- `storage/public/research_dataset_v2/korea/import_report.json`

---

### 第四步：决定是否升级

如果小规模导入 probe 成功，则 Korea 可升级为：

- `v2-main+1` 的第 6 域

如果 probe 失败，则 Korea 保留为：

- `strong expansion candidate`

---

## 4. 你需要做的最少动作

如果希望继续推进 Korea，当前用户侧最值钱的一步是：

### 方案 1：去 `data.go.kr` 申请 `15097515` 的 service key

因为页面显示：

- 自动审批
- OpenAPI

这意味着一旦拿到 key，我们就可以马上做 API 样本审计。

### 方案 2：手动打开 `https://www.gugak.go.kr/digitaleum`

看看：

- 是否能不登录浏览部分音源
- 单条页面是否有播放/下载入口

如果你能确认这一点，后面 Korea 的推进速度会明显加快。

---

## 5. 当前工程建议

在没有 service key 之前，不建议现在就写死 Korea 导入脚本的最终字段映射。  
更稳的方式是：

1. 先拿 sample response 或 sample page
2. 再写：
   - Korea OpenAPI importer
   - Korea metadata mapper

这样可以避免我们对参数名和字段结构做过多猜测。

---

## 6. 当前结论

Korea 现在已经不只是“理论上有可能”，而是：

- 有公开数字音源站
- 有官方 OpenAPI
- 有 archive backup

所以它完全值得继续推进。

但在没有 `digitaleum` 条目样本或 `15097515` service key 之前，  
最合理的状态仍然是：

- **强执行候选域，但尚未正式导入**
