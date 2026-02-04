import { useEffect, useMemo, useState } from 'react'
import './App.css'

function App() {
  const [tab, setTab] = useState<'data' | 'train' | 'recommend' | 'pal'>('data')
  const [files, setFiles] = useState<string[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [toyName, setToyName] = useState('toy')
  const [toyTracks, setToyTracks] = useState(1200)
  const [toyDim, setToyDim] = useState(96)
  const [toySeed, setToySeed] = useState(7)

  const [tracksPath, setTracksPath] = useState('')
  const [interactionsPath, setInteractionsPath] = useState('')
  const [modelPath, setModelPath] = useState('')

  const [trainOutName, setTrainOutName] = useState('model.pt')
  const [trainEpochs, setTrainEpochs] = useState(8)
  const [trainBatchSize, setTrainBatchSize] = useState(256)
  const [trainLr, setTrainLr] = useState(0.002)
  const [trainSeed, setTrainSeed] = useState(42)

  const [recUserId, setRecUserId] = useState('u0')
  const [recTargetCulture, setRecTargetCulture] = useState('india')
  const [recK, setRecK] = useState(10)
  const [recResult, setRecResult] = useState<null | {
    metrics: Record<string, number>
    recommendations: Array<{
      track_id: string
      culture: string
      score: number
      relevance: number
      unexpectedness: number
    }>
  }>(null)

  const [palOutName, setPalOutName] = useState('pal_tasks.jsonl')
  const [palN, setPalN] = useState(50)
  const [palResult, setPalResult] = useState<null | { tasks: string; count: number }>(null)

  const [uploadDir, setUploadDir] = useState('uploads')
  const [uploadFile, setUploadFile] = useState<File | null>(null)

  const availableCultures = useMemo(() => {
    const cultures = new Set<string>()
    for (const f of files) {
      const m = f.match(/^datasets\/([^/]+)\/meta\.txt$/)
      if (m) cultures.add(m[1])
    }
    return Array.from(cultures.values()).sort()
  }, [files])

  async function apiGet<T>(path: string): Promise<T> {
    const res = await fetch(path)
    if (!res.ok) throw new Error(await res.text())
    return (await res.json()) as T
  }

  async function apiPost<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) throw new Error(await res.text())
    return (await res.json()) as T
  }

  async function refreshFiles() {
    const out = await apiGet<{ files: string[] }>('/api/files')
    setFiles(out.files)
  }

  useEffect(() => {
    refreshFiles().catch(() => {})
  }, [])

  async function run<T>(label: string, fn: () => Promise<T>) {
    setError(null)
    setBusy(label)
    try {
      const v = await fn()
      await refreshFiles()
      return v
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      throw e
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="app">
      <div className="topBar">
        <div className="brand">
          <div className="brandTitle">DCAS 控制台</div>
          <div className="brandSub">跨文化音乐推荐 · 解纠缠 · OT 对齐 · PAL</div>
        </div>
        <div className="topRight">
          <button className="ghost" onClick={() => refreshFiles().catch(() => {})} disabled={busy !== null}>
            刷新文件
          </button>
          <a className="ghost" href="/docs" target="_blank" rel="noreferrer">
            API 文档
          </a>
        </div>
      </div>

      <div className="tabs">
        <button className={tab === 'data' ? 'tab active' : 'tab'} onClick={() => setTab('data')}>
          数据
        </button>
        <button className={tab === 'train' ? 'tab active' : 'tab'} onClick={() => setTab('train')}>
          训练
        </button>
        <button className={tab === 'recommend' ? 'tab active' : 'tab'} onClick={() => setTab('recommend')}>
          推荐
        </button>
        <button className={tab === 'pal' ? 'tab active' : 'tab'} onClick={() => setTab('pal')}>
          PAL
        </button>
      </div>

      {error ? (
        <div className="alert">
          <div className="alertTitle">请求失败</div>
          <div className="alertBody">{error}</div>
        </div>
      ) : null}

      <div className="grid">
        <div className="card">
          <div className="cardTitle">工作区</div>
          <div className="kv">
            <div className="k">tracks.npz</div>
            <div className="v mono">{tracksPath || '未选择'}</div>
          </div>
          <div className="kv">
            <div className="k">interactions.csv</div>
            <div className="v mono">{interactionsPath || '未选择'}</div>
          </div>
          <div className="kv">
            <div className="k">model</div>
            <div className="v mono">{modelPath || '未选择'}</div>
          </div>
          <div className="hint">任意一步输出的相对路径都可以直接复制到下一步输入框。</div>
        </div>

        {tab === 'data' ? (
          <div className="card span2">
            <div className="cardTitle">数据管理</div>

            <div className="sectionTitle">生成玩具数据</div>
            <div className="row">
              <label className="field">
                <div className="label">名称</div>
                <input value={toyName} onChange={(e) => setToyName(e.target.value)} />
              </label>
              <label className="field">
                <div className="label">曲目数</div>
                <input type="number" value={toyTracks} onChange={(e) => setToyTracks(Number(e.target.value))} />
              </label>
              <label className="field">
                <div className="label">维度</div>
                <input type="number" value={toyDim} onChange={(e) => setToyDim(Number(e.target.value))} />
              </label>
              <label className="field">
                <div className="label">种子</div>
                <input type="number" value={toySeed} onChange={(e) => setToySeed(Number(e.target.value))} />
              </label>
              <div className="field">
                <div className="label">操作</div>
                <button
                  className="primary"
                  disabled={busy !== null}
                  onClick={() =>
                    run('生成玩具数据', async () => {
                      const out = await apiPost<{ tracks: string; interactions: string }>(
                        '/api/toy/generate',
                        {
                          name: toyName.trim() || 'toy',
                          n_tracks: toyTracks,
                          dim: toyDim,
                          seed: toySeed,
                        },
                      )
                      setTracksPath(out.tracks)
                      setInteractionsPath(out.interactions)
                    })
                  }
                >
                  生成
                </button>
              </div>
            </div>

            <div className="sectionTitle">上传文件</div>
            <div className="row">
              <label className="field grow">
                <div className="label">目标目录</div>
                <input value={uploadDir} onChange={(e) => setUploadDir(e.target.value)} placeholder="uploads" />
              </label>
              <label className="field grow">
                <div className="label">文件</div>
                <input type="file" onChange={(e) => setUploadFile(e.target.files?.item(0) ?? null)} />
              </label>
              <div className="field">
                <div className="label">操作</div>
                <button
                  className="primary"
                  disabled={busy !== null || uploadFile === null}
                  onClick={() =>
                    run('上传文件', async () => {
                      if (!uploadFile) return
                      const fd = new FormData()
                      fd.append('file', uploadFile)
                      const res = await fetch(`/api/files/upload?dir=${encodeURIComponent(uploadDir || 'uploads')}`, {
                        method: 'POST',
                        body: fd,
                      })
                      if (!res.ok) throw new Error(await res.text())
                      await res.json()
                      setUploadFile(null)
                    })
                  }
                >
                  上传
                </button>
              </div>
            </div>

            <div className="sectionTitle">文件列表</div>
            <div className="fileList">
              {files.length === 0 ? <div className="muted">暂无文件</div> : null}
              {files.map((f) => (
                <button
                  key={f}
                  className="fileItem"
                  onClick={() => {
                    if (f.endsWith('tracks.npz')) setTracksPath(f)
                    if (f.endsWith('interactions.csv')) setInteractionsPath(f)
                    if (f.endsWith('.pt')) setModelPath(f)
                  }}
                >
                  <div className="mono">{f}</div>
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {tab === 'train' ? (
          <div className="card span2">
            <div className="cardTitle">训练（解纠缠 + 领域对抗）</div>

            <div className="row">
              <label className="field grow">
                <div className="label">tracks.npz 路径</div>
                <input value={tracksPath} onChange={(e) => setTracksPath(e.target.value)} placeholder="datasets/toy/tracks.npz" />
              </label>
              <label className="field">
                <div className="label">输出文件名</div>
                <input value={trainOutName} onChange={(e) => setTrainOutName(e.target.value)} />
              </label>
            </div>

            <div className="row">
              <label className="field">
                <div className="label">epochs</div>
                <input type="number" value={trainEpochs} onChange={(e) => setTrainEpochs(Number(e.target.value))} />
              </label>
              <label className="field">
                <div className="label">batch</div>
                <input type="number" value={trainBatchSize} onChange={(e) => setTrainBatchSize(Number(e.target.value))} />
              </label>
              <label className="field">
                <div className="label">lr</div>
                <input type="number" step="0.0001" value={trainLr} onChange={(e) => setTrainLr(Number(e.target.value))} />
              </label>
              <label className="field">
                <div className="label">seed</div>
                <input type="number" value={trainSeed} onChange={(e) => setTrainSeed(Number(e.target.value))} />
              </label>
              <div className="field">
                <div className="label">操作</div>
                <button
                  className="primary"
                  disabled={busy !== null || !tracksPath}
                  onClick={() =>
                    run('训练', async () => {
                      const out = await apiPost<{ checkpoint: string }>('/api/train', {
                        tracks_path: tracksPath,
                        out_name: trainOutName,
                        epochs: trainEpochs,
                        batch_size: trainBatchSize,
                        lr: trainLr,
                        seed: trainSeed,
                      })
                      setModelPath(out.checkpoint)
                      setTab('recommend')
                    })
                  }
                >
                  开始训练
                </button>
              </div>
            </div>

            <div className="hint">输出 checkpoint 会自动写入工作区的 model 字段，随后可直接进入推荐。</div>
          </div>
        ) : null}

        {tab === 'recommend' ? (
          <div className="card span2">
            <div className="cardTitle">跨文化推荐（OT 对齐）</div>

            <div className="row">
              <label className="field grow">
                <div className="label">model 路径</div>
                <input value={modelPath} onChange={(e) => setModelPath(e.target.value)} placeholder="models/model.pt" />
              </label>
              <label className="field grow">
                <div className="label">tracks 路径</div>
                <input value={tracksPath} onChange={(e) => setTracksPath(e.target.value)} placeholder="datasets/toy/tracks.npz" />
              </label>
              <label className="field grow">
                <div className="label">interactions 路径</div>
                <input
                  value={interactionsPath}
                  onChange={(e) => setInteractionsPath(e.target.value)}
                  placeholder="datasets/toy/interactions.csv"
                />
              </label>
            </div>

            <div className="row">
              <label className="field">
                <div className="label">user_id</div>
                <input value={recUserId} onChange={(e) => setRecUserId(e.target.value)} />
              </label>
              <label className="field">
                <div className="label">target_culture</div>
                <input value={recTargetCulture} onChange={(e) => setRecTargetCulture(e.target.value)} list="cultures" />
                <datalist id="cultures">
                  {availableCultures.map((c) => (
                    <option key={c} value={c} />
                  ))}
                </datalist>
              </label>
              <label className="field">
                <div className="label">k</div>
                <input type="number" value={recK} onChange={(e) => setRecK(Number(e.target.value))} />
              </label>
              <div className="field">
                <div className="label">操作</div>
                <button
                  className="primary"
                  disabled={busy !== null || !modelPath || !tracksPath || !interactionsPath}
                  onClick={() =>
                    run('推荐', async () => {
                      const out = await apiPost<typeof recResult>('/api/recommend', {
                        model_path: modelPath,
                        tracks_path: tracksPath,
                        interactions_path: interactionsPath,
                        user_id: recUserId,
                        target_culture: recTargetCulture,
                        k: recK,
                      })
                      setRecResult(out)
                      setPalResult(null)
                    })
                  }
                >
                  生成推荐
                </button>
              </div>
            </div>

            {recResult ? (
              <div className="resultGrid">
                <div className="miniCard">
                  <div className="miniTitle">指标</div>
                  <div className="mono small">
                    {Object.entries(recResult.metrics).map(([k, v]) => (
                      <div key={k} className="kvLine">
                        <div className="k">{k}</div>
                        <div className="v">{v.toFixed(6)}</div>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="miniCard span2">
                  <div className="miniTitle">推荐列表</div>
                  <div className="table">
                    <div className="thead">
                      <div>track_id</div>
                      <div>culture</div>
                      <div>score</div>
                      <div>relevance</div>
                      <div>unexpected</div>
                    </div>
                    {recResult.recommendations.map((r) => (
                      <div className="trow" key={r.track_id}>
                        <div className="mono">{r.track_id}</div>
                        <div>{r.culture}</div>
                        <div className="mono">{r.score.toExponential(3)}</div>
                        <div className="mono">{r.relevance.toFixed(6)}</div>
                        <div className="mono">{r.unexpectedness.toFixed(6)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="muted">尚未生成推荐</div>
            )}
          </div>
        ) : null}

        {tab === 'pal' ? (
          <div className="card span2">
            <div className="cardTitle">参与式主动学习（PAL）</div>
            <div className="row">
              <label className="field grow">
                <div className="label">model 路径</div>
                <input value={modelPath} onChange={(e) => setModelPath(e.target.value)} placeholder="models/model.pt" />
              </label>
              <label className="field grow">
                <div className="label">tracks 路径</div>
                <input value={tracksPath} onChange={(e) => setTracksPath(e.target.value)} placeholder="datasets/toy/tracks.npz" />
              </label>
            </div>
            <div className="row">
              <label className="field">
                <div className="label">输出文件名</div>
                <input value={palOutName} onChange={(e) => setPalOutName(e.target.value)} />
              </label>
              <label className="field">
                <div className="label">n</div>
                <input type="number" value={palN} onChange={(e) => setPalN(Number(e.target.value))} />
              </label>
              <div className="field">
                <div className="label">操作</div>
                <button
                  className="primary"
                  disabled={busy !== null || !modelPath || !tracksPath}
                  onClick={() =>
                    run('PAL 选样', async () => {
                      const out = await apiPost<{ tasks: string; count: number }>('/api/pal', {
                        model_path: modelPath,
                        tracks_path: tracksPath,
                        out_name: palOutName,
                        n: palN,
                      })
                      setPalResult(out)
                    })
                  }
                >
                  导出任务
                </button>
              </div>
            </div>

            {palResult ? (
              <div className="miniCard">
                <div className="miniTitle">输出</div>
                <div className="kvLine">
                  <div className="k">tasks</div>
                  <div className="v mono">{palResult.tasks}</div>
                </div>
                <div className="kvLine">
                  <div className="k">count</div>
                  <div className="v mono">{palResult.count}</div>
                </div>
                <div className="row gap">
                  <a className="primaryLink" href={`/api/files/download?path=${encodeURIComponent(palResult.tasks)}`} target="_blank" rel="noreferrer">
                    下载 JSONL
                  </a>
                </div>
              </div>
            ) : (
              <div className="muted">尚未导出任务</div>
            )}
          </div>
        ) : null}
      </div>

      <div className="footer">
        <div className="muted">后端：FastAPI · 前端：React · 运行在本地开发环境</div>
        {busy ? <div className="busy">正在执行：{busy}</div> : <div className="busy ok">空闲</div>}
      </div>
    </div>
  )
}

export default App
