'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'

import { cultureLinks, cultureNodes, otDemoRoutes, type CultureLink, type CultureNode } from '@/data/mock-data'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { bilingual, cultureLabel } from '@/lib/bilingual'
import { clamp, cn } from '@/lib/utils'

type ViewMode = 'emotional' | 'structural'

type GraphNode = d3.SimulationNodeDatum & CultureNode & { x: number; y: number }
type GraphLink = Omit<d3.SimulationLinkDatum<GraphNode>, 'source' | 'target'> &
  Omit<CultureLink, 'source' | 'target'> & {
    source: string | number | GraphNode
    target: string | number | GraphNode
  }

type ZhNodeMeta = {
  name: string
  family: string
  instruments: string[]
  scaleSystem: string
  history: string
}

const zhMetaById: Record<string, ZhNodeMeta> = {
  'western-pop': {
    name: '西方流行',
    family: '全球主流',
    instruments: ['合成器', '架子鼓', '电贝斯'],
    scaleSystem: '十二平均律',
    history: '沿工业时代录音棚生产体系发展，和声规整度高，传播网络成熟。'
  },
  'indian-classical': {
    name: '印度古典',
    family: '南亚传统',
    instruments: ['西塔琴', '萨罗德', '塔布拉鼓'],
    scaleSystem: '拉格-塔拉体系',
    history: '围绕拉格身份与时间性 rasa 展开的即兴语法，强调情境与延展。'
  },
  'turkish-makam': {
    name: '土耳其玛卡姆',
    family: '安纳托利亚传统',
    instruments: ['奈伊笛', '坦布尔', '卡农琴'],
    scaleSystem: '玛卡姆-乌苏尔体系',
    history: '以细腻音高偏移和循环节律构成模态轨迹，旋律走向高度程式化。'
  },
  gamelan: {
    name: '甘美兰',
    family: '东南亚传统',
    instruments: ['锣', '金属琴', '肯当鼓'],
    scaleSystem: 'Slendro-Pelog 调式',
    history: '层叠互锁织体与周期性锣点结构并重，重在群体协同和时间轮回感。'
  },
  'west-african-drumming': {
    name: '西非鼓乐',
    family: '西非传统',
    instruments: ['金贝鼓', '敦敦鼓', '铃'],
    scaleSystem: '复节奏模式系统',
    history: '社区参与式节奏组织，强调身体律动与多层周期叠加。'
  },
  'andean-folk': {
    name: '安第斯民谣',
    family: '拉丁美洲传统',
    instruments: ['查兰戈', '克纳笛', '邦博鼓'],
    scaleSystem: '五声音阶-调式混合',
    history: '山地仪式与舞蹈曲目传统，融合本土与殖民时期音乐层。'
  },
  guqin: {
    name: '中国古琴',
    family: '东亚文人传统',
    instruments: ['古琴'],
    scaleSystem: '减字谱调式实践',
    history: '强调音色细节、留白与哲思叙事，重视个体修身语境。'
  },
  'arabic-maqam': {
    name: '阿拉伯玛卡姆',
    family: '中东传统',
    instruments: ['乌德琴', '卡农琴', '里克鼓'],
    scaleSystem: '玛卡姆-伊卡体系',
    history: '以微分音模态与旋律发展为核心，强调情感表达与行腔转折。'
  }
}

function makeRoutePairs(route: string[]) {
  const pairs: Array<[string, string]> = []
  for (let i = 0; i < route.length - 1; i += 1) {
    pairs.push([route[i], route[i + 1]])
  }
  return pairs
}

function resolveEndpointId(endpoint: string | number | GraphNode) {
  if (typeof endpoint === 'string') return endpoint
  if (typeof endpoint === 'number') return String(endpoint)
  return endpoint.id
}

function isRouteEdge(sourceId: string, targetId: string, pairs: Array<[string, string]>) {
  return pairs.some((pair) => (pair[0] === sourceId && pair[1] === targetId) || (pair[0] === targetId && pair[1] === sourceId))
}

function nodeShortLabel(node: CultureNode, locale: 'zh' | 'en') {
  if (locale !== 'zh') return node.name
  return zhMetaById[node.id]?.name ?? node.name
}

export function CultureGalaxyGraph() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [mode, setMode] = useState<ViewMode>('emotional')
  const [search, setSearch] = useState('')
  const [selectedNode, setSelectedNode] = useState<CultureNode | null>(cultureNodes[0])
  const [routeIndex, setRouteIndex] = useState(0)
  const [linkThreshold, setLinkThreshold] = useState(0.25)
  const [autoRoute, setAutoRoute] = useState(false)
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  useEffect(() => {
    if (!autoRoute) return
    const timer = window.setInterval(() => {
      setRouteIndex((prev) => (prev + 1) % otDemoRoutes.length)
    }, 2600)
    return () => window.clearInterval(timer)
  }, [autoRoute])

  const activeRoutePairs = useMemo(() => makeRoutePairs(otDemoRoutes[routeIndex]), [routeIndex])

  const filteredLinks = useMemo(() => {
    return cultureLinks.filter((item) => {
      const strength = mode === 'structural' ? item.structural : item.emotional
      return strength >= linkThreshold
    })
  }, [linkThreshold, mode])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    container.innerHTML = ''
    const width = container.clientWidth
    const height = Math.max(420, Math.round(container.clientHeight))

    const svg = d3
      .select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('role', 'img')
      .attr('aria-label', isZh ? '文化对齐力导向图（Culture Alignment Force-Directed Graph）' : 'Culture alignment force-directed graph')

    const defs = svg.append('defs')
    defs
      .append('radialGradient')
      .attr('id', 'galaxy-core')
      .selectAll('stop')
      .data([
        { offset: '0%', color: '#f7fbff', opacity: 0.96 },
        { offset: '64%', color: '#edf4ff', opacity: 0.55 },
        { offset: '100%', color: '#e8f0fe', opacity: 0.2 }
      ])
      .enter()
      .append('stop')
      .attr('offset', (d) => d.offset)
      .attr('stop-color', (d) => d.color)
      .attr('stop-opacity', (d) => d.opacity)

    svg.append('rect').attr('width', width).attr('height', height).attr('fill', 'url(#galaxy-core)')

    const nodes: GraphNode[] = cultureNodes.map((item) => ({ ...item, x: width / 2, y: height / 2 }))

    const links: GraphLink[] = filteredLinks.map((link) => ({
      ...link,
      source: link.source,
      target: link.target
    }))

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        'link',
        d3
          .forceLink<GraphNode, GraphLink>(links)
          .id((d) => d.id)
          .distance((d) => 188 - (mode === 'structural' ? d.structural : d.emotional) * 102)
      )
      .force('charge', d3.forceManyBody().strength(-340))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(58))

    const linkLayer = svg.append('g').attr('stroke-linecap', 'round')

    const linkSelection = linkLayer
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke', '#8ea1c8')
      .attr('stroke-opacity', 0.35)
      .attr('stroke-width', (d) => 1 + (mode === 'structural' ? d.structural : d.emotional) * 7)

    const nodeLayer = svg.append('g')
    const nodeSelection = nodeLayer
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('tabindex', 0)
      .attr('role', 'button')
      .attr('aria-label', (d) => (isZh ? `文化节点：${nodeShortLabel(d, locale)}` : `Culture node ${d.name}`))
      .style('cursor', 'pointer')
      .on('click', (_event, d) => setSelectedNode(d))

    const dragBehavior = d3
      .drag<SVGGElement, GraphNode>()
      .on('start', (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart()
        d.fx = d.x
        d.fy = d.y
      })
      .on('drag', (event, d) => {
        d.fx = event.x
        d.fy = event.y
      })
      .on('end', (event, d) => {
        if (!event.active) simulation.alphaTarget(0)
        d.fx = null
        d.fy = null
      })

    nodeSelection.call(dragBehavior)

    nodeSelection.append('circle').attr('r', 33).attr('fill', '#ffffff').attr('stroke', '#188038').attr('stroke-width', 1.5)

    nodeSelection
      .append('text')
      .text((d) => nodeShortLabel(d, locale))
      .attr('text-anchor', 'middle')
      .attr('dy', 4)
      .attr('font-size', 11)
      .attr('fill', '#2a3753')

    simulation.on('tick', () => {
      linkSelection
        .attr('x1', (d) => (d.source as GraphNode).x)
        .attr('y1', (d) => (d.source as GraphNode).y)
        .attr('x2', (d) => (d.target as GraphNode).x)
        .attr('y2', (d) => (d.target as GraphNode).y)

      nodeSelection.attr('transform', (d) => `translate(${d.x},${d.y})`)

      linkSelection
        .attr('stroke', (d) => {
          const sourceId = resolveEndpointId(d.source)
          const targetId = resolveEndpointId(d.target)
          return isRouteEdge(sourceId, targetId, activeRoutePairs) ? '#188038' : '#8ea1c8'
        })
        .attr('stroke-dasharray', (d) => {
          const sourceId = resolveEndpointId(d.source)
          const targetId = resolveEndpointId(d.target)
          return isRouteEdge(sourceId, targetId, activeRoutePairs) ? '8 6' : '0'
        })
        .attr('stroke-opacity', (d) => {
          const sourceId = resolveEndpointId(d.source)
          const targetId = resolveEndpointId(d.target)
          return isRouteEdge(sourceId, targetId, activeRoutePairs) ? 1 : 0.3
        })

      nodeSelection
        .select('circle')
        .attr('stroke', (d) => (selectedNode?.id === d.id ? '#ea4335' : '#188038'))
        .attr('stroke-width', (d) => (selectedNode?.id === d.id ? 3 : 1.5))
        .attr('opacity', (d) => {
          if (!search.trim()) return 1
          const query = search.trim().toLowerCase()
          const english = d.name.toLowerCase()
          const chinese = (zhMetaById[d.id]?.name ?? '').toLowerCase()
          return english.includes(query) || chinese.includes(query) ? 1 : 0.18
        })
    })

    return () => {
      simulation.stop()
    }
  }, [mode, search, selectedNode?.id, activeRoutePairs, isZh, locale, filteredLinks])

  const scoreForSelected = useMemo(() => {
    if (!selectedNode) return 0
    const related = filteredLinks.filter((item) => item.source === selectedNode.id || item.target === selectedNode.id)
    if (!related.length) return 0
    const avg = related.reduce((acc, item) => acc + (mode === 'structural' ? item.structural : item.emotional), 0) / related.length
    return clamp(avg, 0, 1)
  }, [filteredLinks, mode, selectedNode])

  const selectedZhMeta = selectedNode ? zhMetaById[selectedNode.id] : null

  return (
    <div className="grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
      <div className="rounded-3xl paper-card p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div className="inline-flex rounded-full border border-ink/15 bg-white p-1 text-sm">
            <button className={cn('rounded-full px-3 py-1 transition', mode === 'emotional' ? 'bg-za/15 text-za' : 'text-textSub hover:text-textMain')} onClick={() => setMode('emotional')}>
              {isZh ? '情感相似（Emotional Similarity, za）' : 'Emotional Similarity (za)'}
            </button>
            <button className={cn('rounded-full px-3 py-1 transition', mode === 'structural' ? 'bg-zc/15 text-zc' : 'text-textSub hover:text-textMain')} onClick={() => setMode('structural')}>
              {isZh ? '结构相似（Structural Similarity, zc）' : 'Structural Similarity (zc)'}
            </button>
          </div>
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={isZh ? '搜索文化节点（Search Culture）...' : 'Search culture...'}
            aria-label={isZh ? '搜索文化节点' : 'Search culture nodes'}
            className="w-56 rounded-full border border-ink/20 bg-white px-3 py-1.5 text-sm text-textMain outline-none ring-zs placeholder:text-textSub focus:ring-2"
          />
        </div>

        <div className="mb-3 grid gap-2 sm:grid-cols-[1fr_auto_auto] sm:items-end">
          <label className="block">
            <div className="mb-1 flex items-center justify-between text-xs text-textSub">
              <span>{isZh ? '连接阈值（Link Threshold）' : 'Link Threshold'}</span>
              <span>{(linkThreshold * 100).toFixed(0)}%</span>
            </div>
            <input type="range" min={0} max={0.8} step={0.01} value={linkThreshold} onChange={(event) => setLinkThreshold(Number(event.target.value))} className="w-full accent-za" />
          </label>
          <button
            onClick={() => setAutoRoute((prev) => !prev)}
            className={cn('rounded-full border px-3 py-1.5 text-xs font-semibold transition', autoRoute ? 'border-zs/35 bg-zs/10 text-zs' : 'border-ink/20 bg-white text-textSub hover:text-textMain')}
          >
            {isZh ? (autoRoute ? '停止巡航（Stop Auto Route）' : '自动巡航（Auto Route）') : autoRoute ? 'Stop Auto Route' : 'Auto Route'}
          </button>
          <button
            onClick={() => setRouteIndex(Math.floor(Math.random() * otDemoRoutes.length))}
            className="rounded-full border border-ink/20 bg-white px-3 py-1.5 text-xs font-semibold text-textSub transition hover:text-textMain"
          >
            {isZh ? '随机路径（Random Route）' : 'Random Route'}
          </button>
        </div>

        <div ref={containerRef} className="h-[450px] w-full rounded-2xl border border-ink/15 bg-[#f7faff]" />
      </div>

      <div className="space-y-4">
        <div className="rounded-3xl paper-card p-5">
          <h3 className="font-display text-2xl text-textMain">
            {selectedNode ? (isZh ? bilingual(locale, selectedZhMeta?.name ?? selectedNode.name, selectedNode.name) : selectedNode.name) : ''}
          </h3>
          <p className="mt-1 text-sm text-textSub">
            {selectedNode ? (isZh ? bilingual(locale, selectedZhMeta?.family ?? selectedNode.family, selectedNode.family) : selectedNode.family) : ''}
          </p>
          <p className="mt-3 text-sm leading-relaxed text-textSub">{selectedNode ? (isZh ? `${selectedZhMeta?.history ?? ''}（${selectedNode.history}）` : selectedNode.history) : ''}</p>

          <div className="mt-4 grid gap-2 text-sm text-textSub">
            <div>
              <span className="font-semibold text-textMain">{isZh ? '乐器（Instruments）：' : 'Instruments: '}</span>
              {selectedNode
                ? isZh
                  ? `${(selectedZhMeta?.instruments ?? []).join('、')}（${selectedNode.instruments.join(', ')}）`
                  : selectedNode.instruments.join(', ')
                : ''}
            </div>
            <div>
              <span className="font-semibold text-textMain">{isZh ? '音阶 / 语法（Scale / Grammar）：' : 'Scale / Grammar: '}</span>
              {selectedNode ? (isZh ? `${selectedZhMeta?.scaleSystem ?? ''}（${selectedNode.scaleSystem}）` : selectedNode.scaleSystem) : ''}
            </div>
          </div>

          <div className="mt-5">
            <div className="mb-1 flex items-center justify-between text-xs text-textSub">
              <span>{isZh ? '当前模式连接强度（Mode-Conditioned Connection Strength）' : 'Mode-conditioned connection strength'}</span>
              <span>{(scoreForSelected * 100).toFixed(0)}%</span>
            </div>
            <div className="h-2 rounded-full bg-ink/10">
              <div className="h-full rounded-full bg-gradient-to-r from-zc via-zs to-za" style={{ width: `${(scoreForSelected * 100).toFixed(0)}%` }} />
            </div>
          </div>

          <p className="mt-3 text-xs text-textSub">
            {isZh
              ? `当前保留连边 ${filteredLinks.length} 条（Visible Links: ${filteredLinks.length}）`
              : `Visible links: ${filteredLinks.length}`}
          </p>
        </div>

        <div className="rounded-3xl paper-card p-5">
          <h4 className="font-display text-lg text-textMain">{isZh ? '最优传输演示路径（Optimal Transport Demo Route）' : 'Optimal Transport Demo Route'}</h4>
          <p className="mt-1 text-sm text-textSub">{isZh ? '演示用户偏好质量如何从源文化流向目标文化。' : 'Watch preference mass flow from source culture to target culture.'}</p>
          <div className="mt-3 space-y-2">
            {otDemoRoutes.map((route, index) => (
              <button
                key={route.join('-')}
                className={cn(
                  'w-full rounded-xl border px-3 py-2 text-left text-sm transition',
                  routeIndex === index ? 'border-zs/55 bg-zs/10 text-textMain' : 'border-ink/15 bg-white text-textSub hover:border-ink/35'
                )}
                onClick={() => setRouteIndex(index)}
              >
                {route.map((item, i) => (
                  <span key={`${item}-${i}`}>
                    {cultureLabel(locale, cultureNodes.find((node) => node.id === item)?.name ?? item)}
                    {i === route.length - 1 ? '' : '  ->  '}
                  </span>
                ))}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

