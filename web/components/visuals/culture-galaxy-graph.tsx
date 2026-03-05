'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'

import { cultureLinks, cultureNodes, otDemoRoutes, type CultureLink, type CultureNode } from '@/data/mock-data'
import { clamp, cn } from '@/lib/utils'

type ViewMode = 'emotional' | 'structural'

type GraphNode = d3.SimulationNodeDatum & CultureNode & { x: number; y: number }
type GraphLink = Omit<d3.SimulationLinkDatum<GraphNode>, 'source' | 'target'> &
  Omit<CultureLink, 'source' | 'target'> & {
    source: string | number | GraphNode
    target: string | number | GraphNode
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

export function CultureGalaxyGraph() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [mode, setMode] = useState<ViewMode>('emotional')
  const [search, setSearch] = useState('')
  const [selectedNode, setSelectedNode] = useState<CultureNode | null>(cultureNodes[0])
  const [routeIndex, setRouteIndex] = useState(0)

  const activeRoutePairs = useMemo(() => makeRoutePairs(otDemoRoutes[routeIndex]), [routeIndex])

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
      .attr('aria-label', 'Culture alignment force-directed graph')

    const defs = svg.append('defs')
    defs
      .append('radialGradient')
      .attr('id', 'galaxy-core')
      .selectAll('stop')
      .data([
        { offset: '0%', color: '#4ecdc4', opacity: 0.5 },
        { offset: '60%', color: '#a55eea', opacity: 0.18 },
        { offset: '100%', color: '#0a0a0f', opacity: 0 }
      ])
      .enter()
      .append('stop')
      .attr('offset', (d) => d.offset)
      .attr('stop-color', (d) => d.color)
      .attr('stop-opacity', (d) => d.opacity)

    svg
      .append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', 'url(#galaxy-core)')

    const nodes: GraphNode[] = cultureNodes.map((item) => ({ ...item, x: width / 2, y: height / 2 }))

    const links: GraphLink[] = cultureLinks.map((link) => ({
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
          .distance((d) => 180 - (mode === 'structural' ? d.structural : d.emotional) * 90)
      )
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(58))

    const linkLayer = svg.append('g').attr('stroke-linecap', 'round')

    const linkSelection = linkLayer
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke', '#94a3b8')
      .attr('stroke-opacity', 0.3)
      .attr('stroke-width', (d) => 1 + (mode === 'structural' ? d.structural : d.emotional) * 6)

    const nodeLayer = svg.append('g')
    const nodeSelection = nodeLayer
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('tabindex', 0)
      .attr('role', 'button')
      .attr('aria-label', (d) => `Culture node ${d.name}`)
      .style('cursor', 'pointer')
      .on('click', (_event, d) => setSelectedNode(d))

    nodeSelection
      .append('circle')
      .attr('r', 32)
      .attr('fill', '#0f172a')
      .attr('stroke', '#4ecdc4')
      .attr('stroke-width', 1.5)

    nodeSelection
      .append('text')
      .text((d) => d.name)
      .attr('text-anchor', 'middle')
      .attr('dy', 4)
      .attr('font-size', 11)
      .attr('fill', '#e2e8f0')

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
          return isRouteEdge(sourceId, targetId, activeRoutePairs) ? '#4ecdc4' : '#94a3b8'
        })
        .attr('stroke-dasharray', (d) => {
          const sourceId = resolveEndpointId(d.source)
          const targetId = resolveEndpointId(d.target)
          return isRouteEdge(sourceId, targetId, activeRoutePairs) ? '8 6' : '0'
        })
        .attr('stroke-opacity', (d) => {
          const sourceId = resolveEndpointId(d.source)
          const targetId = resolveEndpointId(d.target)
          return isRouteEdge(sourceId, targetId, activeRoutePairs) ? 0.95 : 0.28
        })

      nodeSelection
        .select('circle')
        .attr('stroke', (d) => (selectedNode?.id === d.id ? '#ff6b6b' : '#4ecdc4'))
        .attr('stroke-width', (d) => (selectedNode?.id === d.id ? 3 : 1.5))
        .attr('opacity', (d) => {
          if (!search.trim()) return 1
          return d.name.toLowerCase().includes(search.trim().toLowerCase()) ? 1 : 0.2
        })
    })

    return () => {
      simulation.stop()
    }
  }, [mode, search, selectedNode?.id, activeRoutePairs])

  const scoreForSelected = useMemo(() => {
    if (!selectedNode) return 0
    const related = cultureLinks.filter((item) => item.source === selectedNode.id || item.target === selectedNode.id)
    if (!related.length) return 0
    const avg = related.reduce((acc, item) => acc + (mode === 'structural' ? item.structural : item.emotional), 0) / related.length
    return clamp(avg, 0, 1)
  }, [mode, selectedNode])

  return (
    <div className="grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
      <div className="rounded-3xl border border-white/10 bg-black/25 p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div className="inline-flex rounded-full border border-white/15 bg-black/20 p-1 text-sm">
            <button
              className={cn(
                'rounded-full px-3 py-1 transition',
                mode === 'emotional' ? 'bg-za/30 text-za' : 'text-textSub hover:text-textMain'
              )}
              onClick={() => setMode('emotional')}
            >
              Emotional Similarity (za)
            </button>
            <button
              className={cn(
                'rounded-full px-3 py-1 transition',
                mode === 'structural' ? 'bg-zc/30 text-zc' : 'text-textSub hover:text-textMain'
              )}
              onClick={() => setMode('structural')}
            >
              Structural Similarity (zc)
            </button>
          </div>
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search culture..."
            className="w-56 rounded-full border border-white/20 bg-black/30 px-3 py-1.5 text-sm text-textMain outline-none ring-zs placeholder:text-textSub focus:ring-2"
          />
        </div>

        <div ref={containerRef} className="h-[450px] w-full rounded-2xl border border-white/10 bg-black/30" />
      </div>

      <div className="space-y-4">
        <div className="rounded-3xl border border-white/10 bg-black/30 p-5">
          <h3 className="font-display text-2xl text-textMain">{selectedNode?.name}</h3>
          <p className="mt-1 text-sm text-textSub">{selectedNode?.family}</p>
          <p className="mt-3 text-sm leading-relaxed text-textSub">{selectedNode?.history}</p>

          <div className="mt-4 grid gap-2 text-sm text-textSub">
            <div>
              <span className="font-semibold text-textMain">Instruments: </span>
              {selectedNode?.instruments.join(', ')}
            </div>
            <div>
              <span className="font-semibold text-textMain">Scale / Grammar: </span>
              {selectedNode?.scaleSystem}
            </div>
          </div>

          <div className="mt-5">
            <div className="mb-1 flex items-center justify-between text-xs text-textSub">
              <span>Mode-conditioned connection strength</span>
              <span>{(scoreForSelected * 100).toFixed(0)}%</span>
            </div>
            <div className="h-2 rounded-full bg-white/10">
              <div
                className="h-full rounded-full bg-gradient-to-r from-zc via-zs to-za"
                style={{ width: `${(scoreForSelected * 100).toFixed(0)}%` }}
              />
            </div>
          </div>
        </div>

        <div className="rounded-3xl border border-white/10 bg-black/30 p-5">
          <h4 className="font-display text-lg text-textMain">Optimal Transport Demo Route</h4>
          <p className="mt-1 text-sm text-textSub">Watch preference mass flow from source culture to target culture.</p>
          <div className="mt-3 space-y-2">
            {otDemoRoutes.map((route, index) => (
              <button
                key={route.join('-')}
                className={cn(
                  'w-full rounded-xl border px-3 py-2 text-left text-sm transition',
                  routeIndex === index
                    ? 'border-zs/70 bg-zs/10 text-textMain'
                    : 'border-white/15 bg-black/20 text-textSub hover:border-white/40'
                )}
                onClick={() => setRouteIndex(index)}
              >
                {route.map((item, i) => (
                  <span key={`${item}-${i}`}>
                    {cultureNodes.find((node) => node.id === item)?.name ?? item}
                    {i === route.length - 1 ? '' : '  →  '}
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