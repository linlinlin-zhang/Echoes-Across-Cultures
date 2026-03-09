'use client'

import dynamic from 'next/dynamic'
import { useEffect, useMemo, useState, type ComponentType } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ArrowRight, BrainCircuit, Compass, GaugeCircle, Network, ShieldCheck, SlidersHorizontal, Users, Workflow } from 'lucide-react'

import { TopNav, type NavItem } from '@/components/layout/top-nav'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import type { Locale } from '@/components/ui/types'
import { copyByLocale } from '@/data/copy'
import { cultureNodes, otDemoRoutes, palUncertaintyGrid, songPoints } from '@/data/mock-data'
import { cn } from '@/lib/utils'

const RecommendationDemo = dynamic(() => import('@/components/visuals/recommendation-demo').then((mod) => mod.RecommendationDemo), {
  ssr: false,
  loading: () => <WorkbenchLoading />
})

const DisentanglementLab = dynamic(() => import('@/components/visuals/disentanglement-lab').then((mod) => mod.DisentanglementLab), {
  ssr: false,
  loading: () => <WorkbenchLoading />
})

const CultureGalaxyGraph = dynamic(() => import('@/components/visuals/culture-galaxy-graph').then((mod) => mod.CultureGalaxyGraph), {
  ssr: false,
  loading: () => <WorkbenchLoading />
})

const PalInterface = dynamic(() => import('@/components/visuals/pal-interface').then((mod) => mod.PalInterface), {
  ssr: false,
  loading: () => <WorkbenchLoading />
})

const pageSectionIds = ['overview', 'workspace', 'notes'] as const

type PageSectionId = (typeof pageSectionIds)[number]
type WorkbenchTabId = 'recommendations' | 'lab' | 'alignment' | 'pal'

type SignalCard = {
  id: string
  title: string
  detail: string
  color: string
}

type WorkbenchTabMeta = {
  id: WorkbenchTabId
  label: string
  shortLabel: string
  eyebrow: string
  title: string
  summary: string
  outcomes: string[]
  prompt: string
}

type PageCopy = {
  nav: Record<PageSectionId, string>
  overviewEyebrow: string
  overviewTitle: string
  overviewLead: string
  primaryAction: string
  secondaryAction: string
  quickModulesTitle: string
  quickModulesLead: string
  summaryTitle: string
  summaryLead: string
  structureTitle: string
  structureLead: string
  workbenchEyebrow: string
  workbenchTitle: string
  workbenchLead: string
  notesEyebrow: string
  notesTitle: string
  notesLead: string
  whyTitle: string
  whyItems: string[]
  systemTitle: string
  systemSteps: string[]
  guardrailsTitle: string
  guardrailsItems: string[]
  footer: string
  statusPrefix: string
  summaryStats: Array<{ label: string; value: string; hint: string }>
  signalCards: SignalCard[]
  workbenchTabs: WorkbenchTabMeta[]
}

const pageCopyByLocale: Record<Locale, PageCopy> = {
  zh: {
    nav: {
      overview: '项目总览',
      workspace: '功能工作台',
      notes: '研究说明'
    },
    overviewEyebrow: 'Cross-cultural music recommendation workbench',
    overviewTitle: '把首页从“长篇展览”改成可直接上手的工作台。',
    overviewLead:
      '保留现在这套轻盈、玻璃感、数据化的视觉语言，但把主路径改成四个核心模块。用户先用推荐、潜空间、文化对齐和 PAL 标注，再决定要不要继续读研究说明。',
    primaryAction: '打开推荐控制台',
    secondaryAction: '进入潜空间实验室',
    quickModulesTitle: '四个核心模块',
    quickModulesLead: '不再沿着八个章节一直滚到底，而是在同一个工作台里切换任务。',
    summaryTitle: '项目当前聚焦',
    summaryLead: '页面先回答“这个系统能做什么”，再回答“它为什么这样做”。',
    structureTitle: '新的浏览方式',
    structureLead: '总览负责建立心智模型，工作台负责实际交互，研究说明压缩为补充信息。',
    workbenchEyebrow: 'Feature workbench',
    workbenchTitle: '功能工作台',
    workbenchLead: '每个面板都保留原本的交互深度，但把叙事改成可切换、可停留、可反复试验的操作模式。',
    notesEyebrow: 'Research brief',
    notesTitle: '研究摘要与系统边界',
    notesLead: '如果用户想看方法与责任边界，这里只保留最关键的压缩信息，不再让整页被说明文字主导。',
    whyTitle: '为什么要做这件事',
    whyItems: [
      '语义坍缩：不同文化语法被压成单一距离，推荐结果看似统一，实际失真。',
      '情感漂移：相同标签跨文化迁移时失去语境，用户得到的是误读过的情绪近邻。',
      '曝光不平等：长尾与少数文化长期被挤出 Top-N，只剩表面上的“多样性”。'
    ],
    systemTitle: '系统如何修复这些问题',
    systemSteps: [
      'A. CultureMERT 主干先抽取稳定音频表征。',
      'B. 三因子编码器把内容 zc、文化 zs、情感 za 显式拆开。',
      'C. 泄漏抑制约束减少因子串扰，保证解释不互相污染。',
      'D. 最优传输把跨文化偏好迁移变成可见、可调的路线。',
      'E. PAL 把高不确定样本拉回人工协作，持续修补文化本体。'
    ],
    guardrailsTitle: '必须保持可见的边界',
    guardrailsItems: [
      '公开模型不确定区，而不是用一个总分掩盖薄弱带。',
      '持续跟踪少数文化曝光与修正回路，让公平不是口号。',
      '把协作、标注和本体扩展视为系统一部分，而不是上线后的补丁。'
    ],
    footer: '声界无疆 · 功能优先的跨文化音乐推荐界面',
    statusPrefix: '无障碍状态',
    summaryStats: [
      { label: '文化节点', value: `${cultureNodes.length}`, hint: '在文化对齐面板里可直接浏览与筛选。' },
      { label: '可试验样本', value: `${songPoints.length}`, hint: '推荐、潜空间和 PAL 共享同一批 mock 数据。' },
      { label: 'OT 路径', value: `${otDemoRoutes.length}`, hint: '跨文化迁移不再只是一句说明，而是可见路线。' },
      { label: '高不确定热点', value: `${palUncertaintyGrid.filter((item) => item.value >= 0.6).length}`, hint: 'PAL 面板直接用这些热点组织人工介入。' }
    ],
    signalCards: [
      { id: 'zc', title: '内容 zc', detail: '控制旋律与节奏骨架，决定歌曲本体最稳定的部分。', color: '#ea4335' },
      { id: 'zs', title: '文化 zs', detail: '把配器、语法与文化上下文从其它因素中拆出来。', color: '#188038' },
      { id: 'za', title: '情感 za', detail: '把效价与唤醒度作为可独立调节的表达维度。', color: '#1a73e8' }
    ],
    workbenchTabs: [
      {
        id: 'recommendations',
        label: '推荐控制台',
        shortLabel: '推荐',
        eyebrow: 'Control deck',
        title: '先看系统最后会给用户什么',
        summary: '这里直接比较基线推荐和 DDRL 输出，调节探索度、公平权重与情绪目标，观察结果结构如何变化。',
        outcomes: ['A/B 对比推荐结果', '文化曝光占比变化', '机缘巧合与公平性指标'],
        prompt: '适合先建立对系统价值的直觉。'
      },
      {
        id: 'lab',
        label: '潜空间实验室',
        shortLabel: '实验室',
        eyebrow: 'Latent lab',
        title: '把三因子调节变成直接可玩的控台',
        summary: '不用先读方法章节，直接调 zc / zs / za，听参数如何影响音色、节奏、情感和文化相似度。',
        outcomes: ['实时频谱与状态反馈', '可播放的音色迁移示例', '参数快照与随机组合'],
        prompt: '适合理解“解纠缠”到底带来了什么操作能力。'
      },
      {
        id: 'alignment',
        label: '文化对齐图',
        shortLabel: '对齐',
        eyebrow: 'Culture alignment',
        title: '把跨文化迁移从文字说明变成空间关系',
        summary: '在情感对齐和结构对齐之间切换，查看文化节点连接、检索某个传统，并跟踪最优传输路径。',
        outcomes: ['节点搜索与选择', '对齐模式切换', 'OT 路线高亮'],
        prompt: '适合解释为什么推荐不是黑箱地“跨文化跳转”。'
      },
      {
        id: 'pal',
        label: 'PAL 标注台',
        shortLabel: 'PAL',
        eyebrow: 'Participatory loop',
        title: '把人工协作放回系统主流程',
        summary: '直接处理高不确定样本、提交理由、扩展文化概念，并看 PAL 回合怎样改变覆盖度与风险。',
        outcomes: ['不确定热区筛选', '专家标注与理由输入', '概念扩展与回合指标'],
        prompt: '适合展示系统如何持续修补偏差，而不是一次性训练后冻结。'
      }
    ]
  },
  en: {
    nav: {
      overview: 'Overview',
      workspace: 'Workbench',
      notes: 'Research'
    },
    overviewEyebrow: 'Cross-cultural music recommendation workbench',
    overviewTitle: 'Turn the homepage from a long exhibit into a direct workbench.',
    overviewLead:
      'The visual language stays airy, glassy, and data-forward, but the primary path now centers on four core modules. Users can try recommendation, latent controls, alignment, and PAL first, then decide whether to read the research brief.',
    primaryAction: 'Open recommendation deck',
    secondaryAction: 'Launch latent lab',
    quickModulesTitle: 'Four core modules',
    quickModulesLead: 'Instead of forcing a full eight-chapter scroll, the homepage now switches tasks inside one workspace.',
    summaryTitle: 'Current project focus',
    summaryLead: 'The page answers what can this system do before why was it built this way.',
    structureTitle: 'New browsing logic',
    structureLead: 'Overview builds the mental model, the workbench handles interaction, and research notes stay compressed in the background.',
    workbenchEyebrow: 'Feature workbench',
    workbenchTitle: 'Function-first workspace',
    workbenchLead: 'Each panel keeps the depth of the original interactions, but the narrative has been restructured into switchable, reusable operating modes.',
    notesEyebrow: 'Research brief',
    notesTitle: 'Method summary and system boundaries',
    notesLead: 'If someone wants the method and guardrails, they can still inspect them here, but the page no longer lets explanation dominate the entire journey.',
    whyTitle: 'Why this system exists',
    whyItems: [
      'Semantic collapse: different cultural grammars get flattened into one distance metric.',
      'Affective drift: emotion labels travel across cultures without their social or ritual context.',
      'Exposure inequality: long-tail and minority cultures stay outside the top-N even when users want exploration.'
    ],
    systemTitle: 'How the system repairs that',
    systemSteps: [
      'A. A CultureMERT backbone extracts stable audio representations first.',
      'B. A tri-factor encoder separates content zc, culture zs, and affect za.',
      'C. Leakage suppression reduces factor bleed and keeps explanations distinct.',
      'D. Optimal transport makes cross-cultural preference shifts explicit and tunable.',
      'E. PAL brings high-uncertainty samples back into human collaboration.'
    ],
    guardrailsTitle: 'Boundaries that stay visible',
    guardrailsItems: [
      'Expose uncertainty regions instead of hiding weak zones behind one aggregate score.',
      'Track minority exposure and correction loops so fairness remains operational.',
      'Treat collaboration, annotation, and ontology expansion as part of the system, not post-launch decoration.'
    ],
    footer: 'Soundscape Without Borders · function-first cross-cultural recommendation interface',
    statusPrefix: 'Accessibility',
    summaryStats: [
      { label: 'Culture nodes', value: `${cultureNodes.length}`, hint: 'Browsable and filterable in the alignment module.' },
      { label: 'Usable samples', value: `${songPoints.length}`, hint: 'Recommendation, lab, and PAL share the same mock corpus.' },
      { label: 'OT routes', value: `${otDemoRoutes.length}`, hint: 'Cross-cultural transfer is presented as a visible path, not a claim.' },
      { label: 'High-uncertainty hotspots', value: `${palUncertaintyGrid.filter((item) => item.value >= 0.6).length}`, hint: 'PAL organizes human intervention around these samples.' }
    ],
    signalCards: [
      { id: 'zc', title: 'Content zc', detail: 'Controls melodic and rhythmic structure, the most stable musical backbone.', color: '#ea4335' },
      { id: 'zs', title: 'Culture zs', detail: 'Separates instrumentation, grammar, and cultural context from the rest.', color: '#188038' },
      { id: 'za', title: 'Affect za', detail: 'Keeps valence and arousal as independently controllable dimensions.', color: '#1a73e8' }
    ],
    workbenchTabs: [
      {
        id: 'recommendations',
        label: 'Recommendation Deck',
        shortLabel: 'Recs',
        eyebrow: 'Control deck',
        title: 'Start with the output users actually receive',
        summary: 'Compare baseline results with DDRL recommendations while tuning exploration, fairness weights, and mood targets.',
        outcomes: ['A/B recommendation comparison', 'Culture exposure shift', 'Serendipity and fairness signals'],
        prompt: 'Best for quickly understanding the product value.'
      },
      {
        id: 'lab',
        label: 'Latent Lab',
        shortLabel: 'Lab',
        eyebrow: 'Latent lab',
        title: 'Turn disentanglement into direct controls',
        summary: 'Adjust zc / zs / za without reading the method section first, then hear and inspect how the transfer changes.',
        outcomes: ['Realtime spectrum feedback', 'Playable transfer examples', 'Snapshots and surprise presets'],
        prompt: 'Best for seeing what the factor split actually enables.'
      },
      {
        id: 'alignment',
        label: 'Culture Alignment',
        shortLabel: 'Align',
        eyebrow: 'Culture alignment',
        title: 'Make cross-cultural transfer spatial and inspectable',
        summary: 'Switch between emotional and structural alignment, inspect node neighborhoods, search traditions, and trace OT routes.',
        outcomes: ['Searchable culture nodes', 'Alignment mode switching', 'Highlighted OT routes'],
        prompt: 'Best for explaining why transfer is not a black-box jump.'
      },
      {
        id: 'pal',
        label: 'PAL Console',
        shortLabel: 'PAL',
        eyebrow: 'Participatory loop',
        title: 'Put human collaboration back into the main flow',
        summary: 'Work through high-uncertainty samples, submit rationales, expand concepts, and inspect how PAL rounds shift coverage.',
        outcomes: ['Hotspot triage', 'Annotation and rationale input', 'Concept expansion and round metrics'],
        prompt: 'Best for showing how bias repair stays active after deployment.'
      }
    ]
  }
}

const tabAccentClass: Record<WorkbenchTabId, string> = {
  recommendations: 'border-za/40 bg-za/10 text-za',
  lab: 'border-zc/40 bg-zc/10 text-zc',
  alignment: 'border-zs/40 bg-zs/10 text-zs',
  pal: 'border-ink/25 bg-ink/5 text-textMain'
}

const workbenchComponentByTab: Record<WorkbenchTabId, ComponentType> = {
  recommendations: RecommendationDemo,
  lab: DisentanglementLab,
  alignment: CultureGalaxyGraph,
  pal: PalInterface
}

function WorkbenchLoading() {
  return <div className="min-h-[540px] animate-pulse rounded-[28px] bg-white/80" />
}

function OverviewModuleCard({
  label,
  detail,
  active,
  icon: Icon,
  onClick
}: {
  label: string
  detail: string
  active: boolean
  icon: ComponentType<{ size?: number | string }>
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'rounded-3xl border p-4 text-left transition-all duration-200',
        active ? 'border-za/35 bg-white shadow-[0_12px_36px_rgba(26,115,232,0.12)]' : 'paper-card hover:border-ink/30'
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-white text-za ring-1 ring-ink/10">
          <Icon size={18} />
        </div>
        <span className="chapter-chip">{active ? 'active' : 'module'}</span>
      </div>
      <h3 className="mt-4 font-display text-xl text-textMain">{label}</h3>
      <p className="mt-2 text-sm leading-relaxed text-textSub">{detail}</p>
    </button>
  )
}

export function SoundscapePage() {
  const [activeSection, setActiveSection] = useState<PageSectionId>('overview')
  const [activeTab, setActiveTab] = useState<WorkbenchTabId>('recommendations')
  const { locale, highContrast, reduceMotion } = useAccessibility()

  const chromeCopy = useMemo(() => copyByLocale[locale], [locale])
  const copy = useMemo(() => pageCopyByLocale[locale], [locale])

  const navItems: NavItem[] = useMemo(
    () => [
      { id: 'overview', label: copy.nav.overview },
      { id: 'workspace', label: copy.nav.workspace },
      { id: 'notes', label: copy.nav.notes }
    ],
    [copy.nav.notes, copy.nav.overview, copy.nav.workspace]
  )

  const tabIcons: Record<WorkbenchTabId, ComponentType<{ size?: number | string }>> = {
    recommendations: Compass,
    lab: SlidersHorizontal,
    alignment: Network,
    pal: Users
  }

  const activeTabMeta = useMemo(() => copy.workbenchTabs.find((item) => item.id === activeTab) ?? copy.workbenchTabs[0], [activeTab, copy.workbenchTabs])
  const ActiveWorkbench = workbenchComponentByTab[activeTab]

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)
        if (!visible.length) return
        const id = visible[0].target.getAttribute('data-page-section-id') as PageSectionId | null
        if (id && pageSectionIds.includes(id)) setActiveSection(id)
      },
      { threshold: [0.25, 0.55, 0.8] }
    )

    document.querySelectorAll<HTMLElement>('[data-page-section-id]').forEach((section) => observer.observe(section))
    return () => observer.disconnect()
  }, [])

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id)
    if (!element) return
    element.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' })
  }

  const openWorkbench = (id: WorkbenchTabId) => {
    setActiveTab(id)
    scrollToSection('workspace')
  }

  const fadeUp = reduceMotion
    ? {}
    : {
        initial: { opacity: 0, y: 20 },
        whileInView: { opacity: 1, y: 0 },
        viewport: { once: true, amount: 0.25 },
        transition: { duration: 0.45, ease: 'easeOut' }
      }

  return (
    <div className="relative min-h-screen bg-deepGradient text-textMain">
      <div className="star-fog" />
      <div className="noise-mask" />
      <div className="pointer-events-none fixed inset-0 z-[2] opacity-20 grid-hud" />

      <TopNav brand={chromeCopy.brand} items={navItems} activeItem={activeSection} onNavigate={scrollToSection} />

      <main className="relative z-10 pb-16 pt-24 md:pt-28">
        <section id="overview" data-page-section-id="overview" className="px-4 pb-8 pt-4 md:px-10">
          <div className="mx-auto grid max-w-7xl gap-6 xl:grid-cols-[1.1fr_0.9fr]">
            <motion.div {...fadeUp} className="rounded-[32px] panel-glass p-6 md:p-8">
              <div className="flex flex-wrap items-center gap-2">
                <span className="sticker">{copy.overviewEyebrow}</span>
                <span className="chapter-chip">{chromeCopy.subtitle}</span>
              </div>

              <h1 className="mt-5 max-w-4xl font-display text-4xl font-semibold leading-[0.95] text-textMain md:text-6xl">{copy.overviewTitle}</h1>
              <p className="mt-4 max-w-3xl text-base leading-relaxed text-textSub md:text-lg">{copy.overviewLead}</p>

              <div className="mt-6 flex flex-wrap gap-3">
                <button
                  onClick={() => openWorkbench('recommendations')}
                  className="group inline-flex items-center gap-2 rounded-full bg-za px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-za/90"
                >
                  {copy.primaryAction}
                  <ArrowRight size={15} className="transition group-hover:translate-x-0.5" />
                </button>
                <button
                  onClick={() => openWorkbench('lab')}
                  className="inline-flex items-center gap-2 rounded-full border border-ink/20 bg-white px-5 py-2.5 text-sm font-semibold text-textMain transition hover:border-zc/40 hover:bg-zc/5"
                >
                  {copy.secondaryAction}
                </button>
              </div>

              <div className="mt-8">
                <div className="mb-4">
                  <p className="chapter-chip">{copy.quickModulesTitle}</p>
                  <p className="mt-2 max-w-3xl text-sm leading-relaxed text-textSub">{copy.quickModulesLead}</p>
                </div>

                <div className="grid gap-3 md:grid-cols-2">
                  {copy.workbenchTabs.map((tab) => {
                    const Icon = tabIcons[tab.id]
                    return <OverviewModuleCard key={tab.id} label={tab.label} detail={tab.summary} active={activeTab === tab.id} icon={Icon} onClick={() => openWorkbench(tab.id)} />
                  })}
                </div>
              </div>
            </motion.div>

            <div className="flex flex-col gap-4">
              <motion.div {...fadeUp} className="rounded-[32px] panel-deep p-6">
                <p className="chapter-chip">{copy.summaryTitle}</p>
                <h2 className="mt-3 font-display text-3xl text-textMain">{copy.summaryLead}</h2>
                <div className="mt-5 grid gap-3 sm:grid-cols-2">
                  {copy.summaryStats.map((stat) => (
                    <div key={stat.label} className="note-card">
                      <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-textSub">{stat.label}</p>
                      <p className="mt-2 font-display text-3xl text-textMain">{stat.value}</p>
                      <p className="mt-2 text-sm leading-relaxed text-textSub">{stat.hint}</p>
                    </div>
                  ))}
                </div>
              </motion.div>

              <motion.div {...fadeUp} className="rounded-[32px] panel-deep p-6">
                <p className="chapter-chip">{copy.structureTitle}</p>
                <p className="mt-3 text-sm leading-relaxed text-textSub">{copy.structureLead}</p>
                <div className="mt-5 grid gap-3">
                  {copy.signalCards.map((item) => (
                    <div key={item.id} className="rounded-2xl border border-ink/15 bg-white/90 p-4">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <p className="font-display text-xl text-textMain">{item.title}</p>
                          <p className="mt-1 text-sm leading-relaxed text-textSub">{item.detail}</p>
                        </div>
                        <div className="h-10 w-10 rounded-full" style={{ background: `${item.color}18`, border: `1px solid ${item.color}38` }} />
                      </div>
                      <div className="mt-3 h-1.5 rounded-full bg-ink/10">
                        <div className="h-full rounded-full" style={{ width: '100%', backgroundColor: item.color }} />
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            </div>
          </div>
        </section>

        <section id="workspace" data-page-section-id="workspace" className="px-4 py-8 md:px-10">
          <div className="mx-auto max-w-7xl">
            <motion.div {...fadeUp} className="mb-5 flex flex-wrap items-end justify-between gap-4">
              <div className="max-w-3xl">
                <p className="chapter-chip">{copy.workbenchEyebrow}</p>
                <h2 className="mt-3 font-display text-4xl font-semibold leading-[0.97] text-textMain md:text-5xl">{copy.workbenchTitle}</h2>
                <p className="mt-3 text-base leading-relaxed text-textSub">{copy.workbenchLead}</p>
              </div>
              <div className="rounded-2xl border border-ink/15 bg-white/80 px-4 py-3 text-sm text-textSub">
                <span className="mr-2 font-mono uppercase tracking-[0.12em]">{activeTabMeta.eyebrow}</span>
                <span className="font-semibold text-textMain">{activeTabMeta.label}</span>
              </div>
            </motion.div>

            <motion.div {...fadeUp} className="rounded-[36px] panel-glass p-4 md:p-5">
              <div className="flex gap-2 overflow-x-auto pb-1">
                {copy.workbenchTabs.map((tab) => {
                  const Icon = tabIcons[tab.id]
                  const active = tab.id === activeTab

                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={cn(
                        'inline-flex min-w-fit items-center gap-2 rounded-full border px-4 py-2 text-sm font-semibold transition',
                        active ? tabAccentClass[tab.id] : 'border-ink/15 bg-white/80 text-textSub hover:border-ink/30 hover:text-textMain'
                      )}
                    >
                      <Icon size={15} />
                      {tab.label}
                    </button>
                  )
                })}
              </div>

              <div className="mt-5 grid gap-4 xl:grid-cols-[300px_1fr]">
                <div className="rounded-[28px] panel-deep p-5 xl:sticky xl:top-24 xl:h-fit">
                  <span className="sticker">{activeTabMeta.eyebrow}</span>
                  <h3 className="mt-4 font-display text-3xl leading-tight text-textMain">{activeTabMeta.title}</h3>
                  <p className="mt-3 text-sm leading-relaxed text-textSub">{activeTabMeta.summary}</p>

                  <div className="mt-5 space-y-2">
                    {activeTabMeta.outcomes.map((item) => (
                      <div key={item} className="rounded-2xl border border-ink/15 bg-white/85 px-3 py-2 text-sm text-textMain">
                        {item}
                      </div>
                    ))}
                  </div>

                  <div className="mt-5 rounded-2xl border border-zs/30 bg-zs/10 p-4 text-sm leading-relaxed text-textMain">{activeTabMeta.prompt}</div>

                  <div className="mt-5 space-y-2">
                    {copy.workbenchTabs.map((tab) => {
                      const active = tab.id === activeTab
                      return (
                        <button
                          key={`switch-${tab.id}`}
                          onClick={() => setActiveTab(tab.id)}
                          className={cn(
                            'flex w-full items-center justify-between rounded-2xl border px-3 py-2 text-left text-sm transition',
                            active ? tabAccentClass[tab.id] : 'border-ink/15 bg-white/80 text-textSub hover:text-textMain'
                          )}
                        >
                          <span>{tab.shortLabel}</span>
                          <ArrowRight size={14} />
                        </button>
                      )
                    })}
                  </div>
                </div>

                <div className="rounded-[28px] border border-ink/10 bg-white/86 p-2 md:p-3">
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={activeTab}
                      initial={reduceMotion ? undefined : { opacity: 0, y: 10 }}
                      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
                      exit={reduceMotion ? undefined : { opacity: 0, y: -10 }}
                      transition={reduceMotion ? undefined : { duration: 0.24, ease: 'easeOut' }}
                    >
                      <ActiveWorkbench />
                    </motion.div>
                  </AnimatePresence>
                </div>
              </div>
            </motion.div>
          </div>
        </section>

        <section id="notes" data-page-section-id="notes" className="px-4 pt-8 md:px-10">
          <div className="mx-auto max-w-7xl">
            <motion.div {...fadeUp} className="mb-6 max-w-3xl">
              <p className="chapter-chip">{copy.notesEyebrow}</p>
              <h2 className="mt-3 font-display text-4xl font-semibold leading-[0.98] text-textMain md:text-5xl">{copy.notesTitle}</h2>
              <p className="mt-3 text-base leading-relaxed text-textSub">{copy.notesLead}</p>
            </motion.div>

            <div className="grid gap-4 xl:grid-cols-[1fr_1fr_1fr]">
              <motion.div {...fadeUp} className="rounded-[30px] paper-card p-6">
                <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-zc/12 text-zc">
                  <GaugeCircle size={18} />
                </div>
                <h3 className="mt-4 font-display text-2xl text-textMain">{copy.whyTitle}</h3>
                <div className="mt-4 space-y-3">
                  {copy.whyItems.map((item) => (
                    <div key={item} className="note-card text-sm leading-relaxed text-textSub">
                      {item}
                    </div>
                  ))}
                </div>
              </motion.div>

              <motion.div {...fadeUp} className="rounded-[30px] paper-card p-6">
                <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-za/12 text-za">
                  <Workflow size={18} />
                </div>
                <h3 className="mt-4 font-display text-2xl text-textMain">{copy.systemTitle}</h3>
                <div className="mt-4 space-y-3">
                  {copy.systemSteps.map((item, index) => (
                    <div key={item} className="rounded-2xl border border-ink/15 bg-white/90 p-4">
                      <div className="flex items-start gap-3">
                        <span className="inline-flex h-7 w-7 flex-none items-center justify-center rounded-full bg-ink/5 font-mono text-xs text-textMain">{index + 1}</span>
                        <p className="text-sm leading-relaxed text-textSub">{item}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>

              <motion.div {...fadeUp} className="rounded-[30px] paper-card p-6">
                <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-zs/12 text-zs">
                  <ShieldCheck size={18} />
                </div>
                <h3 className="mt-4 font-display text-2xl text-textMain">{copy.guardrailsTitle}</h3>
                <div className="mt-4 space-y-3">
                  {copy.guardrailsItems.map((item) => (
                    <div key={item} className="note-card text-sm leading-relaxed text-textSub">
                      {item}
                    </div>
                  ))}
                </div>
                <button
                  onClick={() => openWorkbench('pal')}
                  className="mt-5 inline-flex items-center gap-2 rounded-full border border-zs/30 bg-zs/10 px-4 py-2 text-sm font-semibold text-zs transition hover:bg-zs/15"
                >
                  <BrainCircuit size={15} />
                  {locale === 'zh' ? '返回 PAL 标注台' : 'Return to PAL console'}
                </button>
              </motion.div>
            </div>
          </div>
        </section>
      </main>

      <footer className="relative z-10 mt-10 border-t border-ink/15 bg-white/75 px-4 py-8 md:px-10">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 text-xs text-textSub">
          <p>{copy.footer}</p>
          <p>
            {locale === 'zh'
              ? `${copy.statusPrefix}：${highContrast ? '高对比已开启' : '标准对比'} · ${reduceMotion ? '减少动画已开启' : '动态模式'}`
              : `${copy.statusPrefix}: ${highContrast ? 'High Contrast On' : 'Standard Contrast'} · ${reduceMotion ? 'Reduced Motion On' : 'Motion On'}`}
          </p>
        </div>
      </footer>
    </div>
  )
}


