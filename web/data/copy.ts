import type { Locale } from '@/components/ui/types'

export const sectionIds = ['hero', 'problem', 'architecture', 'galaxy', 'lab', 'pal', 'results', 'ethics'] as const

export type SectionId = (typeof sectionIds)[number]

type Copy = {
  brand: string
  subtitle: string
  nav: Record<SectionId, string>
  modeLabel: string
  highContrast: string
  reduceMotion: string
  heroTitle: string
  heroLead: string
  heroHint: string
  ctaPrimary: string
  ctaSecondary: string
  sections: {
    problemTitle: string
    architectureTitle: string
    galaxyTitle: string
    labTitle: string
    palTitle: string
    resultsTitle: string
    ethicsTitle: string
  }
}

export const copyByLocale: Record<Locale, Copy> = {
  zh: {
    brand: '声界无疆',
    subtitle: '跨文化音乐推荐的深度解纠缠（Deep Disentanglement for Cross-Cultural Music Recommendation）',
    nav: {
      hero: '潜空间序章',
      problem: '问题叙事',
      architecture: '任务架构',
      galaxy: '文化银河',
      lab: '解纠缠实验室',
      pal: '参与式学习',
      results: '结果控制台',
      ethics: '伦理宣言'
    },
    modeLabel: '语言',
    highContrast: '高对比',
    reduceMotion: '减少动效',
    heroTitle: '深度解纠缠与认知流形对齐',
    heroLead:
      '这不是一个“换歌器”，而是一个跨文化音乐认知界面：把音乐分解为内容 zc、文化风格 zs、情感 za，再通过最优传输与参与式反馈实现可解释、公平、并具机缘巧合的推荐。',
    heroHint: '支持键盘触发交互音垫、拖拽旋转潜空间、悬停粒子查看歌曲解剖卡。',
    ctaPrimary: '进入文化银河',
    ctaSecondary: '启动实验室',
    sections: {
      problemTitle: '数字巴别塔：跨文化推荐为何失真',
      architectureTitle: 'DDRL 任务控制架构',
      galaxyTitle: '文化对齐银河',
      labTitle: '解纠缠实验室',
      palTitle: '参与式主动学习（PAL）',
      resultsTitle: '推荐结果与公平性控制台',
      ethicsTitle: '伦理承诺与协作入口'
    }
  },
  en: {
    brand: 'Soundscape Without Borders',
    subtitle: 'Deep Disentanglement for Cross-Cultural Music Recommendation',
    nav: {
      hero: 'Prelude',
      problem: 'Narrative',
      architecture: 'Mission',
      galaxy: 'Galaxy',
      lab: 'Lab',
      pal: 'PAL',
      results: 'Control Deck',
      ethics: 'Ethics'
    },
    modeLabel: 'Language',
    highContrast: 'High Contrast',
    reduceMotion: 'Reduce Motion',
    heroTitle: 'Deep Disentanglement and Cognitive Manifold Alignment',
    heroLead:
      'This is not just a recommender. It is a cross-cultural music cognition interface: disentangle content (zc), culture/style (zs), and affect (za), then align preference flow with OT and participatory feedback.',
    heroHint: 'Use keyboard pads, orbit latent space, hover particles, and inspect per-song factor anatomy.',
    ctaPrimary: 'Enter Culture Galaxy',
    ctaSecondary: 'Launch Lab',
    sections: {
      problemTitle: 'Digital Babel: Why Cross-Cultural Recommenders Drift',
      architectureTitle: 'DDRL Mission Architecture',
      galaxyTitle: 'Culture Alignment Galaxy',
      labTitle: 'Disentanglement Lab',
      palTitle: 'Participatory Active Learning',
      resultsTitle: 'Recommendation Control Deck',
      ethicsTitle: 'Ethics and Collaboration'
    }
  }
}
