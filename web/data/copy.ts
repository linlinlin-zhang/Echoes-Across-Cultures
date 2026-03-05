import type { Locale } from '@/components/ui/types'

export const sectionIds = [
  'hero',
  'problem',
  'architecture',
  'galaxy',
  'lab',
  'pal',
  'results',
  'ethics'
] as const

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
    subtitle: 'Deep Disentanglement for Cross-Cultural Music Recommendation',
    nav: {
      hero: '潜空间',
      problem: '问题',
      architecture: '架构',
      galaxy: '文化银河',
      lab: '解纠缠实验室',
      pal: '参与式学习',
      results: '结果',
      ethics: '伦理'
    },
    modeLabel: '语言',
    highContrast: '高对比',
    reduceMotion: '减少动效',
    heroTitle: '深度解纠缠与认知流形对齐',
    heroLead:
      '将音乐拆解为内容 zc、文化风格 zs、情感 za，让推荐系统在跨文化迁移中同时保持可解释性、机缘巧合与公平性。',
    heroHint: '拖拽旋转、滚轮缩放、悬停查看歌曲卡片、点击进入因子解剖视图。',
    ctaPrimary: '探索文化银河',
    ctaSecondary: '打开解纠缠实验室',
    sections: {
      problemTitle: '数字巴别塔：跨文化推荐困境',
      architectureTitle: 'DDRL 系统架构',
      galaxyTitle: '文化对齐银河',
      labTitle: '解纠缠实验室',
      palTitle: '参与式主动学习 PAL',
      resultsTitle: '推荐演示与公平性监控',
      ethicsTitle: '伦理宣言与合作入口'
    }
  },
  en: {
    brand: 'Soundscape Without Borders',
    subtitle: 'Deep Disentanglement for Cross-Cultural Music Recommendation',
    nav: {
      hero: 'Latent Space',
      problem: 'Problem',
      architecture: 'Architecture',
      galaxy: 'Culture Galaxy',
      lab: 'Lab',
      pal: 'PAL',
      results: 'Results',
      ethics: 'Ethics'
    },
    modeLabel: 'Language',
    highContrast: 'High Contrast',
    reduceMotion: 'Reduce Motion',
    heroTitle: 'Deep Disentanglement and Cognitive Manifold Alignment',
    heroLead:
      'Decompose music into content (zc), culture/style (zs), and affect (za), enabling explainable, serendipitous, and fair cross-cultural recommendation.',
    heroHint: 'Drag to rotate, wheel to zoom, hover particles for song cards, click to open latent anatomy.',
    ctaPrimary: 'Explore Culture Galaxy',
    ctaSecondary: 'Open Disentanglement Lab',
    sections: {
      problemTitle: 'The Digital Babel Problem',
      architectureTitle: 'DDRL Pipeline',
      galaxyTitle: 'Culture Alignment Galaxy',
      labTitle: 'Disentanglement Lab',
      palTitle: 'Participatory Active Learning',
      resultsTitle: 'Recommendation Demo and Fairness Dashboard',
      ethicsTitle: 'Ethics and Collaboration'
    }
  }
}
