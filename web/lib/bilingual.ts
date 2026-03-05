import type { Locale } from '@/components/ui/types'

type Dict = Record<string, string>

const cultureZhMap: Dict = {
  'Western Pop': '西方流行',
  'Indian Classical': '印度古典',
  'Turkish Makam': '土耳其玛卡姆',
  Gamelan: '甘美兰',
  'West African Drumming': '西非鼓乐',
  'Andean Folk': '安第斯民谣',
  Guqin: '中国古琴',
  'Arabic Maqam': '阿拉伯玛卡姆'
}

const emotionZhMap: Dict = {
  Contemplative: '沉思',
  Ecstatic: '狂喜',
  Melancholic: '忧郁',
  Transcendent: '超验',
  Grounded: '沉稳'
}

export function bilingual(locale: Locale, zh: string, en: string) {
  return locale === 'zh' ? `${zh}（${en}）` : en
}

export function fromDict(locale: Locale, value: string, dict: Dict) {
  const zh = dict[value]
  if (!zh) return value
  return bilingual(locale, zh, value)
}

export function cultureLabel(locale: Locale, value: string) {
  return fromDict(locale, value, cultureZhMap)
}

export function emotionLabel(locale: Locale, value: string) {
  return fromDict(locale, value, emotionZhMap)
}

export function songTitleLabel(locale: Locale, title: string) {
  if (locale !== 'zh') return title
  const match = title.match(/^Echo Piece\s+(\d+)$/i)
  if (!match) return title
  return `回声片段 ${match[1]}（${title}）`
}
