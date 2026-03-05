import type { SongPoint } from '@/data/mock-data'
import { clamp } from '@/lib/utils'

export type FactorMetrics = {
  zcStrength: number
  zsStrength: number
  zaValence: number
  zaArousal: number
  melodySteps: number[]
  timbreBars: number[]
  rhythmDensity: number
  wave: OscillatorType
  filterHz: number
  attack: number
  release: number
  vibratoHz: number
  cultureDescriptorZh: string
  cultureDescriptorEn: string
}

const cultureTimbreMap: Record<string, { wave: OscillatorType; zh: string; en: string; filterBase: number }> = {
  'Western Pop': { wave: 'sawtooth', zh: '明亮合成器质感', en: 'bright synth-like timbre', filterBase: 2400 },
  'Indian Classical': { wave: 'triangle', zh: '弦乐滑音质感', en: 'plucked-glide timbre', filterBase: 1800 },
  'Turkish Makam': { wave: 'square', zh: '簧管/双簧类质感', en: 'reed-like timbre', filterBase: 1550 },
  Gamelan: { wave: 'square', zh: '金属敲击泛音', en: 'metallic percussive overtones', filterBase: 1450 },
  'West African Drumming': { wave: 'triangle', zh: '鼓组脉冲质感', en: 'pulse-heavy drum-like timbre', filterBase: 1300 },
  'Andean Folk': { wave: 'triangle', zh: '木管空气感', en: 'airy pipe-like timbre', filterBase: 1680 },
  Guqin: { wave: 'sine', zh: '柔和丝弦余韵', en: 'soft silk-string resonance', filterBase: 1180 },
  'Arabic Maqam': { wave: 'sawtooth', zh: '乌德琴明暗混合质感', en: 'oud-like mixed brightness timbre', filterBase: 1720 }
}

function normalize(value: number, min: number, max: number) {
  if (max - min === 0) return 0
  return clamp((value - min) / (max - min), 0, 1)
}

export function buildFactorMetrics(song: SongPoint): FactorMetrics {
  const z0 = normalize(song.zcVector[0], -1.2, 1.2)
  const z1 = normalize(song.zcVector[1], -1.2, 1.2)
  const z2 = normalize(song.zcVector[2], -1.2, 1.2)

  const s0 = normalize(song.zsVector[0], -1.1, 1.1)
  const s1 = normalize(song.zsVector[1], -1.1, 1.1)
  const s2 = normalize(song.zsVector[2], -1.1, 1.1)

  const valence = clamp(song.zaVector[0], -1, 1)
  const arousal = clamp(song.zaVector[1], -1, 1)
  const arousalNorm = normalize(arousal, -1, 1)

  const timbrePreset = cultureTimbreMap[song.culture] ?? {
    wave: 'triangle',
    zh: '中性文化音色',
    en: 'neutral cultural timbre',
    filterBase: 1600
  }

  const melodySteps = Array.from({ length: 8 }).map((_, i) => {
    const base = i % 3 === 0 ? z0 : i % 3 === 1 ? z1 : z2
    const swing = i % 2 === 0 ? 0.08 : -0.05
    return clamp(base + swing, 0.08, 1)
  })

  const timbreBars = [s0, s1, s2, (s0 + s1) / 2, (s1 + s2) / 2, (s0 + s2) / 2].map((value) => clamp(value, 0.05, 1))

  const rhythmDensity = clamp((z0 + z1 + z2) / 3, 0.08, 1)
  const zcStrength = rhythmDensity
  const zsStrength = clamp((s0 + s1 + s2) / 3, 0.08, 1)

  const filterHz = timbrePreset.filterBase + s1 * 1900 + (valence + 1) * 180
  const attack = 0.01 + (1 - arousalNorm) * 0.08
  const release = 0.18 + (1 - arousalNorm) * 0.4
  const vibratoHz = 4 + arousalNorm * 4.5

  return {
    zcStrength,
    zsStrength,
    zaValence: valence,
    zaArousal: arousal,
    melodySteps,
    timbreBars,
    rhythmDensity,
    wave: timbrePreset.wave,
    filterHz,
    attack,
    release,
    vibratoHz,
    cultureDescriptorZh: timbrePreset.zh,
    cultureDescriptorEn: timbrePreset.en
  }
}

export function factorSummaryZh(factor: 'zc' | 'zs' | 'za', song: SongPoint, metrics: FactorMetrics) {
  if (factor === 'zc') {
    return `zc 控制旋律/节奏：依据内容向量生成 ${Math.round(metrics.rhythmDensity * 100)}% 密度脉冲。`
  }
  if (factor === 'zs') {
    return `zs 控制文化/音色：${song.culture} 映射为 ${metrics.cultureDescriptorZh}。`
  }
  return `za 控制情感包络：Valence ${metrics.zaValence.toFixed(2)}，Arousal ${metrics.zaArousal.toFixed(2)}。`
}

export function factorSummaryEn(factor: 'zc' | 'zs' | 'za', song: SongPoint, metrics: FactorMetrics) {
  if (factor === 'zc') {
    return `zc drives melody/rhythm with ${Math.round(metrics.rhythmDensity * 100)}% pulse density from content vectors.`
  }
  if (factor === 'zs') {
    return `zs drives cultural timbre: ${song.culture} maps to ${metrics.cultureDescriptorEn}.`
  }
  return `za drives affective envelope: valence ${metrics.zaValence.toFixed(2)}, arousal ${metrics.zaArousal.toFixed(2)}.`
}
