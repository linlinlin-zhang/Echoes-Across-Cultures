export type LatentFactor = 'zc' | 'zs' | 'za'

export type SongPoint = {
  id: string
  title: string
  culture: string
  emotion: string
  zcVector: [number, number, number]
  zsVector: [number, number, number]
  zaVector: [number, number]
  audioUrl: string
}

export type CultureNode = {
  id: string
  name: string
  family: string
  instruments: string[]
  scaleSystem: string
  history: string
  ontologyCoverage: number
}

export type CultureLink = {
  source: string
  target: string
  structural: number
  emotional: number
}

const cultures = [
  'Western Pop',
  'Indian Classical',
  'Turkish Makam',
  'Gamelan',
  'West African Drumming',
  'Andean Folk',
  'Guqin',
  'Arabic Maqam'
] as const

const emotions = ['Contemplative', 'Ecstatic', 'Melancholic', 'Transcendent', 'Grounded'] as const

function seededRandom(seed: number) {
  let t = seed
  return () => {
    t = (t * 9301 + 49297) % 233280
    return t / 233280
  }
}

const rand = seededRandom(42)

function r3(scale = 1): [number, number, number] {
  return [
    (rand() * 2 - 1) * scale,
    (rand() * 2 - 1) * scale,
    (rand() * 2 - 1) * scale
  ]
}

function r2(scale = 1): [number, number] {
  return [(rand() * 2 - 1) * scale, (rand() * 2 - 1) * scale]
}

export const songPoints: SongPoint[] = Array.from({ length: 48 }).map((_, i) => {
  const culture = cultures[i % cultures.length]
  const emotion = emotions[i % emotions.length]
  return {
    id: `song-${i + 1}`,
    title: `Echo Piece ${String(i + 1).padStart(2, '0')}`,
    culture,
    emotion,
    zcVector: r3(1.2),
    zsVector: r3(1.1),
    zaVector: r2(1.0),
    audioUrl: `/audio/demo-${(i % 4) + 1}.mp3`
  }
})

export const cultureNodes: CultureNode[] = [
  {
    id: 'western-pop',
    name: 'Western Pop',
    family: 'Global Mainstream',
    instruments: ['Synth', 'Drum Kit', 'Electric Bass'],
    scaleSystem: 'Equal Temperament',
    history: 'Industrial-era studio production lineage with high harmonic regularity.',
    ontologyCoverage: 0.94
  },
  {
    id: 'indian-classical',
    name: 'Indian Classical',
    family: 'South Asian',
    instruments: ['Sitar', 'Sarod', 'Tabla'],
    scaleSystem: 'Raga-Tala',
    history: 'Improvisational grammar centered around raga identity and temporal rasa.',
    ontologyCoverage: 0.76
  },
  {
    id: 'turkish-makam',
    name: 'Turkish Makam',
    family: 'Anatolian',
    instruments: ['Ney', 'Tanbur', 'Kanun'],
    scaleSystem: 'Makam-Usul',
    history: 'Modal trajectories with nuanced pitch inflection and cyclical rhythm.',
    ontologyCoverage: 0.62
  },
  {
    id: 'gamelan',
    name: 'Gamelan',
    family: 'Southeast Asian',
    instruments: ['Gong', 'Metallophone', 'Kendang'],
    scaleSystem: 'Slendro-Pelog',
    history: 'Layered interlocking textures and cyclical gong punctuation.',
    ontologyCoverage: 0.58
  },
  {
    id: 'west-african-drumming',
    name: 'West African Drumming',
    family: 'West African',
    instruments: ['Djembe', 'Dunun', 'Bell'],
    scaleSystem: 'Polyrhythmic Pattern Systems',
    history: 'Community-based rhythmic structures emphasizing participatory groove.',
    ontologyCoverage: 0.69
  },
  {
    id: 'andean-folk',
    name: 'Andean Folk',
    family: 'Latin American',
    instruments: ['Charango', 'Quena', 'Bombo'],
    scaleSystem: 'Pentatonic-Modal Hybrids',
    history: 'Mountain ritual and dance repertoires blending indigenous and colonial layers.',
    ontologyCoverage: 0.57
  },
  {
    id: 'guqin',
    name: 'Guqin',
    family: 'East Asian',
    instruments: ['Guqin'],
    scaleSystem: 'Jianzipu-based modal practice',
    history: 'Scholar tradition focused on timbral nuance, silence, and philosophical narrative.',
    ontologyCoverage: 0.51
  },
  {
    id: 'arabic-maqam',
    name: 'Arabic Maqam',
    family: 'Middle Eastern',
    instruments: ['Oud', 'Qanun', 'Riqq'],
    scaleSystem: 'Maqam-Iqa',
    history: 'Microtonal modal systems with expressive melodic development.',
    ontologyCoverage: 0.65
  }
]

const baseLinks: Array<[string, string]> = [
  ['western-pop', 'indian-classical'],
  ['western-pop', 'turkish-makam'],
  ['western-pop', 'gamelan'],
  ['indian-classical', 'arabic-maqam'],
  ['turkish-makam', 'arabic-maqam'],
  ['gamelan', 'guqin'],
  ['west-african-drumming', 'andean-folk'],
  ['west-african-drumming', 'western-pop'],
  ['andean-folk', 'guqin'],
  ['indian-classical', 'guqin'],
  ['gamelan', 'west-african-drumming'],
  ['arabic-maqam', 'andean-folk']
]

export const cultureLinks: CultureLink[] = baseLinks.map(([source, target], i) => ({
  source,
  target,
  structural: 0.25 + ((i * 0.13) % 0.55),
  emotional: 0.22 + ((i * 0.17) % 0.6)
}))

export const otDemoRoutes = [
  ['western-pop', 'indian-classical', 'arabic-maqam'],
  ['western-pop', 'gamelan', 'guqin'],
  ['western-pop', 'west-african-drumming', 'andean-folk']
]

export const palUncertaintyGrid = Array.from({ length: 64 }).map((_, i) => {
  const value = 0.1 + ((Math.sin(i * 0.71) + 1) * 0.45)
  return {
    id: `u-${i + 1}`,
    value: Number(Math.min(0.99, value).toFixed(3)),
    culture: cultures[i % cultures.length],
    hint: songPoints[i % songPoints.length].title
  }
})

export const baselineRecommendations = songPoints.slice(0, 10).map((song, i) => ({
  rank: i + 1,
  title: song.title,
  culture: i < 7 ? 'Western Pop' : song.culture,
  relevance: Number((0.78 - i * 0.03).toFixed(3)),
  unexpectedness: Number((0.22 + i * 0.02).toFixed(3))
}))

export const dcasRecommendations = songPoints.slice(12, 22).map((song, i) => ({
  rank: i + 1,
  title: song.title,
  culture: song.culture,
  relevance: Number((0.8 - i * 0.022).toFixed(3)),
  unexpectedness: Number((0.41 + i * 0.03).toFixed(3))
}))
