'use client'

import type { SongPoint } from '@/data/mock-data'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { bilingual, cultureLabel, emotionLabel, songTitleLabel } from '@/lib/bilingual'

function toPolar([x, y, z]: [number, number, number]) {
  const ax = Math.abs(x)
  const ay = Math.abs(y)
  const az = Math.abs(z)
  return [Number((ax % 1.4).toFixed(2)), Number((ay % 1.4).toFixed(2)), Number((az % 1.4).toFixed(2))]
}

export function SongRadar({ song }: { song: SongPoint }) {
  const { locale } = useAccessibility()
  const [m, r, c] = toPolar(song.zcVector)
  const [inst, color, context] = toPolar(song.zsVector)
  const [valence, arousal] = song.zaVector

  const stats = [
    { label: bilingual(locale, '旋律', 'Melody'), value: m, color: 'var(--zc)' },
    { label: bilingual(locale, '节奏', 'Rhythm'), value: r, color: 'var(--zc)' },
    { label: bilingual(locale, '轮廓', 'Contour'), value: c, color: 'var(--zc)' },
    { label: bilingual(locale, '乐器性', 'Instrumental'), value: inst, color: 'var(--zs)' },
    { label: bilingual(locale, '音色', 'Color'), value: color, color: 'var(--zs)' },
    { label: bilingual(locale, '语境', 'Context'), value: context, color: 'var(--zs)' },
    { label: bilingual(locale, '效价', 'Valence'), value: Math.abs(valence), color: 'var(--za)' },
    { label: bilingual(locale, '唤醒度', 'Arousal'), value: Math.abs(arousal), color: 'var(--za)' }
  ]

  return (
    <div className="paper-card rounded-2xl p-4">
      <div className="mb-3">
        <span className="chapter-chip">{locale === 'zh' ? '歌曲解剖（Song Anatomy）' : 'song anatomy'}</span>
        <h4 className="mt-2 font-display text-xl text-textMain">{songTitleLabel(locale, song.title)}</h4>
        <p className="text-sm text-textSub">
          {cultureLabel(locale, song.culture)} · {emotionLabel(locale, song.emotion)}
        </p>
      </div>
      <div className="grid grid-cols-2 gap-2">
        {stats.map((item) => (
          <div key={item.label} className="stat-tile">
            <div className="flex items-center justify-between text-xs text-textSub">
              <span>{item.label}</span>
              <span>{item.value.toFixed(2)}</span>
            </div>
            <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-ink/10">
              <div className="h-full rounded-full" style={{ width: `${Math.min(100, item.value * 80)}%`, background: item.color }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
