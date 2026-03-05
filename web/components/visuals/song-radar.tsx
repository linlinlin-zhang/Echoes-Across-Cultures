import type { SongPoint } from '@/data/mock-data'

function toPolar([x, y, z]: [number, number, number]) {
  const ax = Math.abs(x)
  const ay = Math.abs(y)
  const az = Math.abs(z)
  return [Number((ax % 1.4).toFixed(2)), Number((ay % 1.4).toFixed(2)), Number((az % 1.4).toFixed(2))]
}

export function SongRadar({ song }: { song: SongPoint }) {
  const [m, r, c] = toPolar(song.zcVector)
  const [inst, color, context] = toPolar(song.zsVector)
  const [valence, arousal] = song.zaVector

  const stats = [
    { label: 'Melody', value: m, color: 'var(--zc)' },
    { label: 'Rhythm', value: r, color: 'var(--zc)' },
    { label: 'Contour', value: c, color: 'var(--zc)' },
    { label: 'Instrumental', value: inst, color: 'var(--zs)' },
    { label: 'Color', value: color, color: 'var(--zs)' },
    { label: 'Context', value: context, color: 'var(--zs)' },
    { label: 'Valence', value: Math.abs(valence), color: 'var(--za)' },
    { label: 'Arousal', value: Math.abs(arousal), color: 'var(--za)' }
  ]

  return (
    <div className="paper-card rounded-2xl p-4">
      <div className="mb-3">
        <span className="chapter-chip">song anatomy</span>
        <h4 className="mt-2 font-display text-xl text-textMain">{song.title}</h4>
        <p className="text-sm text-textSub">{song.culture} · {song.emotion}</p>
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