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
    <div className="rounded-2xl border border-white/15 bg-white/5 p-4 backdrop-blur-md">
      <div className="mb-3">
        <h4 className="font-display text-lg text-textMain">{song.title}</h4>
        <p className="font-body text-sm text-textSub">{song.culture} · {song.emotion}</p>
      </div>
      <div className="grid grid-cols-2 gap-2">
        {stats.map((item) => (
          <div key={item.label} className="rounded-lg bg-black/20 px-2 py-1.5">
            <div className="flex items-center justify-between text-xs text-textSub">
              <span>{item.label}</span>
              <span>{item.value.toFixed(2)}</span>
            </div>
            <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-white/10">
              <div className="h-full rounded-full" style={{ width: `${Math.min(100, item.value * 80)}%`, background: item.color }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
