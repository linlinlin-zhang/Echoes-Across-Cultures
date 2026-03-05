'use client'

import { useEffect } from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'

export default function Error({
  error,
  reset
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    console.error(error)
  }, [error])

  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col items-center justify-center bg-deepGradient px-6 text-center text-textMain">
      <div className="paper-card rounded-3xl px-8 py-8">
        <div className="mb-6 inline-flex h-14 w-14 items-center justify-center rounded-full border border-zc/40 bg-zc/10 text-zc">
          <AlertTriangle size={26} />
        </div>
        <h1 className="font-display text-3xl font-semibold md:text-4xl">页面渲染出现错误</h1>
        <p className="mt-4 max-w-xl text-base text-textSub md:text-lg">渲染引擎中断。请重试恢复交互模块。</p>
        <button
          onClick={reset}
          className="mt-8 inline-flex items-center gap-2 rounded-full border border-zs/40 bg-zs/10 px-5 py-2 font-semibold text-zs transition hover:bg-zs/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zs"
        >
          <RefreshCw size={16} />
          重新加载
        </button>
        {error.digest ? <p className="mt-4 font-mono text-xs text-textSub">digest: {error.digest}</p> : null}
      </div>
    </main>
  )
}