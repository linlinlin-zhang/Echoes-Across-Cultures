export default function Loading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-deepGradient p-6 text-center text-textMain">
      <div className="rounded-3xl panel-deep px-8 py-6">
        <p className="font-display text-3xl">Loading Soundscape...</p>
        <p className="mt-3 font-mono text-sm text-textSub">Preparing maps, stories, and interaction modules.</p>
      </div>
    </div>
  )
}