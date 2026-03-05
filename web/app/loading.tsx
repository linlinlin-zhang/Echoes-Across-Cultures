export default function Loading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-deepGradient p-6 text-center text-textMain">
      <div>
        <p className="font-display text-3xl">Loading Soundscape...</p>
        <p className="mt-3 font-mono text-sm text-textSub">Initializing latent space, culture graph, and interaction modules.</p>
      </div>
    </div>
  )
}
