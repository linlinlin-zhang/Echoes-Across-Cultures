# Soundscape Without Borders Web

Next.js 14 + TypeScript single-page immersive showcase for the DDRL cross-cultural music recommendation project (声界无疆).

## What This Frontend Demonstrates

- Deep disentanglement latent space with three factors:
  - `zc`: content (melody/rhythm)
  - `zs`: style/culture (instrumentation/context)
  - `za`: affect (valence/arousal)
- Cross-cultural manifold alignment and OT route visualization
- Interactive disentanglement lab with real-time controls and audio demo
- Participatory Active Learning (PAL) annotation loop and cognitive justice indicators
- Recommendation A/B demo with serendipity and fairness monitoring

## Tech Stack

- Framework: Next.js 14 (App Router) + React 18 + TypeScript
- Styling: Tailwind CSS + custom global CSS theme
- Motion: Framer Motion + GSAP ScrollTrigger
- 3D: Three.js + @react-three/fiber + @react-three/drei
- Data Viz: D3.js + Recharts
- Audio Demo: Tone.js + Web Audio API
- State: Zustand
- Icons: Lucide React

## Run Locally

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Production Checks

```bash
npm run lint
npm run build
npm run start
```

## Project Structure

```text
web/
  app/
    layout.tsx            # Root layout + provider mount
    page.tsx              # Main page entry
    loading.tsx           # Global loading state
    error.tsx             # Global error boundary UI
    globals.css           # Theme, typography, accessibility global styles
  components/
    site/
      soundscape-page.tsx # Main orchestration + section tracking + scroll animation hooks
    layout/
      top-nav.tsx         # Glass top nav + language/accessibility toggles
      side-nav.tsx        # Right-side dot progress nav
      section-shell.tsx   # Shared section container
    providers/
      accessibility-provider.tsx
    sections/
      hero-section.tsx
      problem-section.tsx
      architecture-section.tsx
      galaxy-section.tsx
      lab-section.tsx
      pal-section.tsx
      results-section.tsx
      ethics-section.tsx
    visuals/
      latent-space-canvas.tsx
      culture-galaxy-graph.tsx
      disentanglement-lab.tsx
      pal-interface.tsx
      recommendation-demo.tsx
      song-radar.tsx
    state/
      scene-store.ts
  data/
    copy.ts               # zh/en copy and section IDs
    mock-data.ts          # songs, cultures, links, OT routes, PAL and rec mock data
  hooks/
    use-media-query.ts
  lib/
    utils.ts
```

## Section Overview

1. Hero (3D latent space)
- Particle streams for `zc/zs/za`
- Scroll-driven disentanglement animation
- Hover/click reveals song anatomy card and radar

2. Problem
- Narrative framing of cross-cultural recommendation gaps

3. Architecture
- Stepwise DDRL pipeline explanation

4. Culture Galaxy
- D3 force graph of music cultures
- Emotional vs structural alignment modes
- OT route highlighting

5. Disentanglement Lab
- Real-time latent sliders (`zc/zs/za`)
- Tone.js playback demo and spectrum comparison
- Content contour, affect plane, and culture similarity meter

6. PAL Interface
- Uncertainty heatmap
- Expert annotation form + ontology expansion
- Feedback loop and coverage indicator

7. Results
- Traditional vs DDRL recommendation A/B panel
- Serendipity and fairness dashboards

8. Ethics
- Responsible AI principles and contact form placeholder

## Accessibility

- High contrast mode toggle
- Reduced motion mode toggle (also respects system preference)
- Keyboard-focusable controls
- Screen-reader support via aria labels and textual fallback descriptions

## Performance Notes

- 3D modules are dynamically imported (`ssr: false`)
- Instanced mesh for particle rendering
- Non-critical heavy components are lazy-loaded
- Mobile automatically uses simplified hero preview instead of full 3D canvas

## Data and Audio

- Uses mock data to demonstrate interaction flows
- Audio upload in lab is a UI placeholder; transfer logic currently demonstrates factor-controlled synthesis workflow

## Design System Tokens

- Background: `#0a0a0f -> #1a1a2e`
- `zc` accent: `#ff6b6b`
- `zs` accent: `#4ecdc4`
- `za` accent: `#a55eea`
- Main text: `#e0e0e0`
- Secondary text: `#8892b0`
- Fonts:
  - Display: Space Grotesk
  - Body: Crimson Pro
  - Mono: JetBrains Mono