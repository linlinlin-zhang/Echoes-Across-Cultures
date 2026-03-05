import { create } from 'zustand'

type SceneState = {
  separation: number
  hoveredSongId: string | null
  selectedSongId: string | null
  setSeparation: (value: number) => void
  setHoveredSongId: (value: string | null) => void
  setSelectedSongId: (value: string | null) => void
}

export const useSceneStore = create<SceneState>((set) => ({
  separation: 0,
  hoveredSongId: null,
  selectedSongId: null,
  setSeparation: (value) => set({ separation: Math.max(0, Math.min(1, value)) }),
  setHoveredSongId: (value) => set({ hoveredSongId: value }),
  setSelectedSongId: (value) => set({ selectedSongId: value })
}))
