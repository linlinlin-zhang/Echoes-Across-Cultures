import { create } from 'zustand'

type AuditionFactor = 'zc' | 'zs' | 'za' | null

type SceneState = {
  separation: number
  hoveredSongId: string | null
  selectedSongId: string | null
  auditionFactor: AuditionFactor
  setSeparation: (value: number) => void
  setHoveredSongId: (value: string | null) => void
  setSelectedSongId: (value: string | null) => void
  setAuditionFactor: (value: AuditionFactor) => void
}

export const useSceneStore = create<SceneState>((set) => ({
  separation: 0,
  hoveredSongId: null,
  selectedSongId: null,
  auditionFactor: null,
  setSeparation: (value) => set({ separation: Math.max(0, Math.min(1, value)) }),
  setHoveredSongId: (value) => set({ hoveredSongId: value }),
  setSelectedSongId: (value) => set({ selectedSongId: value }),
  setAuditionFactor: (value) => set({ auditionFactor: value })
}))
