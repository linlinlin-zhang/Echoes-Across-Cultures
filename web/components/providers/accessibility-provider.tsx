'use client'

import { createContext, useContext, useEffect, useMemo, useState } from 'react'

type Locale = 'zh' | 'en'

type AccessibilityContextValue = {
  locale: Locale
  setLocale: (locale: Locale) => void
  highContrast: boolean
  setHighContrast: (value: boolean) => void
  reduceMotion: boolean
  setReduceMotion: (value: boolean) => void
}

const AccessibilityContext = createContext<AccessibilityContextValue | null>(null)

const STORAGE_KEYS = {
  locale: 'soundscape-locale',
  highContrast: 'soundscape-high-contrast',
  reduceMotion: 'soundscape-reduce-motion'
}

export function AccessibilityProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocale] = useState<Locale>('zh')
  const [highContrast, setHighContrast] = useState(false)
  const [reduceMotion, setReduceMotion] = useState(false)

  useEffect(() => {
    const cachedLocale = window.localStorage.getItem(STORAGE_KEYS.locale) as Locale | null
    const cachedContrast = window.localStorage.getItem(STORAGE_KEYS.highContrast)
    const cachedMotion = window.localStorage.getItem(STORAGE_KEYS.reduceMotion)

    if (cachedLocale === 'zh' || cachedLocale === 'en') {
      setLocale(cachedLocale)
    }

    if (cachedContrast) {
      setHighContrast(cachedContrast === '1')
    }

    if (cachedMotion) {
      setReduceMotion(cachedMotion === '1')
    } else {
      const prefersReduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      setReduceMotion(prefersReduce)
    }
  }, [])

  useEffect(() => {
    document.documentElement.classList.toggle('high-contrast', highContrast)
    document.documentElement.classList.toggle('reduce-motion', reduceMotion)
    document.documentElement.lang = locale === 'zh' ? 'zh-CN' : 'en'
    window.localStorage.setItem(STORAGE_KEYS.locale, locale)
    window.localStorage.setItem(STORAGE_KEYS.highContrast, highContrast ? '1' : '0')
    window.localStorage.setItem(STORAGE_KEYS.reduceMotion, reduceMotion ? '1' : '0')
  }, [locale, highContrast, reduceMotion])

  const value = useMemo(
    () => ({
      locale,
      setLocale,
      highContrast,
      setHighContrast,
      reduceMotion,
      setReduceMotion
    }),
    [locale, highContrast, reduceMotion]
  )

  return <AccessibilityContext.Provider value={value}>{children}</AccessibilityContext.Provider>
}

export function useAccessibility() {
  const context = useContext(AccessibilityContext)
  if (!context) {
    throw new Error('useAccessibility must be used within AccessibilityProvider')
  }
  return context
}
