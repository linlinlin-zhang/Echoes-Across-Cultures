import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: ['class'],
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}', './data/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        abyss: '#0a0a0f',
        indigoVoid: '#1a1a2e',
        zc: '#ff6b6b',
        zs: '#4ecdc4',
        za: '#a55eea',
        textMain: '#e0e0e0',
        textSub: '#8892b0',
        glass: 'rgba(15, 23, 42, 0.5)'
      },
      fontFamily: {
        display: ['Space Grotesk', 'Inter', 'system-ui', 'sans-serif'],
        body: ['Crimson Pro', 'Georgia', 'serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace']
      },
      boxShadow: {
        neon: '0 0 24px rgba(165, 94, 234, 0.35), 0 0 36px rgba(78, 205, 196, 0.2)',
        glow: '0 0 20px rgba(255, 107, 107, 0.35)'
      },
      backgroundImage: {
        deepGradient: 'radial-gradient(circle at 20% 20%, rgba(165,94,234,0.22), transparent 40%), radial-gradient(circle at 80% 10%, rgba(78,205,196,0.2), transparent 35%), linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%)'
      },
      animation: {
        floatSlow: 'floatSlow 8s ease-in-out infinite',
        pulseGlow: 'pulseGlow 3s ease-in-out infinite'
      },
      keyframes: {
        floatSlow: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-12px)' }
        },
        pulseGlow: {
          '0%, 100%': { opacity: '0.75' },
          '50%': { opacity: '1' }
        }
      }
    }
  },
  plugins: []
}

export default config
