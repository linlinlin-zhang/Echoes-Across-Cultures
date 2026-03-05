import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: ['class'],
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}', './data/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        abyss: '#1f2430',
        indigoVoid: '#eef2ff',
        zc: '#ff6f61',
        zs: '#00a7a0',
        za: '#7e57c2',
        textMain: '#20232d',
        textSub: '#596172',
        glass: 'rgba(255, 255, 255, 0.8)',
        paper: '#f7f2e9',
        ink: '#1f2430'
      },
      fontFamily: {
        display: ['Fraunces', 'Georgia', 'serif'],
        body: ['DM Sans', 'Segoe UI', 'sans-serif'],
        mono: ['IBM Plex Mono', 'JetBrains Mono', 'Consolas', 'monospace']
      },
      boxShadow: {
        neon: '0 10px 28px rgba(42, 49, 67, 0.12)',
        glow: '0 8px 20px rgba(255, 111, 97, 0.18)'
      },
      backgroundImage: {
        deepGradient:
          'radial-gradient(circle at 15% 20%, rgba(255,111,97,0.16), transparent 45%), radial-gradient(circle at 80% 10%, rgba(0,167,160,0.18), transparent 40%), linear-gradient(180deg, #f7f2e9 0%, #f2efe7 100%)'
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