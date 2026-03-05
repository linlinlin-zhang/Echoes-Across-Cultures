import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: ['class'],
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}', './data/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        abyss: '#202124',
        indigoVoid: '#f6f7f8',
        zc: '#ea4335',
        zs: '#188038',
        za: '#1a73e8',
        textMain: '#202124',
        textSub: '#5f6368',
        glass: 'rgba(255,255,255,0.78)',
        paper: '#f6f7f8',
        ink: '#202124'
      },
      fontFamily: {
        display: ['Manrope', 'Noto Sans SC', 'Segoe UI', 'sans-serif'],
        body: ['Manrope', 'Noto Sans SC', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'Consolas', 'monospace']
      },
      boxShadow: {
        neon: '0 10px 30px rgba(32,33,36,0.07)',
        glow: '0 10px 30px rgba(26,115,232,0.14)'
      },
      backgroundImage: {
        deepGradient:
          'radial-gradient(circle at 14% 16%, rgba(26,115,232,0.09), transparent 36%), radial-gradient(circle at 86% 8%, rgba(24,128,56,0.08), transparent 34%), radial-gradient(circle at 52% 88%, rgba(234,67,53,0.08), transparent 42%), linear-gradient(180deg, #f6f7f8 0%, #eef1f5 52%, #e8edf4 100%)'
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
