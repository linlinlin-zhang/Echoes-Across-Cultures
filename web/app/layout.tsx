import type { Metadata } from 'next'

import '@/app/globals.css'
import { AccessibilityProvider } from '@/components/providers/accessibility-provider'

export const metadata: Metadata = {
  title: 'Soundscape Without Borders | DDRL Cross-Cultural Recommender',
  description:
    'Immersive interactive showcase for cross-cultural music recommendation with deep disentanglement representation learning.'
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body className="font-body antialiased">
        <AccessibilityProvider>{children}</AccessibilityProvider>
      </body>
    </html>
  )
}
