import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

// Self-hosted so the UI works on a machine with no route to the internet, which
// the tailnet-only deployment frequently is. `wdth` is what gives Archivo
// Expanded for section labels (Spec 3 SS3).
import '@fontsource-variable/archivo/wdth.css'
import '@fontsource/martian-mono/400.css'
import '@fontsource/martian-mono/500.css'

import './styles/tokens.css'
import './styles/app.css'

import { App } from './App'
import { AppProvider } from './state/store'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AppProvider>
      <App />
    </AppProvider>
  </StrictMode>,
)
