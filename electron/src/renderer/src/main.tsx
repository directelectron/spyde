import React from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import { AppFailed, ErrorBoundary } from './components/ErrorBoundary'

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary scope="app" fallback={(error, retry) => <AppFailed error={error} retry={retry} />}>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
)
