import { useEffect, useMemo, useState } from 'react'
import { JobRail, filterJobs } from './components/JobRail'
import { StatusBar } from './components/StatusBar'
import { HelpOverlay } from './components/HelpOverlay'
import { ComparePage } from './pages/ComparePage'
import { JobPage } from './pages/JobPage'
import { JobsPage } from './pages/JobsPage'
import { NewJobPage } from './pages/NewJobPage'
import { navigate, useRoute } from './router'
import { useApp } from './state/store'

export function App() {
  const route = useRoute()
  const { jobs, stateFilter, message } = useApp()
  const [helpOpen, setHelpOpen] = useState(false)
  const [focusedId, setFocusedId] = useState<string | null>(null)

  const selectedId = route.name === 'job' ? route.id : null

  // The rail's own ordering, so j/k walk what is actually on screen.
  const ordered = useMemo(
    () => filterJobs(jobs, { states: stateFilter, text: '', sort: 'newest' }),
    [jobs, stateFilter],
  )

  useEffect(() => {
    if (selectedId) setFocusedId(selectedId)
  }, [selectedId])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      const typing =
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)
      if (event.metaKey || event.ctrlKey || event.altKey) return

      if (event.key === 'Escape') {
        setHelpOpen(false)
        if (typing) (target as HTMLElement).blur()
        return
      }
      if (typing) return

      if (event.key === '?') {
        event.preventDefault()
        setHelpOpen((open) => !open)
        return
      }
      if (event.key === '/') {
        event.preventDefault()
        document.getElementById('rail-filter')?.focus()
        return
      }
      if (event.key === 'n') {
        event.preventDefault()
        navigate({ name: 'new' })
        return
      }
      if (event.key === 'j' || event.key === 'k') {
        event.preventDefault()
        if (ordered.length === 0) return
        const index = ordered.findIndex((job) => job.id === focusedId)
        const next =
          event.key === 'j'
            ? Math.min(ordered.length - 1, index + 1)
            : Math.max(0, (index === -1 ? 0 : index) - 1)
        setFocusedId(ordered[next].id)
        return
      }
      if (event.key === 'Enter' && focusedId) {
        event.preventDefault()
        navigate({ name: 'job', id: focusedId })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [ordered, focusedId])

  return (
    <>
      <div className="too-narrow">
        <strong>SLDgen needs a wider window.</strong>
        <p className="note">
          The prep canvas and the contact sheet both need room. This interface declares a minimum
          width of 900px rather than degrading into something that cannot do either.
        </p>
      </div>

      <div className="shell">
        <header className="header">
          <span className="header__mark">SLDgen</span>
          <button
            type="button"
            className="btn btn--primary"
            onClick={() => navigate({ name: 'new' })}
          >
            New job
          </button>
          <span className="header__spacer" />
          <nav>
            <button
              type="button"
              aria-current={route.name === 'jobs' || route.name === 'job'}
              onClick={() => navigate({ name: 'jobs' })}
            >
              Jobs
            </button>
            <button
              type="button"
              aria-current={route.name === 'compare'}
              onClick={() => navigate({ name: 'compare', ids: [] })}
            >
              Compare
            </button>
            <button type="button" onClick={() => setHelpOpen(true)} title="Keyboard shortcuts">
              ?
            </button>
          </nav>
        </header>

        {/* The rail and the status bar never unmount, so the running job is
            visible no matter which route you are on (SS4). */}
        <JobRail selectedId={selectedId} focusedId={focusedId} />

        <main className="main">
          {route.name === 'jobs' && <JobsPage />}
          {route.name === 'job' && <JobPage jobId={route.id} />}
          {route.name === 'compare' && <ComparePage ids={route.ids} />}
          {route.name === 'new' && <NewJobPage />}
        </main>

        <StatusBar />
      </div>

      {helpOpen && <HelpOverlay onClose={() => setHelpOpen(false)} />}
      {message && <div className="toast">{message}</div>}
    </>
  )
}
