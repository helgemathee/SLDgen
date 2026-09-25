import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { RenameResult } from '../api/types'
import { jobLabel } from '../lib/format'
import { useApp } from '../state/store'

/**
 * "Call these 7 jobs foo" — the rename for the rail's ticked selection.
 *
 * You type one base name; each job becomes `foo · s<seed>`, with `· v01`,
 * `· v02` where the seed alone does not tell them apart. The preview is the
 * server's dry run of exactly the rename it then performs, because the server
 * is the one that knows the real seeds and every other job's title — a name is
 * only unique if it is checked against all of them.
 */
export function RenameJobsDialog({
  ids,
  onClose,
}: {
  ids: string[]
  onClose: () => void
}) {
  const { jobsById, refreshJobs, toast } = useApp()
  const [base, setBase] = useState(() => {
    const first = jobsById.get(ids[0])
    return first?.title?.split(' · ')[0] ?? ''
  })
  const [preview, setPreview] = useState<RenameResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    setPreview(null)
    setError(null)
    if (!base.trim()) return
    let cancelled = false
    const timer = setTimeout(() => {
      api
        .renameJobs(ids, base, true)
        .then((result) => {
          if (!cancelled) setPreview(result)
        })
        .catch((problem: unknown) => {
          if (!cancelled) setError(problem instanceof Error ? problem.message : 'Could not preview')
        })
    }, 200)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [ids, base])

  const confirm = async () => {
    setBusy(true)
    try {
      const result = await api.renameJobs(ids, base, false)
      toast(`Renamed ${result.items.length} ${result.items.length === 1 ? 'job' : 'jobs'}.`)
      refreshJobs()
      onClose()
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : 'Rename failed')
      setBusy(false)
    }
  }

  return (
    <div className="overlay" onClick={onClose} role="presentation">
      <div
        className="overlay__card"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Rename selected jobs"
      >
        <div className="panel__head">
          <span className="eyebrow">Rename</span>
          <button type="button" className="btn btn--small" onClick={onClose}>
            Close
          </button>
        </div>

        <form
          className="panel__body"
          style={{ display: 'grid', gap: 10 }}
          onSubmit={(event) => {
            event.preventDefault()
            if (preview && !busy) confirm()
          }}
        >
          <label style={{ display: 'grid', gap: 4 }}>
            <span>
              New name for{' '}
              <strong>{ids.length === 1 ? 'this job' : `these ${ids.length} jobs`}</strong>
            </span>
            <input
              className="input"
              autoFocus
              value={base}
              onChange={(event) => setBase(event.target.value)}
              placeholder="e.g. owl"
            />
          </label>

          <div className="note">
            Each job gets its seed appended, read from what it actually ran with — not from its
            old title. Where the seed alone does not make the name unique, a v01, v02 … follows.
          </div>

          <div className="mono" style={{ maxHeight: 260, overflow: 'auto' }}>
            {preview
              ? preview.items.map((item) => (
                  <div key={item.id}>
                    <span className="muted">{item.old_title ?? item.id.slice(-6)}</span>
                    {' → '}
                    {item.title}
                  </div>
                ))
              : ids.map((id) => {
                  const job = jobsById.get(id)
                  return (
                    <div key={id} className="muted">
                      {job ? jobLabel(job) : id}
                    </div>
                  )
                })}
          </div>

          {error && <div className="warn">{error}</div>}

          <div className="btn-row">
            <button type="submit" className="btn btn--primary" disabled={busy || !preview}>
              {busy ? 'Renaming…' : `Rename ${ids.length} ${ids.length === 1 ? 'job' : 'jobs'}`}
            </button>
            <button type="button" className="btn" onClick={onClose} disabled={busy}>
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
