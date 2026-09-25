import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { CleanupResult, JobSummary } from '../api/types'
import { formatBytes, jobLabel } from '../lib/format'
import { useApp } from '../state/store'

/** States where a delete has to stop a segment first, rather than just unlink. */
const LIVE_STATES = new Set(['running', 'waiting', 'queued', 'paused'])

/**
 * "Delete these 7 jobs" — the confirmation for the rail's ticked selection.
 *
 * The count and the byte figure come from the server's own dry run of the very
 * action about to be performed (`/api/maintenance/cleanup`, `dry_run`), not
 * from adding up what the browser happens to know. That is the same rule the
 * disk panel follows, and the reason for it is that a confirmation is only
 * worth asking for if the number in it is true.
 *
 * Unlike the disk panel's sweeps, this one asks for no typed confirmation: you
 * ticked each of these jobs yourself, one at a time, and the list is in front
 * of you. What it does insist on saying is which of them are still running —
 * deleting those throws away GPU time that is happening right now, and that is
 * the mistake worth a second of friction.
 */
export function DeleteJobsDialog({
  ids,
  onClose,
}: {
  ids: string[]
  onClose: () => void
}) {
  const { jobsById, setSelection, refreshJobs, refreshDisk, toast } = useApp()
  const [preview, setPreview] = useState<CleanupResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    let cancelled = false
    api
      .deleteJobs(ids, true)
      .then((result) => {
        if (!cancelled) setPreview(result)
      })
      .catch((problem: unknown) => {
        if (!cancelled) {
          setError(problem instanceof Error ? problem.message : 'Could not work out what that frees')
        }
      })
    return () => {
      cancelled = true
    }
  }, [ids])

  const selected = ids
    .map((id) => jobsById.get(id))
    .filter((job): job is JobSummary => job !== undefined)
  const live = selected.filter((job) => LIVE_STATES.has(job.state))

  const confirm = async () => {
    setBusy(true)
    try {
      const result = await api.deleteJobs(ids, false)
      toast(
        `Deleted ${result.job_count} ${result.job_count === 1 ? 'job' : 'jobs'} · ${formatBytes(
          result.bytes,
        )} freed.`,
      )
      setSelection([])
      refreshJobs()
      refreshDisk(true)
      onClose()
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : 'Delete failed')
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
        aria-label="Delete selected jobs"
      >
        <div className="panel__head">
          <span className="eyebrow">Delete</span>
          <button type="button" className="btn btn--small" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="panel__body" style={{ display: 'grid', gap: 10 }}>
          <p style={{ margin: 0 }}>
            Are you sure you want to delete{' '}
            <strong>
              {ids.length === 1 ? 'this job' : `these ${ids.length} jobs`}
            </strong>
            ? This operation cannot be undone.
          </p>

          <div className="note">
            Everything each job produced goes with it: frames, SVGs, checkpoints and logs.
            {preview && ` Frees ${formatBytes(preview.bytes)}.`}
          </div>

          {live.length > 0 && (
            <p className="warn" style={{ margin: 0 }}>
              <strong>
                {live.length === 1
                  ? 'One of these is still in flight.'
                  : `${live.length} of these are still in flight.`}
              </strong>{' '}
              A running job is stopped at its next checkpoint and then removed, so the GPU time it
              has spent since is lost with it.
            </p>
          )}

          <div className="mono" style={{ maxHeight: 220, overflow: 'auto' }}>
            {(preview?.items ?? selected.map((job) => ({ id: job.id, title: job.title, bytes: 0 })))
              .slice(0, 60)
              .map((item) => {
                const job = jobsById.get(item.id)
                return (
                  <div key={item.id}>
                    {job ? jobLabel(job) : (item.title ?? item.id)}
                    {job ? ` · ${job.state} · ${job.current_epoch}/${job.num_iter}` : ''}
                    {preview ? ` · ${formatBytes(item.bytes)}` : ''}
                  </div>
                )
              })}
            {(preview?.items.length ?? ids.length) > 60 && (
              <div className="muted">…and {(preview?.items.length ?? ids.length) - 60} more</div>
            )}
          </div>

          {error && <div className="warn">{error}</div>}

          <div className="btn-row">
            <button
              type="button"
              className="btn btn--danger"
              disabled={busy || ids.length === 0}
              onClick={confirm}
            >
              {busy
                ? 'Deleting…'
                : `Delete ${ids.length} ${ids.length === 1 ? 'job' : 'jobs'}`}
            </button>
            <button type="button" className="btn" onClick={onClose} disabled={busy}>
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
