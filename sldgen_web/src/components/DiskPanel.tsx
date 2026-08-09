import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { CleanupResult } from '../api/types'
import { formatBytes } from '../lib/format'
import { useApp } from '../state/store'

/**
 * Disk breakdown (Spec 3 SS10) and cleanup (SS11).
 *
 * Every action states the exact number of jobs and bytes it will free *before*
 * confirmation, and that figure comes from the same server-side selection code
 * that performs the action (`dry_run`), so it cannot drift from what actually
 * happens.
 *
 * Friction is proportional to consequence: more than one job, or anything over
 * 1 GB, requires typing the count.
 */

const ACTIONS: { action: string; label: string; describe: string }[] = [
  { action: 'delete_failed', label: 'Delete all failed jobs', describe: 'Everything in the failed state, with its logs.' },
  {
    action: 'delete_completed_older_than',
    label: 'Delete completed jobs older than…',
    describe: 'Reached the horizon and finished more than N days ago.',
  },
  {
    action: 'prune_checkpoints',
    label: 'Prune checkpoints from completed jobs',
    describe: 'Keeps the last one, so the job can still be promoted.',
  },
  {
    action: 'prune_frames',
    label: 'Prune frames from completed jobs',
    describe: 'Only where sketch.mp4 exists, so the frames are never the last copy.',
  },
  {
    action: 'delete_orphan_uploads',
    label: 'Remove orphaned uploads',
    describe: 'Uploaded images no job references any more.',
  },
]

const CONFIRM_BYTES = 1024 ** 3

export function DiskPanel({ onClose }: { onClose: () => void }) {
  const { disk, refreshDisk, refreshJobs, toast, jobsById } = useApp()
  const [days, setDays] = useState(30)
  const [pending, setPending] = useState<{ action: string; result: CleanupResult } | null>(null)
  const [confirmText, setConfirmText] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    refreshDisk()
  }, [refreshDisk])

  const preview = async (action: string) => {
    setConfirmText('')
    try {
      const result = await api.cleanup({ action, days, dry_run: true })
      setPending({ action, result })
    } catch (error) {
      toast(error instanceof Error ? error.message : 'Could not work out what that would free')
    }
  }

  const needsTyping =
    pending !== null && (pending.result.job_count > 1 || pending.result.bytes > CONFIRM_BYTES)
  const confirmed = !needsTyping || confirmText.trim() === String(pending?.result.job_count)

  const apply = async () => {
    if (!pending || !confirmed) return
    setBusy(true)
    try {
      const result = await api.cleanup({ action: pending.action, days, dry_run: false })
      toast(`Freed ${formatBytes(result.bytes)} across ${result.job_count} items.`)
      setPending(null)
      setConfirmText('')
      refreshDisk(true)
      refreshJobs()
    } catch (error) {
      toast(error instanceof Error ? error.message : 'Cleanup failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="overlay" onClick={onClose} role="presentation">
      <div
        className="overlay__card overlay__card--wide"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-label="Disk usage and cleanup"
      >
        <div className="panel__head">
          <span className="eyebrow">Disk · {formatBytes(disk?.total_bytes ?? null)}</span>
          <button type="button" className="btn btn--small" onClick={() => refreshDisk(true)}>
            Recompute
          </button>
          <button type="button" className="btn btn--small" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="panel__body" style={{ display: 'grid', gap: 16 }}>
          <section>
            <div className="eyebrow" style={{ marginBottom: 6 }}>
              By category
            </div>
            <table className="table">
              <tbody>
                {Object.entries(disk?.by_category ?? {}).map(([name, bytes]) => (
                  <tr key={name}>
                    <th>{name}</th>
                    <td>{formatBytes(bytes)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section>
            <div className="eyebrow" style={{ marginBottom: 6 }}>
              Ten largest jobs
            </div>
            <table className="table">
              <tbody>
                {(disk?.by_job ?? []).slice(0, 10).map((entry) => (
                  <tr key={entry.job_id}>
                    <th style={{ width: 'auto' }}>
                      {entry.title ?? jobsById.get(entry.job_id)?.title ?? entry.job_id.slice(-6)}
                    </th>
                    <td>{formatBytes(entry.bytes)}</td>
                  </tr>
                ))}
                {(disk?.by_job.length ?? 0) === 0 && (
                  <tr>
                    <td className="muted">Nothing on disk yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </section>

          <section>
            <div className="eyebrow" style={{ marginBottom: 6 }}>
              Cleanup
            </div>
            <div style={{ display: 'grid', gap: 5 }}>
              {ACTIONS.map((entry) => (
                <div key={entry.action} className="btn-row">
                  <button
                    type="button"
                    className="btn btn--small"
                    onClick={() => preview(entry.action)}
                  >
                    {entry.label}
                  </button>
                  {entry.action === 'delete_completed_older_than' && (
                    <input
                      className="input"
                      style={{ width: 62 }}
                      type="number"
                      min={0}
                      value={days}
                      onChange={(event) => setDays(Number(event.target.value))}
                      aria-label="Days"
                    />
                  )}
                  <span className="note">{entry.describe}</span>
                </div>
              ))}
            </div>
            <p className="warn" style={{ marginTop: 10 }}>
              Logs are never pruned. A job's logs are deleted only when the job is.
            </p>
          </section>

          {pending && (
            <section className="panel" style={{ borderColor: 'var(--ink)' }}>
              <div className="panel__body">
                <div style={{ marginBottom: 8 }}>
                  This frees <strong>{formatBytes(pending.result.bytes)}</strong> across{' '}
                  <strong>{pending.result.job_count}</strong>{' '}
                  {pending.result.job_count === 1 ? 'item' : 'items'}.
                </div>
                {pending.result.job_count === 0 ? (
                  <div className="note">Nothing to do.</div>
                ) : (
                  <>
                    <div
                      className="mono"
                      style={{ maxHeight: 130, overflow: 'auto', marginBottom: 8 }}
                    >
                      {pending.result.items.slice(0, 40).map((item) => (
                        <div key={item.id}>
                          {item.title ?? item.id} · {formatBytes(item.bytes)}
                        </div>
                      ))}
                      {pending.result.items.length > 40 && (
                        <div className="muted">
                          …and {pending.result.items.length - 40} more
                        </div>
                      )}
                    </div>
                    <div className="btn-row">
                      {needsTyping && (
                        <>
                          <span className="note">
                            Type {pending.result.job_count} to confirm:
                          </span>
                          <input
                            className="input"
                            style={{ width: 70 }}
                            value={confirmText}
                            onChange={(event) => setConfirmText(event.target.value)}
                            aria-label="Confirmation"
                          />
                        </>
                      )}
                      <button
                        type="button"
                        className="btn btn--danger"
                        disabled={!confirmed || busy}
                        onClick={apply}
                      >
                        {busy ? 'Working…' : 'Delete'}
                      </button>
                      <button
                        type="button"
                        className="btn"
                        onClick={() => {
                          setPending(null)
                          setConfirmText('')
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  </>
                )}
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  )
}
