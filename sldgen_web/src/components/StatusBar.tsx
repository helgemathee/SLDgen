import { useState } from 'react'
import { estimateRemaining, formatBytes, formatDuration, formatRate, jobLabel } from '../lib/format'
import { navigate } from '../router'
import { useApp } from '../state/store'
import { DiskPanel } from './DiskPanel'

/**
 * Always visible, one line, monospace throughout (Spec 3 SS10).
 *
 * The worker liveness dot earns its place: if the worker unit is down, this is
 * the *only* place in the application that would show it, so the whole bar turns
 * red and says so rather than leaving you to wonder why nothing starts.
 */
export function StatusBar() {
  const { jobs, health, queueDepth, disk, diskDelta, connection, setStateFilter } = useApp()
  const [diskOpen, setDiskOpen] = useState(false)

  const running = jobs.find((job) => job.state === 'running')
  const workerDown = health !== null && !health.worker_alive
  const itersPerSec = null // per-job rate arrives on the detail page's own stream

  return (
    <>
      <div className={`statusbar${workerDown ? ' statusbar--down' : ''}`}>
        {workerDown ? (
          <span className="statusbar__cell">
            ● Worker not running — nothing will start.{' '}
            <a href="/api/logs/worker" target="_blank" rel="noreferrer">
              worker journal
            </a>
          </span>
        ) : running ? (
          <button
            type="button"
            className="statusbar__cell statusbar__cell--button"
            style={{ flex: 1, minWidth: 0, justifyContent: 'flex-start' }}
            onClick={() => navigate({ name: 'job', id: running.id })}
          >
            <span className="state-running">●</span> {jobLabel(running)} ·{' '}
            {running.current_epoch}/{running.num_iter} · {formatRate(itersPerSec)}
            {(() => {
              const left = estimateRemaining(
                running.current_epoch,
                running.target_epoch,
                itersPerSec,
              )
              return left === null ? '' : ` · ~${formatDuration(left)} left`
            })()}
          </button>
        ) : (
          <span className="statusbar__cell muted">idle</span>
        )}

        <button
          type="button"
          className="statusbar__cell statusbar__cell--button"
          title="Show only queued jobs"
          onClick={() => setStateFilter(new Set(['queued']))}
        >
          queue {queueDepth}
        </button>

        <span className="statusbar__cell">
          GPU{' '}
          {health?.gpu_free_mb != null
            ? `${(health.gpu_free_mb / 1024).toFixed(1)} GB free`
            : '—'}
        </span>

        <button
          type="button"
          className="statusbar__cell statusbar__cell--button"
          onClick={() => setDiskOpen(true)}
        >
          disk {formatBytes(disk?.total_bytes ?? null)}
          {diskDelta > 0 ? ` (+${formatBytes(diskDelta)})` : ''}
        </button>

        <span className="statusbar__cell" title={`connection: ${connection}`}>
          {connection === 'live' && 'live'}
          {connection === 'polling' && 'polling'}
          {connection === 'connecting' && 'connecting…'}
          {connection === 'reconnecting' && 'reconnecting…'}
        </span>

        <span className="statusbar__cell" style={{ borderRight: 'none', paddingRight: 0 }}>
          worker <span className={workerDown ? '' : 'state-complete'}>●</span>
        </span>
      </div>
      {diskOpen && <DiskPanel onClose={() => setDiskOpen(false)} />}
    </>
  )
}
