import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { fileUrl } from '../api/client'
import type { JobDetail } from '../api/types'
import { formatDuration } from '../lib/format'
import { promoteSteps } from '../lib/promote'
import { MAX_PRIORITY, queueLabel, queuePositions, runNextPriority } from '../lib/queue'
import { useApp } from '../state/store'

/**
 * Actions, grouped by consequence, with destructive ones below a rule
 * (Spec 3 SS6.3).
 *
 * `Promote` is the primary action on this page and looks like it: the whole
 * workflow is four short runs, look at them together, promote one.
 */
export function ActionsPanel({
  job,
  favorites,
  onRunAgain,
  onChanged,
}: {
  job: JobDetail
  /** Starred epochs, passed in so the button appears the moment one is set. */
  favorites: number[]
  onRunAgain: () => void
  onChanged: () => void
}) {
  const { toast, jobs, jobsById, toggleStar, setPriority } = useApp()
  const [promoteTo, setPromoteTo] = useState(job.num_iter)
  const [title, setTitle] = useState(job.title ?? '')
  const [withCheckpoints, setWithCheckpoints] = useState(false)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    setPromoteTo(job.num_iter)
  }, [job.id, job.num_iter])

  useEffect(() => {
    setTitle(job.title ?? '')
  }, [job.id, job.title])

  const cleanTitle = title.trim().replace(/\s+/g, ' ')
  const titleChanged = cleanTitle !== (job.title ?? '')
  const titleTaken =
    cleanTitle !== '' && jobs.some((other) => other.id !== job.id && other.title === cleanTitle)

  const run = async (label: string, action: () => Promise<unknown>) => {
    setBusy(true)
    try {
      await action()
      toast(label)
      onChanged()
    } catch (error) {
      toast(error instanceof Error ? error.message : `${label} failed`)
    } finally {
      setBusy(false)
    }
  }

  /**
   * "You have already seen enough": stop now and stop asking for more.
   *
   * Spec 3 SS6.3 describes this as leaving the job at `waiting`, but Spec 2's
   * state machine reaches `waiting` only by *reaching* a budget, and `paused` is
   * precisely the state for "the user intervened". So cancel pauses and then
   * pulls the budget back to the epoch actually reached — which makes `Resume`
   * refuse with "promote it instead", exactly the way a job that finished its
   * budget behaves. Same meaning, no new state.
   */
  const cancel = () =>
    run('Cancelled — the run stopped at the epoch it reached.', async () => {
      await api.pause(job.id)
      for (let attempt = 0; attempt < 120; attempt += 1) {
        const current = await api.getJob(job.id)
        if (current.state !== 'running') {
          const reached = Math.max(1, current.current_epoch)
          if (reached < current.target_epoch) {
            await api.patchJob(job.id, { target_epoch: reached })
          }
          return
        }
        await new Promise((resolve) => setTimeout(resolve, 1000))
      }
      throw new Error('The segment did not stop within two minutes.')
    })

  // From the live list rather than `job`, so the button flips the moment it is
  // clicked -- the detail is only refetched when the job changes state.
  const starred = jobsById.get(job.id)?.starred ?? job.starred
  // Priority, likewise from the live list, and where that puts the job.
  const live = jobsById.get(job.id)
  const priority = live?.priority ?? job.priority
  const position = queuePositions(jobs).get(job.id)
  const nextPriority = live ? runNextPriority(live, jobs) : priority
  // Priority orders the queue; it matters for a job that is queued now or may
  // be again (promote/resume put it back in the queue with its priority).
  const prioritisable = !['running', 'complete', 'deleting'].includes(job.state)
  const running = job.state === 'running'
  const waiting = job.state === 'waiting'
  const resumable = job.state === 'paused' && job.current_epoch < job.target_epoch
  const promotable = ['waiting', 'paused', 'complete', 'failed', 'queued'].includes(job.state)
  const rate = job.state_json?.iters_per_sec ?? null
  const estimate =
    rate && promoteTo > job.current_epoch
      ? formatDuration((promoteTo - job.current_epoch) / rate)
      : null

  return (
    <div className="panel">
      <div className="panel__head">
        <span className="eyebrow">Actions</span>
        <span className={`mono state-${job.state}`}>{job.state}</span>
      </div>
      <div className="panel__body" style={{ display: 'grid', gap: 10 }}>
        {running && (
          <div className="btn-row">
            <button type="button" className="btn" disabled={busy} onClick={() => run('Pausing — it will checkpoint first.', () => api.pause(job.id))}>
              Pause
            </button>
            <button type="button" className="btn" disabled={busy} onClick={cancel}>
              Cancel
            </button>
            <span className="note">
              Pause keeps this run's budget. Cancel gives it up at the current epoch.
            </span>
          </div>
        )}

        {resumable && (
          <div className="btn-row">
            <button
              type="button"
              className="btn btn--primary"
              disabled={busy}
              onClick={() => run('Resumed.', () => api.resume(job.id))}
            >
              Resume to {job.target_epoch}
            </button>
          </div>
        )}

        {promotable && (
          <div>
            <div className="eyebrow" style={{ marginBottom: 5 }}>
              {waiting ? 'Reached its budget — promote to continue' : 'Promote to continue'}
            </div>
            <div className="note" style={{ marginBottom: 5 }}>
              Same job: it resumes from the last checkpoint and adds a segment, so the epoch count
              keeps climbing rather than starting a second job.
            </div>
            <div className="btn-row">
              <button
                type="button"
                className="btn btn--primary"
                disabled={busy || promoteTo <= job.current_epoch || promoteTo > job.num_iter}
                onClick={() =>
                  run(`Promoted to ${promoteTo}.`, () => api.promote(job.id, promoteTo))
                }
              >
                Promote to
              </button>
              <input
                className="input"
                style={{ width: 84 }}
                type="number"
                min={job.current_epoch + 1}
                max={job.num_iter}
                value={promoteTo}
                onChange={(event) => setPromoteTo(Number(event.target.value))}
                aria-label="Promote to iteration"
              />
              {promoteSteps(job.num_iter, job.current_epoch).map((step) => (
                <button
                  key={step}
                  type="button"
                  className="btn btn--small"
                  onClick={() => setPromoteTo(Math.min(job.num_iter, job.current_epoch + step))}
                >
                  +{step}
                </button>
              ))}
              <button
                type="button"
                className="btn btn--small"
                onClick={() => setPromoteTo(job.num_iter)}
              >
                to {job.num_iter}
              </button>
            </div>
            {estimate && <div className="note" style={{ marginTop: 4 }}>About {estimate} of GPU time.</div>}
            {promoteTo > job.num_iter && (
              <div className="note">
                The horizon is {job.num_iter}. It defines the sparse-loss ramp for every iteration,
                so going past it means a new job, not a promotion.
              </div>
            )}
          </div>
        )}

        {prioritisable && (
          <div>
            <div className="eyebrow" style={{ marginBottom: 5 }}>
              Queue priority
            </div>
            <div className="btn-row">
              <button
                type="button"
                className="btn btn--small"
                aria-label="Lower priority"
                disabled={priority <= 0}
                onClick={() => setPriority([job.id], priority - 1)}
              >
                −
              </button>
              <span className="priority-value mono" aria-label="Priority">
                {priority}
              </span>
              <button
                type="button"
                className="btn btn--small"
                aria-label="Raise priority"
                disabled={priority >= MAX_PRIORITY}
                onClick={() => setPriority([job.id], priority + 1)}
              >
                +
              </button>
              {job.state === 'queued' && position !== 1 && (
                <button
                  type="button"
                  className="btn btn--small"
                  disabled={nextPriority === priority}
                  title={
                    nextPriority === priority
                      ? `Jobs at the top priority (${MAX_PRIORITY}) that were submitted earlier stay ahead.`
                      : `Sets priority ${nextPriority}, one above the job at the head of the queue.`
                  }
                  onClick={() => setPriority([job.id], nextPriority)}
                >
                  Run next
                </button>
              )}
              {priority > 0 && (
                <button type="button" className="btn btn--small" onClick={() => setPriority([job.id], 0)}>
                  Reset
                </button>
              )}
              <span className="note">
                {job.state === 'queued' && position
                  ? `${queueLabel(position)[0].toUpperCase()}${queueLabel(position).slice(1)}. `
                  : ''}
                Higher runs sooner; equal priorities go in submission order. A running job is never
                interrupted.
              </span>
            </div>
          </div>
        )}

        <div className="btn-row">
          <button
            type="button"
            className={`btn${starred ? ' btn--star-on' : ''}`}
            aria-pressed={starred}
            onClick={() => toggleStar(job.id)}
          >
            {starred ? '★ Unmark favourite' : '☆ Mark as favourite'}
          </button>
          <span className="note">Filter the rail to favourites with its ★ chip.</span>
        </div>

        <div className="btn-row">
          <button type="button" className="btn" onClick={onRunAgain}>
            Run again with changes…
          </button>
          <span className="note">
            Forks a new job — a changed parameter is a different drawing, so it cannot continue
            this one.
          </span>
          {job.state === 'failed' && (
            <button
              type="button"
              className="btn"
              disabled={busy}
              onClick={() => run('Retrying.', () => api.retry(job.id))}
            >
              Retry
            </button>
          )}
        </div>

        <hr className="rule" />

        <div>
          <div className="eyebrow" style={{ marginBottom: 5 }}>
            Downloads
          </div>
          <div className="btn-row">
            {job.artifacts.some((artifact) => artifact.name === 'final_sld.svg') && (
              <a className="btn btn--small" href={fileUrl(job.id, 'target/run/final_sld.svg')} download>
                final SVG
              </a>
            )}
            {job.artifacts.some((artifact) => artifact.name === 'final_sld.png') && (
              <a className="btn btn--small" href={fileUrl(job.id, 'target/run/final_sld.png')} download>
                final PNG
              </a>
            )}
            {job.artifacts.some((artifact) => artifact.name === 'image_loss_log.csv') && (
              <a
                className="btn btn--small"
                href={fileUrl(job.id, 'target/run/image_loss_log.csv')}
                download
              >
                fidelity CSV
              </a>
            )}
            {job.artifacts.some((artifact) => artifact.name === 'sketch.mp4') && (
              <a className="btn btn--small" href={fileUrl(job.id, 'target/run/sketch.mp4')} download>
                mp4
              </a>
            )}
            {favorites.length > 0 && (
              <a
                className="btn btn--small"
                href={`/api/jobs/${job.id}/favorites.zip`}
                download
                title={`Epoch ${favorites.join(', ')}`}
              >
                ★ {favorites.length} starred SVG{favorites.length === 1 ? '' : 's'} (.zip)
              </a>
            )}
            <a
              className="btn btn--small"
              href={`/api/jobs/${job.id}/download.zip${withCheckpoints ? '?checkpoints=true' : ''}`}
              download
            >
              Download everything (.zip)
            </a>
            <label className="note" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <input
                type="checkbox"
                checked={withCheckpoints}
                onChange={(event) => setWithCheckpoints(event.target.checked)}
              />
              include checkpoints
            </label>
          </div>
          {withCheckpoints && (
            <div className="note" style={{ marginTop: 4 }}>
              Checkpoints are large and useless outside this service.
            </div>
          )}
        </div>

        <hr className="rule" />

        <form
          className="btn-row"
          onSubmit={(event) => {
            event.preventDefault()
            if (!titleChanged || busy) return
            run(cleanTitle ? `Renamed to ${cleanTitle}.` : 'Title cleared.', () =>
              api.patchJob(job.id, { title: cleanTitle }),
            )
          }}
        >
          <input
            className="input"
            style={{ flex: 1, minWidth: 0 }}
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            placeholder={job.id.slice(-6)}
            aria-label="Job title"
          />
          <button type="submit" className="btn" disabled={busy || !titleChanged}>
            Rename
          </button>
        </form>
        {titleChanged && titleTaken && (
          <div className="note warn" style={{ marginTop: 4 }}>
            Another job already has this name.
          </div>
        )}

        <div className="btn-row" style={{ marginTop: 10 }}>
          <button
            type="button"
            className="btn btn--danger"
            disabled={busy}
            onClick={() => {
              if (!window.confirm(`Delete this job and everything it produced?`)) return
              run('Deleting.', () => api.remove(job.id))
            }}
          >
            Delete
          </button>
          <span className="note">Its logs go with it. Nothing else depends on it.</span>
        </div>
      </div>
    </div>
  )
}
