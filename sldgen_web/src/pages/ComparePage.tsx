import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import type { FramesResponse, JobDetail } from '../api/types'
import { Ring } from '../components/Ring'
import { formatParamValue } from '../lib/params'
import { distinguishingParams, paramLabel } from '../lib/paramdiff'
import { jobLabel } from '../lib/format'
import { navigate } from '../router'
import { useApp } from '../state/store'

/**
 * Compare (Spec 3 SS7).
 *
 * Where the core workflow completes: it should be possible to go from four
 * finished previews to one promoted job in two clicks, without visiting a detail
 * page. So every cell carries Promote directly.
 *
 * Cells show only the parameters that differ *relative to the others*, which for
 * a seed batch means the grid is captioned purely by seed number.
 */
export function ComparePage({ ids }: { ids: string[] }) {
  const { selection, toast, jobs } = useApp()
  const active = ids.length > 0 ? ids : selection
  const [details, setDetails] = useState<JobDetail[]>([])
  const [frames, setFrames] = useState<Record<string, FramesResponse>>({})
  const [sharedEpoch, setSharedEpoch] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (active.length === 0) {
      setDetails([])
      return
    }
    let cancelled = false
    Promise.all(active.map((id) => api.getJob(id).catch(() => null))).then((results) => {
      if (!cancelled) setDetails(results.filter((job): job is JobDetail => job !== null))
    })
    Promise.all(
      active.map((id) => api.frames(id).then((value) => [id, value] as const).catch(() => null)),
    ).then((results) => {
      if (cancelled) return
      setFrames(Object.fromEntries(results.filter(Boolean) as [string, FramesResponse][]))
    })
    return () => {
      cancelled = true
    }
    // `jobs` is in the deps so a state change (a run finishing) refreshes the grid.
  }, [active.join(','), jobs])

  const differing = useMemo(
    () => distinguishingParams(details.map((job) => job.params)),
    [details],
  )

  /**
   * The shared scrubber is offered only where the cells overlap: moving all four
   * to epoch 300 is more informative than comparing them at their respective
   * finishes, but only if all four have an epoch 300.
   */
  const overlap = useMemo(() => {
    const lists = details.map((job) => frames[job.id]?.frames.map((frame) => frame.epoch) ?? [])
    if (lists.length < 2 || lists.some((list) => list.length === 0)) return []
    const [first, ...rest] = lists
    return first.filter((epoch) => rest.every((list) => list.includes(epoch)))
  }, [details, frames])

  useEffect(() => {
    if (overlap.length > 0 && (sharedEpoch === null || !overlap.includes(sharedEpoch))) {
      setSharedEpoch(overlap[overlap.length - 1])
    }
  }, [overlap, sharedEpoch])

  if (active.length === 0) {
    return (
      <div className="empty">
        <strong>Nothing selected.</strong>
        <span className="note">
          Tick two or more jobs in the rail — or a batch heading — to compare them here.
        </span>
      </div>
    )
  }

  const promote = async (job: JobDetail, target: number) => {
    setBusy(true)
    try {
      await api.promote(job.id, target)
      toast(`Promoted ${jobLabel(job)} to ${target}.`)
      setDetails((current) =>
        current.map((entry) =>
          entry.id === job.id ? { ...entry, state: 'queued', target_epoch: target } : entry,
        ),
      )
    } catch (error) {
      toast(error instanceof Error ? error.message : 'Could not promote')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="compare">
      <div className="compare__toolbar">
        <span className="eyebrow">Comparing {details.length}</span>
        {overlap.length > 0 && sharedEpoch !== null && (
          <>
            <span className="mono">epoch {sharedEpoch}</span>
            <input
              type="range"
              style={{ flex: 1, maxWidth: 420, accentColor: 'var(--ink)' }}
              min={0}
              max={overlap.length - 1}
              value={Math.max(0, overlap.indexOf(sharedEpoch))}
              aria-label="Shared epoch"
              onChange={(event) => setSharedEpoch(overlap[Number(event.target.value)])}
            />
            <span className="note">All cells at the same epoch</span>
          </>
        )}
        {overlap.length === 0 && details.length > 1 && (
          <span className="note">
            No epoch these runs share yet — each cell shows its own latest frame.
          </span>
        )}
      </div>

      {details.map((job) => {
        const strip = frames[job.id]
        const atShared =
          sharedEpoch !== null
            ? strip?.frames.find((frame) => frame.epoch === sharedEpoch) ?? null
            : null
        const image = atShared?.png_url ?? `${job.preview_url}?v=${job.current_epoch}`
        return (
          <div className="cell" key={job.id}>
            <div className="cell__art">
              <img src={image} alt={jobLabel(job)} />
            </div>
            <div className="cell__body">
              <div className="cell__head">
                <Ring
                  size={18}
                  state={job.state}
                  currentEpoch={job.current_epoch}
                  targetEpoch={job.target_epoch}
                  numIter={job.num_iter}
                />
                <strong>{jobLabel(job)}</strong>
                <button
                  type="button"
                  className="btn btn--small btn--ghost"
                  onClick={() => navigate({ name: 'job', id: job.id })}
                >
                  open
                </button>
              </div>

              <div className="deltas">
                {differing.length === 0 ? (
                  <span className="muted">identical parameters</span>
                ) : (
                  differing.map((name) => (
                    <div key={name}>
                      <span>{paramLabel(name)}</span>
                      <span>{formatParamValue(job.params[name] ?? null)}</span>
                    </div>
                  ))
                )}
              </div>

              <div className="mono muted">
                {job.current_epoch}/{job.num_iter} · budget {job.target_epoch}
              </div>

              <div className="btn-row">
                <button
                  type="button"
                  className="btn btn--small btn--primary"
                  disabled={busy || job.target_epoch >= job.num_iter && job.state === 'complete'}
                  onClick={() => promote(job, job.num_iter)}
                >
                  Promote to {job.num_iter}
                </button>
                <button
                  type="button"
                  className="btn btn--small"
                  disabled={busy || job.current_epoch + 500 > job.num_iter}
                  onClick={() => promote(job, Math.min(job.num_iter, job.current_epoch + 500))}
                >
                  +500
                </button>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
