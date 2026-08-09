import { useMemo, useState } from 'react'
import type { JobState, JobSummary } from '../api/types'
import { JOB_STATES } from '../api/types'
import { ERROR_COPY, formatAgo, formatDuration, jobLabel } from '../lib/format'
import { combinedRing } from '../lib/ring'
import { navigate } from '../router'
import { useApp } from '../state/store'
import { Ring } from './Ring'

export type RailSort = 'newest' | 'longest'

/** One line of monospace data per row -- the rail is for scanning, not reading. */
function rowMeta(job: JobSummary): string {
  if (job.state === 'failed') return job.error_class ?? 'failed'
  if (job.state === 'queued') return `queued ${formatAgo(job.created_at)}`
  if (job.state === 'complete') return `${job.num_iter} done`
  return `${job.current_epoch}/${job.num_iter}`
}

export function filterJobs(
  jobs: JobSummary[],
  { states, text, sort }: { states: Set<JobState>; text: string; sort: RailSort },
): JobSummary[] {
  const needle = text.trim().toLowerCase()
  const filtered = jobs.filter((job) => {
    if (states.size > 0 && !states.has(job.state)) return false
    if (!needle) return true
    return (
      (job.title ?? '').toLowerCase().includes(needle) ||
      (job.resolved_caption ?? '').toLowerCase().includes(needle) ||
      job.id.toLowerCase().includes(needle)
    )
  })
  if (sort === 'longest') {
    // "Longest running" means elapsed since the job first started, which is the
    // question being asked when you sort by it: what has been on the card
    // longest, not what was submitted first.
    return [...filtered].sort((a, b) => {
      const started = (job: JobSummary) =>
        job.started_at ? Date.parse(job.started_at) : Number.POSITIVE_INFINITY
      return started(a) - started(b)
    })
  }
  return filtered
}

/** Group by batch, keeping ungrouped jobs in place (Spec 3 SS6.6). */
export function groupByBatch(jobs: JobSummary[]): { batchId: string | null; jobs: JobSummary[] }[] {
  const groups: { batchId: string | null; jobs: JobSummary[] }[] = []
  const byBatch = new Map<string, { batchId: string | null; jobs: JobSummary[] }>()
  for (const job of jobs) {
    if (!job.batch_id) {
      groups.push({ batchId: null, jobs: [job] })
      continue
    }
    const existing = byBatch.get(job.batch_id)
    if (existing) {
      existing.jobs.push(job)
    } else {
      const group = { batchId: job.batch_id, jobs: [job] }
      byBatch.set(job.batch_id, group)
      groups.push(group)
    }
  }
  // A "batch" of one is just a job; a heading around it is noise.
  return groups.map((group) =>
    group.jobs.length === 1 ? { batchId: null, jobs: group.jobs } : group,
  )
}

export function JobRail({
  selectedId,
  focusedId,
}: {
  selectedId: string | null
  focusedId: string | null
}) {
  const { jobs, selection, toggleSelected, setSelection, stateFilter, setStateFilter } = useApp()
  const states = stateFilter
  const setStates = setStateFilter
  const [text, setText] = useState('')
  const [sort, setSort] = useState<RailSort>('newest')

  const visible = useMemo(() => filterJobs(jobs, { states, text, sort }), [jobs, states, text, sort])
  const groups = useMemo(() => groupByBatch(visible), [visible])

  const toggleState = (state: JobState) => {
    const next = new Set(states)
    if (next.has(state)) next.delete(state)
    else next.add(state)
    setStates(next)
  }

  return (
    <div className="rail">
      <div className="rail__controls">
        <input
          className="rail__filter"
          id="rail-filter"
          placeholder="Filter by title or caption"
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <div className="chips">
          {JOB_STATES.filter((state) => state !== 'deleting').map((state) => (
            <button
              key={state}
              type="button"
              className="chip"
              aria-pressed={states.has(state)}
              onClick={() => toggleState(state)}
            >
              <span className={`chip__dot state-${state}`} />
              {state}
            </button>
          ))}
          <button
            type="button"
            className="chip"
            onClick={() => setSort(sort === 'newest' ? 'longest' : 'newest')}
            title="Sort order"
          >
            {sort === 'newest' ? 'newest' : 'longest running'}
          </button>
        </div>
        {selection.length > 0 && (
          <div className="btn-row">
            <button
              type="button"
              className="btn btn--small btn--primary"
              disabled={selection.length < 2}
              onClick={() => navigate({ name: 'compare', ids: selection })}
            >
              Compare {selection.length}
            </button>
            <button type="button" className="btn btn--small" onClick={() => setSelection([])}>
              Clear
            </button>
          </div>
        )}
      </div>

      <div className="rail__list">
        {visible.length === 0 && (
          <div className="empty">
            <div>{jobs.length === 0 ? 'No jobs yet.' : 'Nothing matches that filter.'}</div>
            {jobs.length === 0 && (
              <button
                type="button"
                className="btn btn--primary"
                onClick={() => navigate({ name: 'new' })}
              >
                Prepare an image to get started
              </button>
            )}
          </div>
        )}

        {groups.map((group, index) =>
          group.batchId ? (
            <details key={group.batchId} className="rail__batch" open>
              <summary
                onClick={(event) => {
                  // Clicking the heading selects the whole batch for compare,
                  // which is how a batch is consumed (SS6.6); the disclosure
                  // triangle still opens and closes it.
                  if ((event.target as HTMLElement).closest('.rail__batch-select')) return
                }}
              >
                <Ring size={18} {...combinedRing(group.jobs)} />
                <span className="rail__batch-label">
                  batch · {group.jobs.length} variants
                </span>
                <button
                  type="button"
                  className="btn btn--small rail__batch-select"
                  onClick={(event) => {
                    event.preventDefault()
                    event.stopPropagation()
                    setSelection(group.jobs.map((job) => job.id))
                    navigate({ name: 'compare', ids: group.jobs.map((job) => job.id) })
                  }}
                >
                  compare
                </button>
              </summary>
              {group.jobs.map((job) => (
                <RailRow
                  key={job.id}
                  job={job}
                  selected={job.id === selectedId}
                  checked={selection.includes(job.id)}
                  focused={job.id === focusedId}
                  onToggle={toggleSelected}
                />
              ))}
            </details>
          ) : (
            <RailRow
              key={group.jobs[0]?.id ?? index}
              job={group.jobs[0]}
              selected={group.jobs[0].id === selectedId}
              checked={selection.includes(group.jobs[0].id)}
              focused={group.jobs[0].id === focusedId}
              onToggle={toggleSelected}
            />
          ),
        )}
      </div>
    </div>
  )
}

function RailRow({
  job,
  selected,
  checked,
  focused,
  onToggle,
}: {
  job: JobSummary
  selected: boolean
  checked: boolean
  focused: boolean
  onToggle: (id: string, additive: boolean) => void
}) {
  const [thumbBroken, setThumbBroken] = useState(false)
  const title =
    job.state === 'failed' && job.error_class
      ? ERROR_COPY[job.error_class]?.headline
      : job.resolved_caption ?? undefined

  return (
    <div
      className={`row${focused ? ' row--focused' : ''}`}
      role="button"
      tabIndex={-1}
      aria-current={selected}
      title={title}
      onClick={(event) => {
        if (event.shiftKey || event.metaKey || event.ctrlKey) {
          onToggle(job.id, true)
          return
        }
        navigate({ name: 'job', id: job.id })
      }}
    >
      <input
        type="checkbox"
        className="row__check"
        checked={checked}
        aria-label={`Select ${jobLabel(job)}`}
        onClick={(event) => event.stopPropagation()}
        onChange={() => onToggle(job.id, true)}
      />
      {thumbBroken ? (
        <span className="row__thumb row__thumb--empty" />
      ) : (
        <img
          className="row__thumb"
          src={`${job.preview_url}?v=${job.current_epoch}`}
          alt=""
          loading="lazy"
          onError={() => setThumbBroken(true)}
        />
      )}
      <span className="row__body">
        <span className="row__title">{jobLabel(job)}</span>
        <span className="row__meta">
          {rowMeta(job)}
          {job.state === 'running' && job.started_at
            ? ` · ${formatDuration((Date.now() - Date.parse(job.started_at)) / 1000)}`
            : ''}
        </span>
      </span>
      <Ring
        size={20}
        state={job.state}
        currentEpoch={job.current_epoch}
        targetEpoch={job.target_epoch}
        numIter={job.num_iter}
      />
    </div>
  )
}
