import { useMemo, useState } from 'react'
import type { JobState, JobSummary } from '../api/types'
import { JOB_STATES } from '../api/types'
import { ERROR_COPY, formatAgo, formatDuration, jobLabel } from '../lib/format'
import { navigate } from '../router'
import { useApp } from '../state/store'
import { JobThumb } from './JobThumb'
import { SelectionActions } from './SelectionActions'
import { JobStatus } from './JobStatus'
import { StarToggle } from './StarToggle'
import { queueFirst, queueLabel, queuePositions } from '../lib/queue'

export type RailSort = 'newest' | 'queue' | 'longest'

const SORT_LABEL: Record<RailSort, string> = {
  newest: 'newest',
  queue: 'queue order',
  longest: 'longest running',
}
const NEXT_SORT: Record<RailSort, RailSort> = { newest: 'queue', queue: 'longest', longest: 'newest' }

/** One line of monospace data per row -- the rail is for scanning, not reading. */
function rowMeta(job: JobSummary, position?: number): string {
  if (job.state === 'failed') return job.error_class ?? 'failed'
  if (job.state === 'queued') {
    const where = queueLabel(position)
    return `queued ${formatAgo(job.created_at)}${where ? ` · ${where}` : ''}`
  }
  if (job.state === 'complete') return `${job.num_iter} done`
  return `${job.current_epoch}/${job.num_iter}`
}

export function filterJobs(
  jobs: JobSummary[],
  {
    states,
    starredOnly = false,
    text,
    sort,
  }: { states: Set<JobState>; starredOnly?: boolean; text: string; sort: RailSort },
): JobSummary[] {
  const needle = text.trim().toLowerCase()
  const filtered = jobs.filter((job) => {
    if (starredOnly && !job.starred) return false
    if (states.size > 0 && !states.has(job.state)) return false
    if (!needle) return true
    return (
      (job.title ?? '').toLowerCase().includes(needle) ||
      (job.resolved_caption ?? '').toLowerCase().includes(needle) ||
      job.id.toLowerCase().includes(needle)
    )
  })
  if (sort === 'queue') return queueFirst(filtered)
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

export function JobRail({
  selectedId,
  focusedId,
}: {
  selectedId: string | null
  focusedId: string | null
}) {
  const { jobs, selection, toggleSelected, stateFilter, setStateFilter, starredOnly, setStarredOnly } =
    useApp()
  const states = stateFilter
  const setStates = setStateFilter
  const [text, setText] = useState('')
  const [sort, setSort] = useState<RailSort>('newest')

  // Positions over the whole queue, not just the filtered rows.
  const positions = useMemo(() => queuePositions(jobs), [jobs])
  const visible = useMemo(
    () => filterJobs(jobs, { states, starredOnly, text, sort }),
    [jobs, states, starredOnly, text, sort],
  )

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
          <button
            type="button"
            className="chip chip--star"
            aria-pressed={starredOnly}
            aria-label="Only favourites"
            title="Only favourites"
            onClick={() => setStarredOnly(!starredOnly)}
          >
            ★
          </button>
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
            onClick={() => setSort(NEXT_SORT[sort])}
            title="Sort order: newest first, the queue in the order the worker takes it, or longest running"
          >
            {SORT_LABEL[sort]}
          </button>
        </div>
        <SelectionActions compact />
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

        {visible.map((job) => (
          <RailRow
            key={job.id}
            job={job}
            selected={job.id === selectedId}
            checked={selection.includes(job.id)}
            focused={job.id === focusedId}
            position={positions.get(job.id)}
            onToggle={toggleSelected}
          />
        ))}
      </div>
    </div>
  )
}

function RailRow({
  job,
  selected,
  checked,
  focused,
  position,
  onToggle,
}: {
  job: JobSummary
  selected: boolean
  checked: boolean
  focused: boolean
  position?: number
  onToggle: (id: string, additive: boolean) => void
}) {
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
      <JobThumb
        job={job}
        className="row__thumb"
        emptyClassName="row__thumb row__thumb--empty"
      />
      <span className="row__body">
        <span className="row__title">{jobLabel(job)}</span>
        <span className="row__meta">
          {rowMeta(job, position)}
          {job.state === 'running' && job.started_at
            ? ` · ${formatDuration((Date.now() - Date.parse(job.started_at)) / 1000)}`
            : ''}
        </span>
      </span>
      <StarToggle job={job} />
      <JobStatus job={job} size={20} position={position} />
    </div>
  )
}
