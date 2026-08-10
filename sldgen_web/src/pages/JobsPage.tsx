import { JobThumb } from '../components/JobThumb'
import { Ring } from '../components/Ring'
import { formatAgo, jobLabel } from '../lib/format'
import { navigate } from '../router'
import { useApp } from '../state/store'

/**
 * The overview grid.
 *
 * The rail is the authoritative list; this is the same set at a size where the
 * artwork is actually judgeable, which is what you want when four candidates
 * have just finished and the rail's 44px thumbnails are too small to choose by.
 */
export function JobsPage() {
  const { jobs, selection, toggleSelected } = useApp()

  if (jobs.length === 0) {
    return (
      <div className="empty">
        <strong>No jobs yet.</strong>
        <span className="note">Prepare an image to get started.</span>
        <button type="button" className="btn btn--primary" onClick={() => navigate({ name: 'new' })}>
          New job
        </button>
      </div>
    )
  }

  return (
    <div className="compare">
      <div className="compare__toolbar">
        <span className="eyebrow">{jobs.length} jobs</span>
        <span className="note">
          Tick two or more to compare them. Shift-click a rail row does the same.
        </span>
      </div>
      {jobs.map((job) => (
        <div className="cell" key={job.id}>
          <div
            className="cell__art"
            role="button"
            tabIndex={0}
            onClick={() => navigate({ name: 'job', id: job.id })}
            onKeyDown={(event) => {
              if (event.key === 'Enter') navigate({ name: 'job', id: job.id })
            }}
          >
            <JobThumb job={job} alt={jobLabel(job)} />
          </div>
          <div className="cell__body">
            <div className="cell__head">
              <input
                type="checkbox"
                checked={selection.includes(job.id)}
                aria-label={`Select ${jobLabel(job)}`}
                onChange={() => toggleSelected(job.id, true)}
              />
              <Ring
                size={18}
                state={job.state}
                currentEpoch={job.current_epoch}
                targetEpoch={job.target_epoch}
                numIter={job.num_iter}
              />
              <strong>{jobLabel(job)}</strong>
            </div>
            <div className="mono muted">
              {job.current_epoch}/{job.num_iter} · {job.state} · {formatAgo(job.created_at)}
            </div>
            {job.resolved_caption && <div className="note">“{job.resolved_caption}”</div>}
          </div>
        </div>
      ))}
    </div>
  )
}
