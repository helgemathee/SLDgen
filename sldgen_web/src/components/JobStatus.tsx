import type { JobSummary } from '../api/types'
import { isQueued, queueLabel } from '../lib/queue'
import { Ring } from './Ring'

/**
 * A job's status mark: the progress ring, except for a queued job that has
 * been given a priority, which shows that priority as a fat number instead.
 * When the worker takes the job it is `running`, and the ring comes back.
 */
export function JobStatus({
  job,
  size = 20,
  position,
}: {
  job: JobSummary
  size?: number
  /** Place in the queue (1 = next), for the tooltip. */
  position?: number
}) {
  if (isQueued(job) && job.priority > 0) {
    const where = queueLabel(position)
    const label = `Priority ${job.priority}${where ? ` · ${where}` : ''}`
    return (
      <span
        className="priority-badge"
        style={{ width: size, height: size, fontSize: Math.round(size * 0.68) }}
        role="img"
        aria-label={label}
        title={label}
      >
        {job.priority}
      </span>
    )
  }
  return (
    <Ring
      size={size}
      state={job.state}
      currentEpoch={job.current_epoch}
      targetEpoch={job.target_epoch}
      numIter={job.num_iter}
    />
  )
}
