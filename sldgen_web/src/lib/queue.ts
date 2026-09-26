/**
 * The queue as the worker sees it: queued jobs that want to run, highest
 * priority first, then earliest submitted (Spec 2 SS6; the worker's
 * `claim_next_job` orders by priority DESC, created_at ASC, id ASC).
 *
 * Pure functions, so the order shown is tested against the worker's rule.
 */
import type { JobSummary } from '../api/types'

/** The highest priority the UI offers: a single fat digit in the rail. */
export const MAX_PRIORITY = 9

export function clampPriority(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.min(MAX_PRIORITY, Math.max(0, Math.round(value)))
}

/** Is this job waiting in the queue for the worker? */
export function isQueued(job: JobSummary): boolean {
  return job.state === 'queued' && job.desired_state === 'run'
}

/** Queued jobs in the order the worker will take them. */
export function queueOrder(jobs: JobSummary[]): JobSummary[] {
  return jobs.filter(isQueued).sort(
    (a, b) =>
      b.priority - a.priority ||
      Date.parse(a.created_at) - Date.parse(b.created_at) ||
      (a.id < b.id ? -1 : a.id > b.id ? 1 : 0),
  )
}

/** Job id -> 1-based place in the queue (1 = next). */
export function queuePositions(jobs: JobSummary[]): Map<string, number> {
  return new Map(queueOrder(jobs).map((job, index) => [job.id, index + 1]))
}

/**
 * The priority that puts this job at the head of the queue: one above the
 * highest other queued job (capped), or its own if it already leads.
 */
export function runNextPriority(job: JobSummary, jobs: JobSummary[]): number {
  const order = queueOrder(jobs)
  if (order[0]?.id === job.id) return job.priority
  const others = order.filter((other) => other.id !== job.id)
  if (!others.length) return job.priority
  const top = others[0].priority
  return clampPriority(Math.max(job.priority, top + 1))
}

/** Rail order "queue": the queue first, next job on top, then everything else newest first. */
export function queueFirst(jobs: JobSummary[]): JobSummary[] {
  const queued = queueOrder(jobs)
  const ids = new Set(queued.map((job) => job.id))
  return [...queued, ...jobs.filter((job) => !ids.has(job.id))]
}

/** `#1 next`, `#3 in queue`, or '' for a job that is not queued. */
export function queueLabel(position: number | undefined): string {
  if (!position) return ''
  return position === 1 ? 'next up' : `#${position} in queue`
}
