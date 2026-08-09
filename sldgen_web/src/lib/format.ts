/** Formatters for the monospace half of the interface. */

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '—'
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`
}

/** Compact duration: 45s, 8m, 2h14m. Long jobs are minutes, not seconds. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—'
  const total = Math.max(0, Math.round(seconds))
  if (total < 60) return `${total}s`
  const minutes = Math.floor(total / 60)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  return `${hours}h${String(minutes % 60).padStart(2, '0')}m`
}

/**
 * Time left in this run -- to `target_epoch`, not to the horizon.
 *
 * The budget is where the run stops, so an ETA to the horizon would be an
 * estimate for work nobody has asked for yet (Spec 1 SS3).
 */
export function estimateRemaining(
  currentEpoch: number,
  targetEpoch: number,
  itersPerSec: number | null | undefined,
): number | null {
  if (!itersPerSec || itersPerSec <= 0) return null
  const remaining = targetEpoch - currentEpoch
  if (remaining <= 0) return 0
  return remaining / itersPerSec
}

/** Mean it/s across finished segments -- what the queue estimate is built on. */
export function meanItersPerSec(
  segments: { start_epoch: number; end_epoch: number | null; started_at: string; finished_at: string | null }[],
): number | null {
  let epochs = 0
  let seconds = 0
  for (const segment of segments) {
    if (segment.end_epoch === null || segment.finished_at === null) continue
    const elapsed =
      (Date.parse(segment.finished_at) - Date.parse(segment.started_at)) / 1000
    const done = segment.end_epoch - segment.start_epoch
    if (elapsed > 0 && done > 0) {
      epochs += done
      seconds += elapsed
    }
  }
  return seconds > 0 ? epochs / seconds : null
}

export function formatRate(itersPerSec: number | null | undefined): string {
  if (!itersPerSec) return '—'
  return `${itersPerSec.toFixed(1)} it/s`
}

export function formatTimestamp(value: string | null | undefined): string {
  if (!value) return '—'
  const parsed = Date.parse(value)
  if (Number.isNaN(parsed)) return value
  return new Date(parsed).toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatAgo(value: string | null | undefined, now = Date.now()): string {
  if (!value) return '—'
  const parsed = Date.parse(value)
  if (Number.isNaN(parsed)) return value
  return `${formatDuration((now - parsed) / 1000)} ago`
}

/** Titles are optional in the schema; the id's tail is a usable stand-in. */
export function jobLabel(job: { id: string; title: string | null }): string {
  return job.title?.trim() || job.id.slice(-6)
}

/** Plain-words version of the failure taxonomy (Spec 2 SS15, Spec 3 SS15). */
export const ERROR_COPY: Record<string, { headline: string; advice: string }> = {
  validation: {
    headline: 'The parameters were rejected',
    advice: 'Fix them and run again — this job cannot be retried unchanged.',
  },
  environment: {
    headline: 'The machine needs attention',
    advice:
      'Hugging Face auth, gated model access, CUDA or Concorde. Fix it on the host and retry; the parameters are fine.',
  },
  oom: {
    headline: 'The GPU ran out of memory',
    advice: 'Usually another process is holding the card. Free it and retry.',
  },
  interrupted: {
    headline: 'The run was interrupted',
    advice: 'A stop, a reboot or a worker restart. Resume it from its last checkpoint.',
  },
  unknown: {
    headline: 'The run failed',
    advice: 'The log below is the whole story.',
  },
}
