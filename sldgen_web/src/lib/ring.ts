import type { JobState } from '../api/types'

/**
 * The progress ring (Spec 3 SS5).
 *
 * State is encoded by the ring plus colour, not by a separate badge, because the
 * ring does double duty:
 *
 *   filled arc = current_epoch / num_iter   -- how far through the horizon
 *   tick mark  = target_epoch               -- where this job is currently headed
 *
 * A job at 400 of 4000 with a budget of 400 therefore shows a small filled arc
 * with the tick sitting exactly at its leading edge: complete against its
 * budget, incomplete against the horizon. That is the "partially completed"
 * distinction, and it needs no words.
 */

export interface RingInput {
  state: JobState
  currentEpoch: number
  targetEpoch: number
  numIter: number
  radius?: number
}

export interface RingGeometry {
  radius: number
  circumference: number
  /** Fraction of the horizon completed, clamped to [0, 1]. */
  progress: number
  /** Fraction of the horizon this run is headed for, clamped to [0, 1]. */
  target: number
  /** stroke-dasharray for the filled arc. */
  dashArray: string
  /** Degrees clockwise from 12 o'clock for the target tick. */
  tickAngle: number
  /** Degrees clockwise from 12 o'clock for the running leading-edge marker. */
  leadingAngle: number
  strokeDash: string | undefined
  colorVar: string
  opacity: number
}

const COLOR_BY_STATE: Record<JobState, string> = {
  queued: 'var(--st-queued)',
  running: 'var(--st-running)',
  waiting: 'var(--st-waiting)',
  paused: 'var(--st-queued)',
  complete: 'var(--st-complete)',
  failed: 'var(--st-failed)',
  deleting: 'var(--st-deleting)',
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, value))

export function ringGeometry(input: RingInput): RingGeometry {
  const radius = input.radius ?? 8
  const circumference = 2 * Math.PI * radius
  const horizon = input.numIter > 0 ? input.numIter : 1
  const progress = clamp01(input.currentEpoch / horizon)
  const target = clamp01(input.targetEpoch / horizon)

  // A complete job fills the ring regardless of rounding: reaching the horizon
  // and displaying 99% would be a lie told by floating point.
  const filled = input.state === 'complete' ? 1 : progress

  return {
    radius,
    circumference,
    progress,
    target,
    dashArray: `${(filled * circumference).toFixed(3)} ${circumference.toFixed(3)}`,
    tickAngle: target * 360,
    leadingAngle: filled * 360,
    // Paused is the same colour as queued, so the dash is what distinguishes it.
    strokeDash: input.state === 'paused' ? '2 2' : undefined,
    colorVar: COLOR_BY_STATE[input.state],
    opacity: input.state === 'deleting' ? 0.3 : 1,
  }
}

/** Point on the ring at an angle measured clockwise from 12 o'clock. */
export function ringPoint(radius: number, degrees: number, center: number) {
  const radians = ((degrees - 90) * Math.PI) / 180
  return {
    x: center + radius * Math.cos(radians),
    y: center + radius * Math.sin(radians),
  }
}
