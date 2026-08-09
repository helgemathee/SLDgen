import { describe, expect, it } from 'vitest'
import { SSE_FAILURES_BEFORE_POLLING, backoffDelay } from './backoff'
import {
  ERROR_COPY,
  estimateRemaining,
  formatBytes,
  formatDuration,
  formatRate,
  jobLabel,
  meanItersPerSec,
} from './format'

describe('formatBytes', () => {
  it('scales through the units', () => {
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(2048)).toBe('2.0 KB')
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MB')
    expect(formatBytes(4.2 * 1024 ** 3)).toBe('4.2 GB')
  })

  it('drops the decimal once the number is large enough not to need it', () => {
    expect(formatBytes(310 * 1024 * 1024)).toBe('310 MB')
  })

  it('renders an unknown size as a dash rather than 0 B', () => {
    expect(formatBytes(null)).toBe('—')
    expect(formatBytes(undefined)).toBe('—')
    expect(formatBytes(0)).toBe('0 B')
  })
})

describe('formatDuration', () => {
  it('uses seconds, minutes then hours', () => {
    expect(formatDuration(45)).toBe('45s')
    expect(formatDuration(8 * 60)).toBe('8m')
    expect(formatDuration(2 * 3600 + 14 * 60)).toBe('2h14m')
  })

  it('zero-pads the minutes in an hour figure', () => {
    expect(formatDuration(3600 + 5 * 60)).toBe('1h05m')
  })

  it('handles nothing to report', () => {
    expect(formatDuration(null)).toBe('—')
    expect(formatDuration(Number.POSITIVE_INFINITY)).toBe('—')
    expect(formatDuration(-5)).toBe('0s')
  })
})

describe('estimateRemaining', () => {
  it('counts down to the budget, not to the horizon', () => {
    // 1512 of a 4000 horizon, budgeted to 2000, at 4.9 it/s: the run stops at
    // 2000, so an estimate to 4000 would be for work nobody has asked for.
    expect(estimateRemaining(1512, 2000, 4.9)).toBeCloseTo(488 / 4.9, 6)
  })

  it('is zero once the budget is reached', () => {
    expect(estimateRemaining(400, 400, 5)).toBe(0)
    expect(estimateRemaining(500, 400, 5)).toBe(0)
  })

  it('declines to guess without a rate', () => {
    expect(estimateRemaining(0, 400, null)).toBeNull()
    expect(estimateRemaining(0, 400, 0)).toBeNull()
  })
})

describe('meanItersPerSec', () => {
  const segment = (start: number, end: number | null, seconds: number, finished = true) => ({
    start_epoch: start,
    end_epoch: end,
    started_at: '2026-08-09T10:00:00Z',
    finished_at: finished
      ? new Date(Date.parse('2026-08-09T10:00:00Z') + seconds * 1000).toISOString()
      : null,
  })

  it('averages over epochs rather than over segments', () => {
    // 400 epochs in 100s and 1000 epochs in 200s is 1400/300, not (4 + 5)/2.
    expect(meanItersPerSec([segment(0, 400, 100), segment(400, 1400, 200)])).toBeCloseTo(
      1400 / 300,
      6,
    )
  })

  it('ignores a segment still running', () => {
    expect(meanItersPerSec([segment(0, 400, 100), segment(400, null, 0, false)])).toBeCloseTo(4, 6)
  })

  it('returns null when nothing has finished', () => {
    expect(meanItersPerSec([])).toBeNull()
    expect(meanItersPerSec([segment(0, null, 0, false)])).toBeNull()
  })

  it('ignores a zero-length segment rather than dividing by it', () => {
    expect(meanItersPerSec([segment(400, 400, 0)])).toBeNull()
  })
})

describe('formatRate and jobLabel', () => {
  it('formats a rate to one decimal', () => {
    expect(formatRate(4.94)).toBe('4.9 it/s')
    expect(formatRate(null)).toBe('—')
  })

  it('falls back to the id tail when a job has no title', () => {
    expect(jobLabel({ id: '01JABCDEFGHJKMNPQR', title: null })).toBe('KMNPQR')
    expect(jobLabel({ id: '01JABCDEFGHJKMNPQR', title: '  ' })).toHaveLength(6)
    expect(jobLabel({ id: 'x', title: 'car_03' })).toBe('car_03')
  })
})

describe('ERROR_COPY', () => {
  it('covers every class the worker can assign', () => {
    for (const name of ['validation', 'environment', 'oom', 'interrupted', 'unknown']) {
      expect(ERROR_COPY[name]).toBeDefined()
      expect(ERROR_COPY[name].headline).toBeTruthy()
      expect(ERROR_COPY[name].advice).toBeTruthy()
    }
  })

  it('speaks plainly rather than exposing the internal name', () => {
    expect(ERROR_COPY.oom.headline).toBe('The GPU ran out of memory')
    expect(ERROR_COPY.oom.headline.toLowerCase()).not.toContain('oom')
  })
})

describe('backoffDelay', () => {
  it('grows the ceiling exponentially', () => {
    expect(backoffDelay(0, { random: () => 1 })).toBe(500)
    expect(backoffDelay(1, { random: () => 1 })).toBe(1000)
    expect(backoffDelay(4, { random: () => 1 })).toBe(8000)
  })

  it('caps, so a long outage does not turn into a ten-minute wait', () => {
    expect(backoffDelay(50, { random: () => 1 })).toBe(15_000)
  })

  it('keeps the first retry prompt even when the jitter draws zero', () => {
    // An API restart takes under a second and should not cost a visible stall.
    expect(backoffDelay(0, { random: () => 0 })).toBe(250)
    expect(backoffDelay(9, { random: () => 0 })).toBe(250)
  })

  it('jitters, so several tabs do not retry in lockstep', () => {
    expect(backoffDelay(3, { random: () => 0.25 })).not.toBe(backoffDelay(3, { random: () => 0.9 }))
  })

  it('gives up on SSE after a small number of failures', () => {
    expect(SSE_FAILURES_BEFORE_POLLING).toBeGreaterThan(1)
    expect(SSE_FAILURES_BEFORE_POLLING).toBeLessThan(10)
  })
})
