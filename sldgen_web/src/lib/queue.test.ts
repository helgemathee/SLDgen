import { describe, expect, it } from 'vitest'
import type { JobState, JobSummary } from '../api/types'
import {
  MAX_PRIORITY,
  clampPriority,
  queueFirst,
  queueLabel,
  queueOrder,
  queuePositions,
  runNextPriority,
} from './queue'

function job(id: string, state: JobState, priority: number, created: string, desired = 'run'): JobSummary {
  return {
    id,
    title: id,
    state,
    desired_state: desired,
    num_iter: 4000,
    target_epoch: 400,
    current_epoch: 0,
    progress: 0,
    resolved_caption: null,
    target_sha256: 'abc',
    parent_job_id: null,
    batch_id: null,
    priority,
    error_class: null,
    error_message: null,
    disk_bytes: null,
    created_at: created,
    updated_at: created,
    started_at: null,
    finished_at: null,
    preview_url: '',
    viewed_epoch: null,
    favorite_count: 0,
    starred: false,
  } as JobSummary
}

const ids = (list: JobSummary[]) => list.map((entry) => entry.id)

describe('queueOrder (the worker rule: priority desc, then submission)', () => {
  const jobs = [
    job('new', 'queued', 0, '2026-09-26T12:00:00Z'),
    job('old', 'queued', 0, '2026-09-26T10:00:00Z'),
    job('urgent', 'queued', 2, '2026-09-26T13:00:00Z'),
    job('mid', 'queued', 1, '2026-09-26T09:00:00Z'),
    job('running', 'running', 5, '2026-09-26T08:00:00Z'),
    job('paused-ask', 'queued', 9, '2026-09-26T08:00:00Z', 'pause'),
  ]

  it('orders by priority, then earliest submitted', () => {
    expect(ids(queueOrder(jobs))).toEqual(['urgent', 'mid', 'old', 'new'])
  })

  it('skips jobs that are not queued to run', () => {
    expect(ids(queueOrder(jobs))).not.toContain('running')
    expect(ids(queueOrder(jobs))).not.toContain('paused-ask')
  })

  it('numbers positions from 1', () => {
    const positions = queuePositions(jobs)
    expect(positions.get('urgent')).toBe(1)
    expect(positions.get('new')).toBe(4)
    expect(positions.get('running')).toBeUndefined()
  })

  it('breaks exact ties by id, as the worker does', () => {
    const tie = [job('b', 'queued', 0, '2026-09-26T10:00:00Z'), job('a', 'queued', 0, '2026-09-26T10:00:00Z')]
    expect(ids(queueOrder(tie))).toEqual(['a', 'b'])
  })

  it('puts the queue first in the rail, rest in their order', () => {
    expect(ids(queueFirst(jobs))).toEqual(['urgent', 'mid', 'old', 'new', 'running', 'paused-ask'])
  })
})

describe('priorities', () => {
  it('clamps to 0..MAX_PRIORITY integers', () => {
    expect(clampPriority(-3)).toBe(0)
    expect(clampPriority(2.6)).toBe(3)
    expect(clampPriority(99)).toBe(MAX_PRIORITY)
    expect(clampPriority(Number.NaN)).toBe(0)
  })

  it('run next goes one above the current head', () => {
    const jobs = [
      job('a', 'queued', 2, '2026-09-26T10:00:00Z'),
      job('b', 'queued', 0, '2026-09-26T11:00:00Z'),
    ]
    expect(runNextPriority(jobs[1], jobs)).toBe(3)
    expect(runNextPriority(jobs[0], jobs)).toBe(2)
    expect(runNextPriority(job('solo', 'queued', 0, '2026-09-26T10:00:00Z'), [])).toBe(0)
  })

  it('labels the head "next up"', () => {
    expect(queueLabel(1)).toBe('next up')
    expect(queueLabel(3)).toBe('#3 in queue')
    expect(queueLabel(undefined)).toBe('')
  })
})
