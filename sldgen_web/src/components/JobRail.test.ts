import { describe, expect, it } from 'vitest'
import type { JobState, JobSummary } from '../api/types'
import { filterJobs } from './JobRail'

function job(id: string, state: JobState, starred: boolean): JobSummary {
  return {
    id,
    title: id,
    state,
    desired_state: 'run',
    num_iter: 4000,
    target_epoch: 400,
    current_epoch: 0,
    progress: 0,
    resolved_caption: null,
    target_sha256: 'abc',
    parent_job_id: null,
    batch_id: null,
    priority: 0,
    error_class: null,
    error_message: null,
    disk_bytes: null,
    created_at: '2026-09-26T10:00:00Z',
    updated_at: '2026-09-26T10:00:00Z',
    started_at: null,
    finished_at: null,
    preview_url: `/api/jobs/${id}/preview`,
    viewed_epoch: null,
    favorite_count: 0,
    starred,
  }
}

const jobs = [
  job('a', 'running', true),
  job('b', 'running', false),
  job('c', 'waiting', true),
  job('d', 'queued', false),
]
const ids = (list: JobSummary[]) => list.map((entry) => entry.id)

describe('filterJobs favourites', () => {
  it('shows everything when the star chip is off', () => {
    expect(ids(filterJobs(jobs, { states: new Set(), text: '', sort: 'newest' }))).toEqual([
      'a',
      'b',
      'c',
      'd',
    ])
  })

  it('keeps only starred jobs when it is on', () => {
    expect(
      ids(filterJobs(jobs, { states: new Set(), starredOnly: true, text: '', sort: 'newest' })),
    ).toEqual(['a', 'c'])
  })

  it('combines with the state chips as "and"', () => {
    expect(
      ids(
        filterJobs(jobs, {
          states: new Set<JobState>(['running']),
          starredOnly: true,
          text: '',
          sort: 'newest',
        }),
      ),
    ).toEqual(['a'])
  })
})
