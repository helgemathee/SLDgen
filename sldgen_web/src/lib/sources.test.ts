import { describe, expect, it } from 'vitest'
import type { JobState, JobSummary } from '../api/types'
import type { InputRef } from './formstate'
import {
  canSupplySvg,
  finalRef,
  frameRef,
  frameSvgPath,
  overlayUrl,
  refKey,
  refLabel,
  searchSources,
  uploadRef,
} from './sources'

function job(overrides: Partial<JobSummary> = {}): JobSummary {
  return {
    id: '01JOB000000000000000000001',
    title: 'firefighter',
    state: 'complete' as JobState,
    desired_state: 'run',
    num_iter: 4000,
    target_epoch: 4000,
    current_epoch: 4000,
    progress: 1,
    resolved_caption: 'a firefighter carrying a hose',
    target_sha256: 'abc',
    parent_job_id: null,
    batch_id: null,
    priority: 0,
    error_class: null,
    error_message: null,
    disk_bytes: null,
    created_at: '2026-09-01T00:00:00Z',
    updated_at: '2026-09-01T00:00:00Z',
    started_at: null,
    finished_at: null,
    preview_url: '/api/jobs/01JOB000000000000000000001/preview',
    viewed_epoch: null,
    favorite_count: 0,
    starred: false,
    ...overrides,
  }
}

describe('canSupplySvg', () => {
  it('offers a completed job', () => {
    expect(canSupplySvg(job())).toBe(true)
  })

  it('offers a running job that has drawn a frame', () => {
    expect(canSupplySvg(job({ state: 'running', current_epoch: 1700 }))).toBe(true)
  })

  // The point of the feature: "avoid that other run at 1700" while it is still
  // going. Its frames up to now are finished geometry.
  it('offers a paused and a failed job that got far enough to save frames', () => {
    expect(canSupplySvg(job({ state: 'paused', current_epoch: 800 }))).toBe(true)
    expect(canSupplySvg(job({ state: 'failed', current_epoch: 300 }))).toBe(true)
  })

  it('refuses a job that has drawn nothing', () => {
    expect(canSupplySvg(job({ state: 'queued', current_epoch: 0 }))).toBe(false)
  })

  it('refuses a job on its way out, whose files are being unlinked', () => {
    expect(canSupplySvg(job({ state: 'deleting', current_epoch: 4000 }))).toBe(false)
  })
})

describe('searchSources', () => {
  const jobs = [
    job({ id: '01AAA', title: 'firefighter', resolved_caption: 'a firefighter' }),
    job({ id: '01BBB', title: 'skull', resolved_caption: 'a human skull' }),
    job({ id: '01CCC', title: null, resolved_caption: null, state: 'queued', current_epoch: 0 }),
  ]

  it('returns every usable job when the query is blank', () => {
    expect(searchSources(jobs, '   ').map((entry) => entry.id)).toEqual(['01AAA', '01BBB'])
  })

  it('matches the title, case-insensitively', () => {
    expect(searchSources(jobs, 'SKU').map((entry) => entry.id)).toEqual(['01BBB'])
  })

  it('matches the caption', () => {
    expect(searchSources(jobs, 'human').map((entry) => entry.id)).toEqual(['01BBB'])
  })

  it('matches the id, so a link pasted from elsewhere finds its job', () => {
    expect(searchSources(jobs, '01aaa').map((entry) => entry.id)).toEqual(['01AAA'])
  })

  it('never offers a job that has nothing to offer, however well it matches', () => {
    expect(searchSources(jobs, '01CCC')).toEqual([])
  })
})

describe('frame references', () => {
  // run.py pads the PNG (iter_1700.png) and leaves the SVG unpadded. Getting
  // this wrong produces a 404 at submission, not a wrong drawing -- but only
  // after the form has been filled in.
  it('names the SVG the way run.py writes it, relative to the run directory', () => {
    expect(frameSvgPath(1700)).toBe('svg_logs/svg_iter1700.svg')
    expect(frameSvgPath(100)).toBe('svg_logs/svg_iter100.svg')
  })

  it('carries the job id and says which iteration it is', () => {
    const reference = frameRef(job(), 1700)
    expect(reference.source_kind).toBe('job')
    expect(reference.source_job_id).toBe('01JOB000000000000000000001')
    expect(reference.path).toBe('svg_logs/svg_iter1700.svg')
    expect(reference.label).toContain('epoch 1700')
    expect(reference.label).toContain('firefighter')
  })

  it('still offers the final export', () => {
    expect(finalRef(job()).path).toBe('final_sld.svg')
  })

  it('references an upload by digest, not by path', () => {
    const reference = uploadRef(
      { sha256: 'deadbeef', width: null, height: null, bytes: 12, filename: 'ring.svg', url: '' },
      'ring.svg',
    )
    expect(reference).toEqual({ source_kind: 'upload', sha256: 'deadbeef', label: 'ring.svg' })
    expect(reference.path).toBeUndefined()
  })
})

describe('overlayUrl', () => {
  it('draws a frame from the source job run directory', () => {
    expect(overlayUrl(frameRef(job(), 1700))).toBe(
      '/api/jobs/01JOB000000000000000000001/files/target/run/svg_logs/svg_iter1700.svg',
    )
  })

  it('draws a partition piece from the partition directory', () => {
    const reference: InputRef = {
      source_kind: 'partition',
      source_partition_id: '01PART',
      path: 'partition_2.svg',
    }
    expect(overlayUrl(reference)).toBe('/api/partitions/01PART/files/partition_2.svg')
  })

  // An upload has no job directory to answer from; before this it produced
  // /api/jobs/undefined/... and the overlay silently did not appear.
  it('draws an uploaded SVG from the content-addressed store', () => {
    expect(overlayUrl(uploadRef({ sha256: 'abc123' } as never, 'ring.svg'))).toBe(
      '/api/uploads/abc123',
    )
  })

  it('draws nothing for an uploaded PNG, which is a weight map and not geometry', () => {
    expect(overlayUrl(uploadRef({ sha256: 'abc123' } as never, 'weight.png'))).toBeNull()
  })

  it('draws nothing for a reference missing the id it would need', () => {
    expect(overlayUrl({ source_kind: 'job', path: 'final_sld.svg' })).toBeNull()
    expect(overlayUrl({ source_kind: 'upload', label: 'ring.svg' })).toBeNull()
  })
})

describe('refLabel and refKey', () => {
  it('labels a reference by what it was called, then by what it is', () => {
    expect(refLabel(frameRef(job(), 200))).toContain('epoch 200')
    expect(refLabel({ source_kind: 'job', path: 'final_sld.svg' })).toBe('final_sld.svg')
  })

  it('tells two frames of the same job apart', () => {
    expect(refKey(frameRef(job(), 200))).not.toBe(refKey(frameRef(job(), 300)))
  })

  it('tells the same frame of two jobs apart', () => {
    expect(refKey(frameRef(job({ id: 'A' }), 200))).not.toBe(
      refKey(frameRef(job({ id: 'B' }), 200)),
    )
  })
})
