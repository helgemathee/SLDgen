import { describe, expect, it } from 'vitest'
import type { JobSummary, ParamValue } from '../api/types'
import { defaultParams } from './params'
import {
  estimateBatchSeconds,
  findDuplicates,
  randomSeedBlock,
  resizeVariants,
  sequentialSeeds,
  variantTitle,
  type Variant,
} from './variants'

const base = defaultParams()

function job(params: Record<string, ParamValue>): JobSummary {
  return {
    id: `01J${Math.random().toString(36).slice(2, 10).toUpperCase()}`,
    title: 'car_03',
    state: 'complete',
    desired_state: 'run',
    num_iter: 4000,
    target_epoch: 4000,
    current_epoch: 4000,
    progress: 1,
    resolved_caption: null,
    target_sha256: 'abc',
    parent_job_id: null,
    batch_id: null,
    priority: 0,
    error_class: null,
    error_message: null,
    disk_bytes: null,
    created_at: '2026-08-09T10:00:00Z',
    updated_at: '2026-08-09T10:00:00Z',
    started_at: null,
    finished_at: null,
    preview_url: '/api/jobs/x/preview',
    viewed_epoch: null,
    favorite_count: 0,
    params: { ...base, ...params },
  }
}

describe('sequentialSeeds', () => {
  it('runs from the parent seed, exclusive of it', () => {
    expect(sequentialSeeds(1041, 4)).toEqual([1042, 1043, 1044, 1045])
  })

  it('is reproducible -- the same parent and count give the same block', () => {
    expect(sequentialSeeds(7, 3)).toEqual(sequentialSeeds(7, 3))
  })

  it('returns nothing for a count of zero', () => {
    expect(sequentialSeeds(10, 0)).toEqual([])
  })
})

describe('randomSeedBlock', () => {
  it('is contiguous, so a random block still reads as one family', () => {
    const seeds = randomSeedBlock(5, () => 0.5)
    expect(seeds).toHaveLength(5)
    for (let index = 1; index < seeds.length; index += 1) {
      expect(seeds[index]).toBe(seeds[index - 1] + 1)
    }
  })

  it('lands away from the low integers a sequential block would occupy', () => {
    expect(randomSeedBlock(1, () => 0)[0]).toBeGreaterThanOrEqual(1000)
    expect(randomSeedBlock(1, () => 0.999999)[0]).toBeLessThan(1_000_000)
  })
})

describe('resizeVariants', () => {
  const make = (index: number): Variant => ({
    key: `k${index}`,
    enabled: true,
    params: { ...base, seed: index },
  })
  const rows = [make(0), make(1), make(2)]

  it('keeps edits to rows that survive a shrink', () => {
    const edited = [{ ...rows[0], params: { ...base, caption: 'edited' } }, rows[1], rows[2]]
    const shrunk = resizeVariants(edited, 2, make)
    expect(shrunk).toHaveLength(2)
    expect(shrunk[0].params.caption).toBe('edited')
  })

  it('appends new rows when growing and leaves existing ones alone', () => {
    const grown = resizeVariants(rows, 5, make)
    expect(grown).toHaveLength(5)
    expect(grown.slice(0, 3)).toEqual(rows)
  })

  it('is a no-op at the same size', () => {
    expect(resizeVariants(rows, 3, make)).toHaveLength(3)
  })
})

describe('findDuplicates', () => {
  it('flags a variant whose structural parameters already exist', () => {
    const existing = [job({ seed: 1042 }), job({ seed: 1043 })]
    expect(findDuplicates({ ...base, seed: 1042 }, existing)).toHaveLength(1)
    expect(findDuplicates({ ...base, seed: 9999 }, existing)).toHaveLength(0)
  })

  it('ignores operational settings, which never affect the result', () => {
    const existing = [job({ seed: 5, save_interval: 100 })]
    expect(findDuplicates({ ...base, seed: 5, save_interval: 50 }, existing)).toHaveLength(1)
  })

  it('ignores the copied input paths, which differ per job by construction', () => {
    const existing = [job({ seed: 5, avoid: ['jobs/A/inputs/avoid_000.svg'] })]
    const candidate = { ...base, seed: 5, avoid: ['jobs/B/inputs/avoid_000.svg'] }
    expect(findDuplicates(candidate, existing)).toHaveLength(1)
  })

  it('treats a different caption as a different run', () => {
    const existing = [job({ seed: 5, caption: 'a racing car' })]
    expect(findDuplicates({ ...base, seed: 5, caption: 'a bicycle' }, existing)).toHaveLength(0)
  })

  it('skips jobs fetched without params rather than guessing', () => {
    const bare = { ...job({ seed: 5 }), params: undefined }
    expect(findDuplicates({ ...base, seed: 5 }, [bare])).toHaveLength(0)
  })
})

describe('variantTitle', () => {
  const parent = { ...base, seed: 1041, caption: 'a vintage racing car' }

  it('names the seed when only the seed differs', () => {
    expect(variantTitle('car_03', { ...parent, seed: 1042 }, parent)).toBe('car_03 · s1042')
  })

  it('names both when the caption differs too', () => {
    const title = variantTitle('car_03', { ...parent, seed: 1043, caption: 'side view' }, parent)
    expect(title).toBe('car_03 · s1043 · side view')
  })

  it('says so when a variant hands the caption back to BLIP-2', () => {
    expect(variantTitle('car_03', { ...parent, caption: '' }, parent)).toContain('auto caption')
  })

  it('still identifies a variant that changed neither', () => {
    expect(variantTitle('car_03', { ...parent }, parent)).toBe('car_03 · s1041')
  })

  it('truncates a long caption rather than filling the rail with it', () => {
    const long = 'a single line drawing of a very elaborate vintage racing car at speed'
    const title = variantTitle('car_03', { ...parent, caption: long }, parent)
    expect(title.length).toBeLessThan(50)
    expect(title).toContain('…')
  })
})

describe('estimateBatchSeconds', () => {
  const rows = (count: number, enabled = true): Variant[] =>
    Array.from({ length: count }, (_unused, index) => ({
      key: `k${index}`,
      enabled,
      params: base,
    }))

  it('sums the runs, because the queue is FIFO one at a time', () => {
    expect(estimateBatchSeconds(rows(5), 400, 5)).toBe((5 * 400) / 5)
  })

  it('counts only the enabled rows', () => {
    const mixed = [...rows(2), ...rows(3, false)]
    expect(estimateBatchSeconds(mixed, 400, 5)).toBe((2 * 400) / 5)
  })

  it('declines to estimate without a measured rate', () => {
    expect(estimateBatchSeconds(rows(3), 400, null)).toBeNull()
    expect(estimateBatchSeconds(rows(3), 400, 0)).toBeNull()
  })
})
