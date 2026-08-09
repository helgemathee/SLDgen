import { describe, expect, it } from 'vitest'
import type { JobDetail } from '../api/types'
import {
  emptyFormState,
  formStateFromJob,
  hydrateFormState,
  toInputs,
  toParams,
} from './formstate'
import { defaultParams } from './params'

describe('emptyFormState', () => {
  it('keeps optional parameters out of `params` and in `optional`', () => {
    const state = emptyFormState()
    expect(state.params.origin).toBeUndefined()
    expect(state.optional.origin.enabled).toBe(false)
    expect(state.optional.origin.value).toEqual([0.5, 0.5])
  })
})

describe('the {enabled, value} contract -- Spec 3 SS9', () => {
  it('keeps an origin pin when the origin is switched off', () => {
    const state = emptyFormState()
    state.optional.origin = { enabled: true, value: [0.31, 0.78] }

    // Switching it off must not discard the coordinates: the next form shows
    // the pin exactly where it was, one click from being used again.
    state.optional.origin.enabled = false

    const round = hydrateFormState(JSON.parse(JSON.stringify(state)))
    expect(round.optional.origin.enabled).toBe(false)
    expect(round.optional.origin.value).toEqual([0.31, 0.78])
  })

  it('omits a disabled origin from the submitted parameters', () => {
    const state = emptyFormState()
    state.optional.origin = { enabled: false, value: [0.31, 0.78] }
    expect(toParams(state).origin).toBeNull()

    state.optional.origin.enabled = true
    expect(toParams(state).origin).toEqual([0.31, 0.78])
  })

  it('keeps chosen constraint sources when the constraint is switched off', () => {
    const state = emptyFormState()
    state.optional.avoid = {
      enabled: true,
      value: null,
      inputs: [{ source_kind: 'job', source_job_id: 'A', path: 'final_sld.svg' }],
    }
    state.optional.avoid.enabled = false

    const round = hydrateFormState(JSON.parse(JSON.stringify(state)))
    expect(round.optional.avoid.inputs).toHaveLength(1)
    expect(toInputs(round)).toEqual([])
  })
})

describe('hydrateFormState', () => {
  it('returns defaults for a missing or malformed payload', () => {
    expect(hydrateFormState(null).numIter).toBe(4000)
    expect(hydrateFormState('nonsense').targetEpoch).toBe(400)
    expect(hydrateFormState(42).maskMode).toBe('clean')
  })

  it('fills a parameter added since the payload was written', () => {
    const state = hydrateFormState({ params: { seed: 9 }, numIter: 1000 })
    expect(state.params.seed).toBe(9)
    expect(state.params.conditioning_scale).toBe(defaultParams().conditioning_scale)
    expect(state.numIter).toBe(1000)
  })

  it('rejects a mask mode it does not recognise', () => {
    expect(hydrateFormState({ maskMode: 'wat' }).maskMode).toBe('clean')
    expect(hydrateFormState({ maskMode: 'control' }).maskMode).toBe('control')
  })

  it('refuses a non-positive horizon rather than submitting one', () => {
    expect(hydrateFormState({ numIter: 0 }).numIter).toBe(4000)
    expect(hydrateFormState({ targetEpoch: -5 }).targetEpoch).toBe(400)
  })

  it('round-trips a complete state unchanged', () => {
    const state = emptyFormState()
    state.params.seed = 77
    state.maskMode = 'guide'
    state.numIter = 2000
    state.targetEpoch = 250
    expect(hydrateFormState(JSON.parse(JSON.stringify(state)))).toEqual(state)
  })
})

function jobDetail(overrides: Partial<JobDetail> = {}): JobDetail {
  return {
    id: '01JOB',
    title: 'car_03',
    state: 'complete',
    desired_state: 'run',
    num_iter: 4000,
    target_epoch: 400,
    current_epoch: 400,
    progress: 0.1,
    resolved_caption: 'a racing car',
    target_sha256: 'sha',
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
    preview_url: '/api/jobs/01JOB/preview',
    params: { ...defaultParams(), seed: 1041 },
    structural_params: {},
    operational_params: {},
    segments: [],
    inputs: [],
    artifacts: [],
    state_json: null,
    command: 'python sldgen.py …',
    ...overrides,
  }
}

describe('formStateFromJob', () => {
  it('carries the parent parameters and its budget', () => {
    const state = formStateFromJob(jobDetail())
    expect(state.params.seed).toBe(1041)
    expect(state.numIter).toBe(4000)
    expect(state.targetEpoch).toBe(400)
  })

  it('turns the origin parameter back into an enabled optional field', () => {
    const state = formStateFromJob(
      jobDetail({ params: { ...defaultParams(), origin: [0.2, 0.9] } }),
    )
    expect(state.optional.origin.enabled).toBe(true)
    expect(state.optional.origin.value).toEqual([0.2, 0.9])
  })

  it('leaves the origin off, but placed, when the parent had none', () => {
    const state = formStateFromJob(jobDetail())
    expect(state.optional.origin.enabled).toBe(false)
    expect(state.optional.origin.value).toEqual([0.5, 0.5])
  })

  it('represents inherited inputs as references to the parent copies', () => {
    const state = formStateFromJob(
      jobDetail({
        inputs: [
          {
            id: 1,
            job_id: '01JOB',
            role: 'avoid',
            ordinal: 0,
            source_kind: 'job',
            source_job_id: 'PARENT',
            source_partition_id: null,
            stored_path: 'jobs/01JOB/inputs/avoid_000.svg',
            source_sha256: 'x',
          },
        ],
      }),
    )
    expect(state.optional.avoid.enabled).toBe(true)
    expect(state.optional.avoid.inputs).toEqual([
      {
        source_kind: 'job',
        source_job_id: '01JOB',
        path: 'avoid_000.svg',
        label: 'car_03 · avoid_000.svg',
      },
    ])
  })

  it('never submits an input-backed parameter as a plain parameter', () => {
    // create_job rejects these outright (Spec 2 SS4.3) -- they must travel as
    // `inputs` so the file is copied and its provenance recorded.
    const state = formStateFromJob(
      jobDetail({ params: { ...defaultParams(), avoid: ['jobs/x/inputs/avoid_000.svg'] } }),
    )
    expect(toParams(state).avoid).toBeUndefined()
  })

  it('infers the mask mode from the parent stipple-weight settings', () => {
    expect(formStateFromJob(jobDetail()).maskMode).toBe('clean')
    expect(
      formStateFromJob(
        jobDetail({
          params: { ...defaultParams(), stipple_weight: 'w.png', stipple_weight_mode: 'multiply' },
        }),
      ).maskMode,
    ).toBe('guide')
    expect(
      formStateFromJob(
        jobDetail({
          params: { ...defaultParams(), stipple_weight: 'w.png', stipple_weight_mode: 'replace' },
        }),
      ).maskMode,
    ).toBe('control')
  })
})

describe('toInputs', () => {
  it('emits one entry per chosen reference, for enabled roles only', () => {
    const state = emptyFormState()
    state.optional.avoid = {
      enabled: true,
      value: null,
      inputs: [
        { source_kind: 'job', source_job_id: 'A', path: 'final_sld.svg' },
        { source_kind: 'partition', source_partition_id: 'P', path: 'partition_0.svg' },
      ],
    }
    state.optional.attract = {
      enabled: false,
      value: null,
      inputs: [{ source_kind: 'job', source_job_id: 'B', path: 'final_sld.svg' }],
    }

    const inputs = toInputs(state)
    expect(inputs).toHaveLength(2)
    expect(inputs.every((input) => input.role === 'avoid')).toBe(true)
    expect(inputs[1].source_partition_id).toBe('P')
  })
})
