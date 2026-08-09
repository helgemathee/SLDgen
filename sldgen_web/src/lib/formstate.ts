import type { JobDetail, ParamValue, Params } from '../api/types'
import { PARAM_SPECS, defaultParams } from './params'

/**
 * Parameter persistence (Spec 3 SS9).
 *
 * **The next job starts from the last job's settings.** Every optional parameter
 * is stored as `{enabled, value}` and *both fields persist independently*:
 * disabling the origin does not discard its coordinates, so the next new-job
 * form shows the pin exactly where it was, switched off, one click from being
 * used again. Losing parameter state between submissions is the single most
 * annoying thing this UI could do, and this shape is what prevents it.
 *
 * Held server-side (`/api/params/last`), so it survives a browser change or a
 * cache clear.
 */

export type MaskMode = 'clean' | 'guide' | 'control'

/** A reference to a file the service will copy into the job (Spec 2 SS4.3). */
export interface InputRef {
  source_kind: 'job' | 'partition' | 'upload'
  source_job_id?: string
  source_partition_id?: string
  sha256?: string
  path?: string
  /** What to show in the UI; never sent. */
  label?: string
}

export interface OptionalField {
  enabled: boolean
  value: ParamValue
  /** Set instead of `value` for the four input-backed roles. */
  inputs?: InputRef[]
}

export interface FormState {
  /** Non-optional parameters, at their current values. */
  params: Params
  /** The optional flags, each keeping its value while switched off. */
  optional: Record<string, OptionalField>
  maskMode: MaskMode
  /** The horizon (num_iter) and this run's budget (target_epoch). */
  numIter: number
  targetEpoch: number
}

export const OPTIONAL_NAMES = PARAM_SPECS.filter((spec) => spec.optional).map((spec) => spec.name)

const INPUT_ROLES = ['avoid', 'attract', 'init_points', 'stipple_weight'] as const
export type InputRole = (typeof INPUT_ROLES)[number]

export function emptyFormState(): FormState {
  const params = defaultParams()
  const optional: Record<string, OptionalField> = {}
  for (const name of OPTIONAL_NAMES) {
    optional[name] = {
      enabled: false,
      value: name === 'origin' ? [0.5, 0.5] : null,
      inputs: INPUT_ROLES.includes(name as InputRole) ? [] : undefined,
    }
    delete params[name]
  }
  return {
    params,
    optional,
    maskMode: 'clean',
    numIter: 4000,
    targetEpoch: 400,
  }
}

/**
 * Restore a persisted state, filling anything missing from defaults.
 *
 * Tolerant on purpose: this payload is written by an older build of the UI as
 * often as not, and a parameter added since should appear at its default rather
 * than making the whole form unusable.
 */
export function hydrateFormState(saved: unknown): FormState {
  const base = emptyFormState()
  if (!saved || typeof saved !== 'object') return base
  const candidate = saved as Partial<FormState>

  const params = { ...base.params }
  if (candidate.params && typeof candidate.params === 'object') {
    for (const spec of PARAM_SPECS) {
      if (spec.optional) continue
      const value = (candidate.params as Params)[spec.name]
      if (value !== undefined) params[spec.name] = value
    }
  }

  const optional = { ...base.optional }
  if (candidate.optional && typeof candidate.optional === 'object') {
    for (const name of OPTIONAL_NAMES) {
      const field = (candidate.optional as Record<string, OptionalField>)[name]
      if (!field) continue
      optional[name] = {
        enabled: Boolean(field.enabled),
        value: field.value ?? base.optional[name].value,
        inputs: field.inputs ?? base.optional[name].inputs,
      }
    }
  }

  return {
    params,
    optional,
    maskMode: (['clean', 'guide', 'control'] as MaskMode[]).includes(candidate.maskMode as MaskMode)
      ? (candidate.maskMode as MaskMode)
      : base.maskMode,
    numIter: numberOr(candidate.numIter, base.numIter),
    targetEpoch: numberOr(candidate.targetEpoch, base.targetEpoch),
  }
}

function numberOr(value: unknown, fallback: number): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

/**
 * Carry an existing job's parameters into the form (Spec 3 SS6.5).
 *
 * Input-backed parameters come back as the service's stored paths, which a new
 * job may not set. They are represented as references to the parent's copies
 * instead, so run-again re-copies the same bytes rather than pointing at them.
 */
export function formStateFromJob(job: JobDetail): FormState {
  const state = emptyFormState()
  for (const spec of PARAM_SPECS) {
    if (spec.optional) continue
    if (job.params[spec.name] !== undefined) state.params[spec.name] = job.params[spec.name]
  }

  const origin = job.params.origin as number[] | null
  state.optional.origin = {
    enabled: origin != null,
    value: origin ?? [0.5, 0.5],
  }

  for (const role of INPUT_ROLES) {
    const records = job.inputs.filter((input) => input.role === role)
    state.optional[role] = {
      enabled: records.length > 0,
      value: null,
      inputs: records.map((record) => ({
        source_kind: 'job',
        source_job_id: job.id,
        path: record.stored_path.split('/').pop(),
        label: `${job.title ?? job.id.slice(-6)} · ${record.stored_path.split('/').pop()}`,
      })),
    }
  }

  state.numIter = job.num_iter
  state.targetEpoch = job.target_epoch
  state.maskMode = job.params.stipple_weight
    ? job.params.stipple_weight_mode === 'replace'
      ? 'control'
      : 'guide'
    : 'clean'
  return state
}

/** The parameter object to submit: optional flags folded in only when enabled. */
export function toParams(state: FormState): Params {
  const params: Params = { ...state.params, num_iter: state.numIter }
  for (const name of OPTIONAL_NAMES) {
    const field = state.optional[name]
    if (INPUT_ROLES.includes(name as InputRole)) continue // declared as inputs, not params
    params[name] = field?.enabled ? field.value : null
  }
  return params
}

/** The `inputs` array for POST /api/jobs, from the enabled input-backed roles. */
export function toInputs(state: FormState): Record<string, unknown>[] {
  const inputs: Record<string, unknown>[] = []
  for (const role of INPUT_ROLES) {
    const field = state.optional[role]
    if (!field?.enabled) continue
    for (const reference of field.inputs ?? []) {
      inputs.push({
        role,
        source_kind: reference.source_kind,
        source_job_id: reference.source_job_id,
        source_partition_id: reference.source_partition_id,
        sha256: reference.sha256,
        path: reference.path,
      })
    }
  }
  return inputs
}

/**
 * The stipple-weight flags implied by the mask mode (Spec 3 SS8.2).
 *
 * SLDgen always runs RMBG-1.4 on the target and the RMBG mask also drives the
 * bounding-box rescale, so a browser selection cannot simply "be the mask".
 * What a painted selection *can* do is modulate stipple density, which is what
 * `--stipple-weight` was built for.
 */
export const MASK_MODES: {
  mode: MaskMode
  name: string
  exports: string
  effect: string
}[] = [
  {
    mode: 'clean',
    name: 'Clean up the image',
    exports: 'target.png with the selection knocked out to white',
    effect: 'RMBG still decides density; you have only removed distractions.',
  },
  {
    mode: 'guide',
    name: 'Guide the ink',
    exports: 'also a grayscale weight.png, applied as --stipple-weight-mode multiply',
    effect: 'Your painting modulates density within what RMBG considers subject.',
  },
  {
    mode: 'control',
    name: 'Control the ink',
    exports: 'also weight.png, applied as --stipple-weight-mode replace',
    effect:
      'Your painting is the density field. RMBG no longer affects density (it still affects the bounding-box rescale).',
  },
]
