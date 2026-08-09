import type { Params, ParamValue } from '../api/types'
import { PARAM_SPECS, SPEC_BY_NAME } from './params'

/** Deep-ish equality, enough for the value kinds a parameter can hold. */
export function sameValue(a: ParamValue, b: ParamValue): boolean {
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((value, index) => value === b[index])
  }
  if (Array.isArray(a) || Array.isArray(b)) return false
  // 1 and 1.0 are the same number to SLDgen, and JSON round-tripping loses the
  // distinction anyway, so numeric comparison must not be string comparison.
  if (typeof a === 'number' && typeof b === 'number') return a === b
  return a === b
}

/** Parameter names whose values differ, in the canonical parameter order. */
export function diffParams(before: Params, after: Params): string[] {
  return PARAM_SPECS.map((spec) => spec.name).filter(
    (name) => !sameValue(before[name] ?? null, after[name] ?? null),
  )
}

/**
 * The fields that distinguish a set of jobs from each other (Spec 3 SS7).
 *
 * Compare shows only these, so a seed batch is captioned purely by seed number
 * and a mixed batch by seed and caption -- whatever actually tells the cells
 * apart, and nothing that does not.
 */
export function distinguishingParams(sets: Params[]): string[] {
  if (sets.length < 2) return []
  return PARAM_SPECS.map((spec) => spec.name).filter((name) => {
    const first = sets[0][name] ?? null
    return sets.some((set) => !sameValue(set[name] ?? null, first))
  })
}

/**
 * Whether two jobs would produce the same result.
 *
 * Operational settings are excluded because they never affect the result
 * (Spec 2 SS4.2), and the input-backed paths are excluded because every job
 * gets its own copy under its own id -- two jobs with identical inputs have
 * different paths recorded and are nonetheless the same run.
 */
const IDENTITY_EXCLUDED = new Set([
  ...PARAM_SPECS.filter((spec) => spec.group === 'operational').map((spec) => spec.name),
  'avoid',
  'attract',
  'init_points',
  'stipple_weight',
])

export function sameRun(a: Params, b: Params): boolean {
  return PARAM_SPECS.every(
    (spec) =>
      IDENTITY_EXCLUDED.has(spec.name) || sameValue(a[spec.name] ?? null, b[spec.name] ?? null),
  )
}

/** Human label for a parameter, falling back to the raw name for unknown ones. */
export function paramLabel(name: string): string {
  return SPEC_BY_NAME[name]?.label ?? name
}
