import { describe, expect, it } from 'vitest'
import {
  OPERATIONAL_NAMES,
  PARAM_SPECS,
  SPEC_BY_NAME,
  coerceParam,
  defaultParams,
  formatParamValue,
  validateParams,
  withoutInputPaths,
} from './params'
import { diffParams, distinguishingParams, sameRun, sameValue } from './paramdiff'

describe('the parameter schema', () => {
  it('has no duplicate names', () => {
    const names = PARAM_SPECS.map((spec) => spec.name)
    expect(new Set(names).size).toBe(names.length)
  })

  it('marks exactly the four operational settings as operational', () => {
    expect([...OPERATIONAL_NAMES].sort()).toEqual([
      'checkpoint_interval',
      'debug',
      'save_interval',
      'verbose',
    ])
  })

  it('marks exactly the nullable flags as optional', () => {
    expect(PARAM_SPECS.filter((spec) => spec.optional).map((spec) => spec.name).sort()).toEqual([
      'attract',
      'avoid',
      'init_points',
      'origin',
      'stipple_weight',
    ])
  })

  it('defaults checkpoint_interval to the service value, not SLDgen\'s zero', () => {
    // Spec 2 SS8: a crash should never cost more than 200 iterations.
    expect(SPEC_BY_NAME.checkpoint_interval.default).toBe(200)
  })

  it('gives every optional parameter a null default', () => {
    for (const spec of PARAM_SPECS.filter((entry) => entry.optional)) {
      expect(spec.default).toBeNull()
    }
  })
})

describe('defaultParams', () => {
  it('is canonical -- every known name is present', () => {
    const params = defaultParams()
    expect(Object.keys(params).sort()).toEqual(PARAM_SPECS.map((spec) => spec.name).sort())
  })
})

describe('withoutInputPaths', () => {
  it('drops the four the service refuses to accept directly', () => {
    const stripped = withoutInputPaths({
      ...defaultParams(),
      avoid: ['x.svg'],
      attract: ['y.svg'],
      init_points: 'z.svg',
      stipple_weight: 'w.png',
      seed: 4,
    })
    expect(stripped.avoid).toBeUndefined()
    expect(stripped.attract).toBeUndefined()
    expect(stripped.init_points).toBeUndefined()
    expect(stripped.stipple_weight).toBeUndefined()
    expect(stripped.seed).toBe(4)
  })
})

describe('coerceParam', () => {
  it('parses numbers and falls back to the default on nonsense', () => {
    expect(coerceParam(SPEC_BY_NAME.seed, '42')).toBe(42)
    expect(coerceParam(SPEC_BY_NAME.seed, 'not a number')).toBe(0)
    expect(coerceParam(SPEC_BY_NAME.conditioning_scale, '0.75')).toBe(0.75)
  })

  it('keeps width numeric when it is a number and textual when it is a mode', () => {
    expect(coerceParam(SPEC_BY_NAME.width, '1.5')).toBe(1.5)
    expect(coerceParam(SPEC_BY_NAME.width, 'optim')).toBe('optim')
  })

  it('treats a trailing-dot entry as text rather than silently rounding it', () => {
    // "1." parses as 1 but is not what the user typed yet; round-tripping it as
    // a number would move the cursor while they are still typing "1.5".
    expect(coerceParam(SPEC_BY_NAME.width, '1.')).toBe('1.')
  })

  it('coerces flags to booleans', () => {
    expect(coerceParam(SPEC_BY_NAME.calligraphy, true)).toBe(true)
    expect(coerceParam(SPEC_BY_NAME.optimize_cp_weights, false)).toBe(false)
  })

  it('coerces an origin pair to numbers', () => {
    expect(coerceParam(SPEC_BY_NAME.origin, [0.25, 0.75])).toEqual([0.25, 0.75])
  })
})

describe('validateParams', () => {
  it('accepts the defaults', () => {
    expect(validateParams(defaultParams())).toEqual([])
  })

  it('mirrors the server rule that origin and fixed endpoints are exclusive', () => {
    const problems = validateParams({
      ...defaultParams(),
      origin: [0.5, 0.5],
      fixed_endpoints: true,
    })
    expect(problems.some((text) => text.includes('fixed endpoints'))).toBe(true)
  })

  it('requires tsp for origin, init points and stipple weight', () => {
    expect(
      validateParams({ ...defaultParams(), origin: [0.5, 0.5], init_method: 'contour' }),
    ).toContainEqual(expect.stringContaining("init method 'tsp'"))
    expect(
      validateParams({ ...defaultParams(), init_points: 'a.svg', init_method: 'trefoil' }),
    ).toContainEqual(expect.stringContaining("init method 'tsp'"))
  })

  it('rejects origin coordinates outside the unit square', () => {
    expect(validateParams({ ...defaultParams(), origin: [1.4, 0.2] })).toContainEqual(
      expect.stringContaining('between 0 and 1'),
    )
  })

  it('rejects the enumerations the server rejects', () => {
    expect(validateParams({ ...defaultParams(), condition: 'sobel' })).toHaveLength(1)
    expect(validateParams({ ...defaultParams(), stipple_weight_mode: 'add' })).toHaveLength(1)
    expect(validateParams({ ...defaultParams(), width: 'wobbly' })).toHaveLength(1)
  })

  it('rejects non-positive counts', () => {
    expect(validateParams({ ...defaultParams(), num_iter: 0 })).toHaveLength(1)
    expect(validateParams({ ...defaultParams(), save_interval: 0 })).toHaveLength(1)
    expect(validateParams({ ...defaultParams(), checkpoint_interval: -1 })).toHaveLength(1)
  })

  it('allows checkpoint_interval zero, which means "never"', () => {
    expect(validateParams({ ...defaultParams(), checkpoint_interval: 0 })).toEqual([])
  })
})

describe('formatParamValue', () => {
  it('renders an omitted flag as a dash and an auto caption as (auto)', () => {
    expect(formatParamValue(null)).toBe('—')
    expect(formatParamValue('')).toBe('(auto)')
    expect(formatParamValue([])).toBe('—')
  })

  it('renders booleans as words and lists as a joined string', () => {
    expect(formatParamValue(true)).toBe('yes')
    expect(formatParamValue(false)).toBe('no')
    expect(formatParamValue(['a.svg', 'b.svg'])).toBe('a.svg, b.svg')
  })
})

describe('sameValue', () => {
  it('compares arrays element-wise', () => {
    expect(sameValue([0.1, 0.2], [0.1, 0.2])).toBe(true)
    expect(sameValue([0.1, 0.2], [0.1, 0.3])).toBe(false)
    expect(sameValue([0.1], [0.1, 0.2])).toBe(false)
  })

  it('does not treat an array and a scalar as equal', () => {
    expect(sameValue([1], 1)).toBe(false)
  })

  it('treats 1 and 1.0 as the same number', () => {
    expect(sameValue(1, 1.0)).toBe(true)
  })
})

describe('diffParams', () => {
  it('lists changed names in the canonical order', () => {
    const before = defaultParams()
    const after = { ...before, seed: 5, caption: 'a car' }
    expect(diffParams(before, after)).toEqual(['caption', 'seed'])
  })

  it('is empty for identical sets', () => {
    expect(diffParams(defaultParams(), defaultParams())).toEqual([])
  })
})

describe('distinguishingParams', () => {
  it('returns nothing for fewer than two sets', () => {
    expect(distinguishingParams([defaultParams()])).toEqual([])
  })

  it('returns only the fields that actually tell the cells apart', () => {
    const base = defaultParams()
    const sets = [
      { ...base, seed: 1 },
      { ...base, seed: 2 },
      { ...base, seed: 3 },
    ]
    expect(distinguishingParams(sets)).toEqual(['seed'])
  })

  it('catches a field that differs in only one of several sets', () => {
    const base = defaultParams()
    const sets = [base, base, { ...base, caption: 'different' }]
    expect(distinguishingParams(sets)).toEqual(['caption'])
  })
})

describe('sameRun', () => {
  it('is true for identical structural parameters', () => {
    expect(sameRun(defaultParams(), defaultParams())).toBe(true)
  })

  it('is false when a structural parameter differs', () => {
    expect(sameRun(defaultParams(), { ...defaultParams(), lr: 0.4 })).toBe(false)
  })
})
