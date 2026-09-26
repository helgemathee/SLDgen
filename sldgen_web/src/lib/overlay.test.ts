import { describe, expect, it } from 'vitest'
import { canOverlay, overlayStyle, withViewBox } from './overlay'

describe('canOverlay', () => {
  it('offers every tab but input and landmarks, when there is an input', () => {
    for (const tab of ['result', 'preview', 'mask', 'condition', 'weight', 'edges'])
      expect(canOverlay(tab, true)).toBe(true)
    expect(canOverlay('input', true)).toBe(false)
    expect(canOverlay('landmarks', true)).toBe(false)
    expect(canOverlay('result', false)).toBe(false)
  })
})

describe('withViewBox', () => {
  const run = '<?xml version="1.0" ?>\n<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="512" height="512">\n<g/></svg>'

  it('adds a viewBox from width/height (the run writes none)', () => {
    const out = withViewBox(run)
    expect(out).toContain('<svg viewBox="0 0 512 512" preserveAspectRatio="none" xmlns=')
    expect(out.startsWith('<?xml')).toBe(true)
    expect(out.endsWith('<g/></svg>')).toBe(true)
  })

  it('leaves an SVG that has one, or no size, alone', () => {
    const has = '<svg viewBox="0 0 10 10" width="20" height="20"></svg>'
    expect(withViewBox(has)).toBe(has)
    const bare = '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
    expect(withViewBox(bare)).toBe(bare)
    expect(withViewBox('not svg')).toBe('not svg')
  })

  it('reads px units and single quotes', () => {
    expect(withViewBox("<svg width='300px' height='200px'></svg>")).toContain('viewBox="0 0 300 200"')
  })
})

describe('overlayStyle', () => {
  it('clamps opacity and passes the blend', () => {
    expect(overlayStyle(1.5, 'multiply')).toEqual({ opacity: 1, mixBlendMode: 'multiply' })
    expect(overlayStyle(-1, 'normal').opacity).toBe(0)
  })
})
