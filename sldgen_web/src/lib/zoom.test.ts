import { describe, expect, it } from 'vitest'
import {
  MAX_SCALE,
  MIN_SCALE,
  ZOOM_STEP,
  actualSizeView,
  clampScale,
  fitView,
  zoomAbout,
  zoomCentered,
} from './zoom'

describe('zoomAbout', () => {
  it('keeps the anchored pixel of the content under the anchor', () => {
    const view = { scale: 2, x: -40, y: 15 }
    const anchorX = 300
    const anchorY = 200
    const contentX = (anchorX - view.x) / view.scale
    const contentY = (anchorY - view.y) / view.scale

    const zoomed = zoomAbout(view, ZOOM_STEP, anchorX, anchorY)

    expect(zoomed.scale).toBeCloseTo(2.2, 6)
    expect(zoomed.x + contentX * zoomed.scale).toBeCloseTo(anchorX, 6)
    expect(zoomed.y + contentY * zoomed.scale).toBeCloseTo(anchorY, 6)
  })

  it('is reversible: one step in, one step out, back where you started', () => {
    const view = { scale: 1, x: 12, y: -7 }
    const round = zoomAbout(zoomAbout(view, ZOOM_STEP, 100, 80), 1 / ZOOM_STEP, 100, 80)
    expect(round.scale).toBeCloseTo(view.scale, 6)
    expect(round.x).toBeCloseTo(view.x, 6)
    expect(round.y).toBeCloseTo(view.y, 6)
  })

  it('clamps at both ends and leaves the offsets alone once clamped', () => {
    const zoomedIn = zoomAbout({ scale: MAX_SCALE, x: 5, y: 6 }, ZOOM_STEP, 100, 100)
    expect(zoomedIn).toEqual({ scale: MAX_SCALE, x: 5, y: 6 })

    const zoomedOut = zoomAbout({ scale: MIN_SCALE, x: 5, y: 6 }, 1 / ZOOM_STEP, 100, 100)
    expect(zoomedOut).toEqual({ scale: MIN_SCALE, x: 5, y: 6 })
  })

  it('never overshoots the range in a single step', () => {
    expect(clampScale(MAX_SCALE * ZOOM_STEP)).toBe(MAX_SCALE)
    expect(clampScale(MIN_SCALE / ZOOM_STEP)).toBe(MIN_SCALE)
  })
})

describe('zoomCentered', () => {
  it('holds the middle of the viewport still', () => {
    const viewport = { width: 800, height: 600 }
    const view = { scale: 1, x: 0, y: 0 }
    const contentX = (400 - view.x) / view.scale
    const contentY = (300 - view.y) / view.scale

    const zoomed = zoomCentered(view, ZOOM_STEP, viewport)

    expect(zoomed.x + contentX * zoomed.scale).toBeCloseTo(400, 6)
    expect(zoomed.y + contentY * zoomed.scale).toBeCloseTo(300, 6)
  })
})

describe('fitView', () => {
  it('fits a wide image by its width and centres it vertically', () => {
    const fitted = fitView({ width: 1000, height: 500 }, { width: 800, height: 600 })!
    expect(fitted.scale).toBeCloseTo(0.768, 6) // 800/1000 * 0.96
    expect(fitted.x).toBeCloseTo((800 - 1000 * fitted.scale) / 2, 6)
    expect(fitted.y).toBeCloseTo((600 - 500 * fitted.scale) / 2, 6)
    expect(1000 * fitted.scale).toBeLessThanOrEqual(800)
    expect(500 * fitted.scale).toBeLessThanOrEqual(600)
  })

  it('fits a tall image by its height', () => {
    const fitted = fitView({ width: 500, height: 1000 }, { width: 800, height: 600 })!
    expect(fitted.scale).toBeCloseTo(0.576, 6) // 600/1000 * 0.96
    expect(500 * fitted.scale).toBeLessThanOrEqual(800)
    expect(1000 * fitted.scale).toBeLessThanOrEqual(600)
  })

  it('enlarges content smaller than the viewport', () => {
    const fitted = fitView({ width: 100, height: 100 }, { width: 800, height: 600 })!
    expect(fitted.scale).toBeGreaterThan(1)
    expect(fitted.scale).toBeCloseTo(5.76, 6)
  })

  it('returns null rather than a nonsense view when nothing is measurable yet', () => {
    expect(fitView({ width: 0, height: 0 }, { width: 800, height: 600 })).toBeNull()
    expect(fitView({ width: 100, height: 100 }, { width: 0, height: 0 })).toBeNull()
    expect(fitView({ width: Number.NaN, height: 100 }, { width: 800, height: 600 })).toBeNull()
  })
})

describe('actualSizeView', () => {
  it('is scale 1, centred', () => {
    const view = actualSizeView({ width: 400, height: 200 }, { width: 800, height: 600 })
    expect(view).toEqual({ scale: 1, x: 200, y: 200 })
  })

  it('falls back to the origin when the content has not been measured', () => {
    expect(actualSizeView({ width: 0, height: 0 }, { width: 800, height: 600 })).toEqual({
      scale: 1,
      x: 0,
      y: 0,
    })
  })
})
