import { describe, expect, it } from 'vitest'
import {
  MAX_SCALE,
  MIN_SCALE,
  ZOOM_STEP,
  actualSizeView,
  clampScale,
  fitView,
  rescaleView,
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

describe('rescaleView', () => {
  const viewport = { width: 800, height: 600 }
  // The case this exists for: a run artefact at render_size, and a weight map
  // at the full resolution of the uploaded photo.
  const run = { width: 1024, height: 1024 }
  const weight = { width: 4032, height: 4032 }

  it('keeps a fitted view fitted when the artefact is four times larger', () => {
    const fitted = fitView(run, viewport)!
    const carried = rescaleView(fitted, run, weight, viewport)
    const expected = fitView(weight, viewport)!
    expect(carried.scale).toBeCloseTo(expected.scale, 6)
    expect(carried.x).toBeCloseTo(expected.x, 6)
    expect(carried.y).toBeCloseTo(expected.y, 6)
  })

  it('keeps the zoom relative to fit, so twice-fit stays twice-fit', () => {
    const fitted = fitView(run, viewport)!
    const zoomed = { ...fitted, scale: fitted.scale * 2 }
    const carried = rescaleView(zoomed, run, weight, viewport)
    expect(carried.scale / fitView(weight, viewport)!.scale).toBeCloseTo(2, 6)
  })

  it('holds the same point of the picture under the middle of the viewport', () => {
    // Zoomed in on a point a quarter across and a third down.
    const scale = 1.5
    const u = 0.25
    const v = 1 / 3
    const view = {
      scale,
      x: viewport.width / 2 - u * scale * run.width,
      y: viewport.height / 2 - v * scale * run.height,
    }
    const carried = rescaleView(view, run, weight, viewport)
    expect((viewport.width / 2 - carried.x) / (carried.scale * weight.width)).toBeCloseTo(u, 6)
    expect((viewport.height / 2 - carried.y) / (carried.scale * weight.height)).toBeCloseTo(v, 6)
  })

  it('handles a change of aspect ratio, since the fit swaps axes', () => {
    const wide = { width: 1600, height: 900 }
    const tall = { width: 900, height: 1600 }
    const carried = rescaleView(fitView(wide, viewport)!, wide, tall, viewport)
    const expected = fitView(tall, viewport)!
    expect(carried.scale).toBeCloseTo(expected.scale, 6)
    expect(carried.x).toBeCloseTo(expected.x, 6)
    expect(carried.y).toBeCloseTo(expected.y, 6)
  })

  it('leaves the view alone when nothing is measurable', () => {
    const view = { scale: 2, x: 10, y: 20 }
    expect(rescaleView(view, { width: 0, height: 0 }, run, viewport)).toEqual(view)
    expect(rescaleView(view, run, { width: 0, height: 0 }, viewport)).toEqual(view)
    expect(rescaleView({ ...view, scale: 0 }, run, weight, viewport)).toEqual({ ...view, scale: 0 })
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
