import { describe, expect, it } from 'vitest'
import {
  PROFILE_TEMPLATE,
  boxFrom,
  clampView,
  fitView,
  fromDetected,
  mergeDetected,
  nextName,
  panBy,
  parseFile,
  placedCount,
  placementHint,
  serialize,
  withTemplate,
  zoomAbout,
  zoomLevel,
  type EditorPoint,
} from './landmarks'

const SIZE: [number, number] = [512, 512]

function point(overrides: Partial<EditorPoint>): EditorPoint {
  return { id: 'x', name: 'p', xy: [0, 0], weight: 1, source: 'mesh', edited: false, ...overrides }
}

describe('mergeDetected', () => {
  const detected = [
    { name: 'left_eye_outer', xy: [10, 10] as [number, number], weight: 3, source: 'mesh' },
    { name: 'nose_tip', xy: [20, 20] as [number, number], weight: 3, source: 'silhouette' },
  ]

  it('adds everything to an empty table', () => {
    const report = mergeDetected([], detected)
    expect(report.added).toBe(2)
    expect(report.points.map((p) => p.name)).toEqual(['left_eye_outer', 'nose_tip'])
    expect(report.points[1].source).toBe('silhouette')
  })

  it('keeps a point the user moved, and does not add a duplicate', () => {
    const moved = point({ name: 'left_eye_outer', xy: [99, 99], edited: true })
    const report = mergeDetected([moved], detected)
    expect(report.kept).toBe(1)
    expect(report.points.filter((p) => p.name === 'left_eye_outer')).toEqual([moved])
    expect(report.added).toBe(1)
  })

  it('keeps manual points', () => {
    const manual = point({ name: 'p1', source: 'manual', xy: [5, 5] })
    const report = mergeDetected([manual], detected)
    expect(report.points[0]).toEqual(manual)
    expect(report.kept).toBe(1)
  })

  it('updates a detected point that was never edited', () => {
    const old = point({ name: 'nose_tip', xy: [1, 1], weight: 1 })
    const report = mergeDetected([old], detected)
    expect(report.updated).toBe(1)
    expect(report.points[0].xy).toEqual([20, 20])
    expect(report.points[0].weight).toBe(3)
    expect(report.points[0].id).toBe(old.id)
  })

  it('removes a detected point that is now culled', () => {
    const hidden = point({ name: 'right_eye_outer' })
    const report = mergeDetected([hidden], detected)
    expect(report.removed).toBe(1)
    expect(report.points.some((p) => p.name === 'right_eye_outer')).toBe(false)
  })

  it('places an unplaced template row at its detected position', () => {
    const row = point({ name: 'nose_tip', xy: null, source: 'manual', weight: 2 })
    const report = mergeDetected([row], detected)
    expect(report.points.find((p) => p.name === 'nose_tip')?.xy).toEqual([20, 20])
    expect(report.points.find((p) => p.name === 'nose_tip')?.weight).toBe(2)
    expect(report.points.filter((p) => p.name === 'nose_tip')).toHaveLength(1)
  })
})

describe('the file', () => {
  it('round-trips, leaving out unplaced rows', () => {
    const points = [
      point({ name: 'a', xy: [1.234, 5.678], weight: 2, source: 'manual', edited: true }),
      point({ name: 'b', xy: null }),
    ]
    const file = serialize(points, SIZE)
    expect(file.space).toBe('canvas')
    expect(file.image_size).toEqual(SIZE)
    expect(file.landmarks).toEqual([
      { name: 'a', xy: [1.23, 5.68], weight: 2, source: 'manual', edited: true },
    ])
    const back = parseFile(JSON.parse(JSON.stringify(file)))
    if (typeof back === 'string') throw new Error(back)
    expect(back.points.map(({ id: _id, ...rest }) => rest)).toEqual([
      { name: 'a', xy: [1.23, 5.68], weight: 2, source: 'manual', edited: true },
    ])
  })

  it('reads a Spec 6 file without the new fields', () => {
    const back = parseFile({
      space: 'canvas',
      image_size: [512, 512],
      landmarks: [{ name: 'chin', xy: [3, 4], weight: 0.5 }],
    })
    if (typeof back === 'string') throw new Error(back)
    expect(back.points[0]).toMatchObject({ name: 'chin', source: 'mesh', edited: false })
  })

  it('refuses a file not in canvas space', () => {
    expect(typeof parseFile({ space: 'image', image_size: [1, 1] })).toBe('string')
  })

  it('counts only placed points with weight', () => {
    expect(placedCount([point({}), point({ xy: null }), point({ weight: 0 })])).toBe(1)
  })
})

describe('template and names', () => {
  it('adds only the profile names not present', () => {
    const rows = withTemplate([point({ name: 'nose_tip' })])
    expect(rows).toHaveLength(PROFILE_TEMPLATE.length)
    expect(rows.filter((p) => p.name === 'nose_tip')).toHaveLength(1)
    expect(rows.slice(1).every((p) => p.xy === null && p.source === 'manual')).toBe(true)
  })

  it('picks the first free manual name', () => {
    expect(nextName([point({ name: 'p1' }), point({ name: 'p3' })])).toBe('p2')
  })

  it('gives detected points fresh distinct ids', () => {
    const points = fromDetected([
      { name: 'a', xy: [0, 0], weight: 1 },
      { name: 'b', xy: [0, 0], weight: 1 },
    ])
    expect(points[0].id).not.toBe(points[1].id)
  })
})

describe('the view box', () => {
  it('zooms about the cursor, keeping that point fixed on screen', () => {
    const view = zoomAbout(fitView(SIZE), 2, [128, 256], SIZE)
    expect(zoomLevel(view, SIZE)).toBe(2)
    expect((128 - view.x) / view.w).toBeCloseTo(128 / 512)
    expect((256 - view.y) / view.h).toBeCloseTo(256 / 512)
  })

  it('never zooms out past the canvas or in past 16x', () => {
    expect(zoomAbout(fitView(SIZE), 0.5, [0, 0], SIZE)).toEqual(fitView(SIZE))
    expect(zoomLevel(zoomAbout(fitView(SIZE), 100, [256, 256], SIZE), SIZE)).toBe(16)
  })

  it('pans but stays inside the canvas', () => {
    const zoomed = clampView({ x: 100, y: 100, w: 128, h: 128 }, SIZE)
    expect(panBy(zoomed, 50, 0, SIZE).x).toBe(50)
    expect(panBy(zoomed, 1000, 0, SIZE).x).toBe(0)
    expect(panBy(zoomed, -1000, 0, SIZE).x).toBe(512 - 128)
  })

  it('normalises and clips a dragged box', () => {
    expect(boxFrom([300, 40], [-10, 600], SIZE)).toEqual([0, 40, 300, 512])
  })
})

describe('placement hints', () => {
  it('explains every template name', () => {
    for (const entry of PROFILE_TEMPLATE) expect(placementHint(entry.name)).not.toMatch(/^Your own/)
  })

  it('reads sided detection names both ways round', () => {
    expect(placementHint('left_eye_outer')).toMatch(/outer corner.*subject’s own left side.*image’s right/)
    expect(placementHint('mouth_right')).toMatch(/corner of the mouth.*own right/)
  })

  it('falls back for the user’s own points', () => {
    expect(placementHint('p3')).toMatch(/^Your own point/)
  })
})
