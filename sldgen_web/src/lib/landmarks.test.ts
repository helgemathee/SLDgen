import { describe, expect, it } from 'vitest'
import {
  LANDMARK_SETS,
  PROFILE_TEMPLATE,
  describePose,
  glassesTemplate,
  insertVertex,
  lineCount,
  lineHint,
  mergeLines,
  newLine,
  savedLabel,
  type EditorLine,
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
  referenceFor,
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

describe('references', () => {
  it('links every template name somewhere, and not the user’s own points', () => {
    for (const entry of PROFILE_TEMPLATE) expect(referenceFor(entry.name)?.url).toMatch(/^https:\/\//)
    expect(referenceFor('p2')).toBeNull()
  })

  it('sends eye corners to the canthus page and detected points to the mesh map', () => {
    expect(referenceFor('left_eye_outer')?.url).toMatch(/Canthus/)
    expect(referenceFor('mouth_right')?.url).toMatch(/canonical_face_model/)
  })
})

function line(overrides: Partial<EditorLine>): EditorLine {
  return {
    id: 'l',
    name: 'line1',
    xy: [
      [0, 0],
      [10, 0],
    ],
    closed: false,
    weight: 1,
    source: 'manual',
    edited: true,
    ...overrides,
  }
}

describe('polylines in the file (Spec 7 addendum SS5)', () => {
  it('round-trips lines and the pose report', () => {
    const rim = line({ name: 'rim', closed: true, weight: 3, xy: [[1, 1], [9, 1], [5, 8.123]] })
    const pose = { yaw: -18.4, pitch: 6.1, roll: -3.2, method: 'rigid-fit' as const, residual_px: 2 }
    const file = serialize([point({ name: 'a', xy: [3, 4] })], SIZE, { lines: [rim], pose })
    expect(file.polylines).toHaveLength(1)
    expect(file.polylines![0].xy[2]).toEqual([5, 8.12])
    expect(file.pose).toEqual(pose)
    const back = parseFile(JSON.parse(JSON.stringify(file)))
    if (typeof back === 'string') throw new Error(back)
    expect(back.lines).toHaveLength(1)
    expect(back.lines[0]).toMatchObject({ name: 'rim', closed: true, weight: 3, source: 'manual' })
    expect(back.pose).toEqual(pose)
  })

  it('leaves out lines with too few vertices, and the key when none remain', () => {
    const file = serialize([point({ name: 'a' })], SIZE, {
      lines: [line({ xy: [[1, 1]] }), line({ closed: true })],
    })
    expect('polylines' in file).toBe(false)
    expect('pose' in file).toBe(false)
  })

  it('reads detector files with polylines and skips broken ones', () => {
    const back = parseFile({
      space: 'canvas',
      image_size: [512, 512],
      landmarks: [],
      polylines: [
        { name: 'hairline', closed: false, weight: 1.5, source: 'hairline', xy: [[1, 2], [3, 4]] },
        { name: 'bad', xy: [[1, 2]] },
      ],
    })
    if (typeof back === 'string') throw new Error(back)
    expect(back.lines.map((l) => l.name)).toEqual(['hairline'])
    expect(back.lines[0].edited).toBe(false)
  })

  it('counts lines in the label', () => {
    expect(savedLabel([point({})], [])).toBe('1 landmarks (edited)')
    expect(savedLabel([point({})], [line({}), line({ weight: 0 }), line({ xy: [] })])).toBe(
      '1 landmarks + 1 line (edited)',
    )
    expect(lineCount([line({}), line({ name: 'b' })])).toBe(2)
  })
})

describe('mergeLines', () => {
  const detected = [{ name: 'hairline', xy: [[0, 0], [5, 5]] as [number, number][], closed: false, weight: 1.5, source: 'hairline' }]

  it('adds, updates unedited, keeps manual and edited, removes vanished', () => {
    const fresh = mergeLines([], detected)
    expect(fresh.added).toBe(1)
    expect(fresh.lines[0].source).toBe('hairline')
    const moved = mergeLines(
      [{ ...fresh.lines[0], xy: [[9, 9], [8, 8]] }],
      [{ ...detected[0], xy: [[1, 1], [2, 2]] }],
    )
    expect(moved.updated).toBe(1)
    expect(moved.lines[0].xy).toEqual([[1, 1], [2, 2]])
    const edited = mergeLines([{ ...fresh.lines[0], edited: true, xy: [[9, 9], [8, 8]] }], detected)
    expect(edited.kept).toBe(1)
    expect(edited.lines[0].xy).toEqual([[9, 9], [8, 8]])
    const rim = line({ name: 'glasses_left_rim' })
    const gone = mergeLines([fresh.lines[0], rim], [])
    expect(gone.removed).toBe(1)
    expect(gone.lines).toEqual([rim])
  })
})

describe('insertVertex', () => {
  const open: [number, number][] = [[0, 0], [10, 0], [20, 0]]

  it('appends while the line is short', () => {
    expect(insertVertex([], false, [1, 1])).toEqual({ xy: [[1, 1]], index: 0 })
    expect(insertVertex([[0, 0]], false, [5, 5]).index).toBe(1)
  })

  it('inserts on the nearest segment', () => {
    expect(insertVertex(open, false, [14, 2])).toEqual({ xy: [[0, 0], [10, 0], [14, 2], [20, 0]], index: 2 })
  })

  it('grows an open line at the end the click is beyond', () => {
    expect(insertVertex(open, false, [25, 1]).index).toBe(3)
    expect(insertVertex(open, false, [-5, 1]).index).toBe(0)
  })

  it('uses the closing segment of a closed line', () => {
    const square: [number, number][] = [[0, 0], [10, 0], [10, 10], [0, 10]]
    expect(insertVertex(square, true, [-1, 5])).toEqual({
      xy: [[0, 0], [10, 0], [10, 10], [0, 10], [-1, 5]],
      index: 4,
    })
  })
})

describe('glassesTemplate', () => {
  const eyes = [
    point({ name: 'right_eye_outer', xy: [180, 200] }),
    point({ name: 'right_eye_inner', xy: [220, 200] }),
    point({ name: 'left_eye_inner', xy: [290, 200] }),
    point({ name: 'left_eye_outer', xy: [330, 200] }),
  ]

  it('rings each eye, the right rim on the image left', () => {
    const lines = glassesTemplate(eyes, [], SIZE)
    expect(lines.map((l) => l.name)).toEqual(['glasses_right_rim', 'glasses_left_rim', 'glasses_bridge'])
    const [right, left, bridge] = lines
    expect(right.closed && left.closed && !bridge.closed).toBe(true)
    expect(right.xy).toHaveLength(12)
    const cx = (l: EditorLine) => l.xy.reduce((sum, p) => sum + p[0], 0) / l.xy.length
    expect(cx(right)).toBeCloseTo(200, 5)
    expect(cx(left)).toBeCloseTo(310, 5)
    expect(right.weight).toBe(3)
    expect(bridge.xy[0][0]).toBeLessThan(bridge.xy[2][0])
  })

  it('skips names already present and works without eyes', () => {
    const existing = line({ name: 'glasses_left_rim' })
    const lines = glassesTemplate([], [existing], SIZE)
    expect(lines.filter((l) => l.name === 'glasses_left_rim')).toEqual([existing])
    expect(lines).toHaveLength(3)
    const rim = lines.find((l) => l.name === 'glasses_right_rim')!
    expect(rim.xy.every(([x, y]) => x > 0 && x < 512 && y > 0 && y < 512)).toBe(true)
  })

  it('mirrors a missing eye from the one that is there', () => {
    const lines = glassesTemplate(eyes.slice(0, 2), [], SIZE)
    const left = lines.find((l) => l.name === 'glasses_left_rim')!
    expect(left.xy.reduce((sum, p) => sum + p[0], 0) / 12).toBeGreaterThan(250)
  })
})

describe('sets, lines, pose text', () => {
  it('offers the four sets, sparse first', () => {
    expect(LANDMARK_SETS.map((s) => s.value)).toEqual(['sparse', 'standard', 'dense', 'pose-locked'])
  })

  it('names new lines line1, line2', () => {
    const first = newLine([])
    expect(first.name).toBe('line1')
    expect(newLine([first]).name).toBe('line2')
    expect(first.xy).toEqual([])
  })

  it('describes the pose', () => {
    expect(describePose({ yaw: -18.4, pitch: 6.1, roll: -3.2, method: 'rigid-fit' })).toBe(
      'yaw -18°, pitch 6°, roll -3°',
    )
    expect(describePose({ yaw: -78.6, pitch: null, roll: null, method: 'pose' })).toBe('yaw -79° (coarse)')
    expect(describePose(null)).toBe('')
  })

  it('hints for mesh points, standard names and lines', () => {
    expect(placementHint('m123')).toContain('Face-mesh point 123')
    expect(referenceFor('m123')).not.toBeNull()
    expect(placementHint('right_lip_peak')).toContain('cupid')
    expect(placementHint('left_jaw_low')).toContain('jawline')
    expect(lineHint('hairline')).toContain('forehead')
    expect(lineHint('whatever')).toContain('Your own line')
  })
})
