import { describe, expect, it } from 'vitest'
import {
  brushStamp,
  buildLabCache,
  combineMasks,
  countMask,
  deltaE,
  featherMask,
  invertMask,
  magicWand,
  maskOutline,
  srgbToLab,
} from './lab'

/** Build a tiny RGBA image from a [r,g,b] per pixel grid. */
function image(pixels: [number, number, number][]): Uint8ClampedArray {
  const data = new Uint8ClampedArray(pixels.length * 4)
  pixels.forEach(([r, g, b], index) => {
    data[index * 4] = r
    data[index * 4 + 1] = g
    data[index * 4 + 2] = b
    data[index * 4 + 3] = 255
  })
  return data
}

const BLACK: [number, number, number] = [0, 0, 0]
const WHITE: [number, number, number] = [255, 255, 255]
const RED: [number, number, number] = [255, 0, 0]

describe('srgbToLab', () => {
  it('places the reference colours where CIE says they are', () => {
    const [blackL] = srgbToLab(0, 0, 0)
    const [whiteL, whiteA, whiteB] = srgbToLab(255, 255, 255)
    expect(blackL).toBeCloseTo(0, 4)
    expect(whiteL).toBeCloseTo(100, 3)
    expect(whiteA).toBeCloseTo(0, 3)
    expect(whiteB).toBeCloseTo(0, 3)

    // sRGB red: L*≈53.24, a*≈80.09, b*≈67.20 (D65).
    const [l, a, b] = srgbToLab(255, 0, 0)
    expect(l).toBeCloseTo(53.24, 1)
    expect(a).toBeCloseTo(80.09, 1)
    expect(b).toBeCloseTo(67.2, 1)
  })

  it('separates colours RGB Euclidean distance calls identical', () => {
    // Both steps are a distance of exactly 60 in RGB, so an RGB tolerance
    // treats them the same. Perceptually they are nothing alike -- a 60-step in
    // green is far more visible than the same step in red -- and Lab says so,
    // by about 40%. This is why an RGB wand produces visibly wrong selections
    // on skies and skin, and why the tolerance here is in Lab units.
    const greenStep = distance(srgbToLab(0, 0, 0), srgbToLab(0, 60, 0))
    const redStep = distance(srgbToLab(0, 0, 0), srgbToLab(60, 0, 0))
    expect(greenStep).toBeGreaterThan(redStep * 1.3)
  })
})

function distance(a: [number, number, number], b: [number, number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
}

describe('deltaE', () => {
  it('is zero for a pixel against itself and positive otherwise', () => {
    const lab = buildLabCache(image([BLACK, WHITE]))
    expect(deltaE(lab, 0, 0)).toBe(0)
    expect(deltaE(lab, 0, 1)).toBeCloseTo(100, 3)
  })
})

describe('magicWand', () => {
  //  W W R
  //  W R R
  //  W W W      (3x3)
  const grid: [number, number, number][] = [WHITE, WHITE, RED, WHITE, RED, RED, WHITE, WHITE, WHITE]
  const lab = buildLabCache(image(grid))
  const base = { lab, width: 3, height: 3, tolerance: 10 }

  it('flood-fills only the connected region', () => {
    const mask = magicWand({ ...base, x: 2, y: 0, contiguous: true })
    expect(Array.from(mask)).toEqual([0, 0, 1, 0, 1, 1, 0, 0, 0])
  })

  it('selects every matching pixel when contiguous is off', () => {
    //  R W R  -- two red regions with no path between them
    const split: [number, number, number][] = [RED, WHITE, RED]
    const splitLab = buildLabCache(image(split))
    const contiguous = magicWand({
      lab: splitLab,
      width: 3,
      height: 1,
      x: 0,
      y: 0,
      tolerance: 10,
      contiguous: true,
    })
    const global = magicWand({
      lab: splitLab,
      width: 3,
      height: 1,
      x: 0,
      y: 0,
      tolerance: 10,
      contiguous: false,
    })
    expect(Array.from(contiguous)).toEqual([1, 0, 0])
    expect(Array.from(global)).toEqual([1, 0, 1])
  })

  it('widens the selection as tolerance rises', () => {
    const tight = countMask(magicWand({ ...base, x: 0, y: 0, tolerance: 5, contiguous: true }))
    // White to red is a Lab distance of about 114.5, so 100 is still short of
    // reaching the red -- a useful reminder that the slider is in Lab units and
    // its top is not "everything".
    const nearlyAll = countMask(
      magicWand({ ...base, x: 0, y: 0, tolerance: 100, contiguous: true }),
    )
    const loose = countMask(magicWand({ ...base, x: 0, y: 0, tolerance: 120, contiguous: true }))
    expect(tight).toBe(6) // the white L-shape
    expect(nearlyAll).toBe(6)
    expect(loose).toBe(9) // everything
  })

  it('returns an empty mask for a click outside the image', () => {
    expect(countMask(magicWand({ ...base, x: 9, y: 9, contiguous: true }))).toBe(0)
  })

  it('does not blow the stack on a large flat region', () => {
    const size = 300
    const flat = buildLabCache(
      image(Array.from({ length: size * size }, () => WHITE as [number, number, number])),
    )
    const mask = magicWand({
      lab: flat,
      width: size,
      height: size,
      x: 0,
      y: 0,
      tolerance: 1,
      contiguous: true,
    })
    expect(countMask(mask)).toBe(size * size)
  })
})

describe('combineMasks', () => {
  const base = Uint8Array.from([1, 1, 0, 0])
  const addition = Uint8Array.from([0, 1, 1, 0])

  it('replaces, adds and subtracts', () => {
    expect(Array.from(combineMasks(base, addition, 'replace'))).toEqual([0, 1, 1, 0])
    expect(Array.from(combineMasks(base, addition, 'add'))).toEqual([1, 1, 1, 0])
    expect(Array.from(combineMasks(base, addition, 'subtract'))).toEqual([1, 0, 0, 0])
  })

  it('never mutates its inputs', () => {
    combineMasks(base, addition, 'add')
    expect(Array.from(base)).toEqual([1, 1, 0, 0])
  })
})

describe('invertMask', () => {
  it('flips every pixel', () => {
    expect(Array.from(invertMask(Uint8Array.from([1, 0, 1])))).toEqual([0, 1, 0])
  })
})

describe('featherMask', () => {
  // Big enough that a 9px feather cannot reach the middle from both sides; a
  // feather wider than the selection legitimately softens all of it.
  const width = 61
  const height = 61
  const mask = new Uint8Array(width * height)
  for (let y = 15; y < 46; y += 1) for (let x = 15; x < 46; x += 1) mask[y * width + x] = 1

  it('is a plain binary alpha at radius zero -- feather is not tolerance', () => {
    const alpha = featherMask(mask, width, height, 0)
    expect(new Set(alpha)).toEqual(new Set([0, 255]))
  })

  it('softens the edge while leaving the interior solid', () => {
    const alpha = featherMask(mask, width, height, 9)
    const centre = alpha[30 * width + 30]
    const justOutside = alpha[30 * width + 48]
    const farOutside = alpha[30 * width + 58]
    expect(centre).toBe(255)
    expect(justOutside).toBeGreaterThan(0)
    expect(justOutside).toBeLessThan(centre)
    expect(farOutside).toBe(0)
  })

  it('produces a monotonic falloff across the boundary', () => {
    const alpha = featherMask(mask, width, height, 9)
    const row = Array.from({ length: 16 }, (_unused, index) => alpha[30 * width + 40 + index])
    for (let index = 1; index < row.length; index += 1) {
      expect(row[index]).toBeLessThanOrEqual(row[index - 1])
    }
  })
})

describe('maskOutline', () => {
  it('marks only the boundary of a filled square', () => {
    const width = 5
    const height = 5
    const mask = new Uint8Array(width * height)
    for (let y = 1; y < 4; y += 1) for (let x = 1; x < 4; x += 1) mask[y * width + x] = 1
    const outline = maskOutline(mask, width, height)
    expect(outline[2 * width + 2]).toBe(0) // the centre is interior
    expect(outline[1 * width + 1]).toBe(1) // a corner is boundary
    expect(countMask(outline)).toBe(8) // 3x3 minus its centre
  })

  it('treats the image edge as a boundary', () => {
    const mask = Uint8Array.from([1, 1, 1, 1])
    expect(countMask(maskOutline(mask, 2, 2))).toBe(4)
  })
})

describe('brushStamp', () => {
  it('covers a disc and nothing outside it', () => {
    const stamp = brushStamp({ width: 21, height: 21, x: 10, y: 10, radius: 5, hardness: 1 })
    expect(stamp.every(({ coverage }) => coverage > 0 && coverage <= 1)).toBe(true)
    const indices = new Set(stamp.map(({ index }) => index))
    expect(indices.has(10 * 21 + 10)).toBe(true)
    expect(indices.has(10 * 21 + 20)).toBe(false)
  })

  it('falls off with distance when soft, and does not when hard', () => {
    const soft = brushStamp({ width: 21, height: 21, x: 10, y: 10, radius: 8, hardness: 0 })
    const edge = soft.find(({ index }) => index === 10 * 21 + 17)!
    const centre = soft.find(({ index }) => index === 10 * 21 + 10)!
    expect(centre.coverage).toBeCloseTo(1, 5)
    expect(edge.coverage).toBeLessThan(0.3)

    const hard = brushStamp({ width: 21, height: 21, x: 10, y: 10, radius: 8, hardness: 1 })
    expect(hard.find(({ index }) => index === 10 * 21 + 17)!.coverage).toBeCloseTo(1, 2)
  })

  it('clips at the image border rather than wrapping', () => {
    const stamp = brushStamp({ width: 10, height: 10, x: 0, y: 0, radius: 4, hardness: 1 })
    expect(stamp.every(({ index }) => index % 10 <= 4)).toBe(true)
    expect(stamp.every(({ index }) => index >= 0 && index < 100)).toBe(true)
  })
})
