/**
 * Selection maths for the prep canvas (Spec 3 SS8.2).
 *
 * Tolerance is evaluated in CIE Lab, not RGB. RGB Euclidean distance produces
 * visibly wrong selections on skies and skin -- it treats a step in dark blue as
 * the same size as the identical step in bright blue, which is exactly the
 * busy-background-behind-a-car case this tool exists for. Lab is perceptually
 * near-uniform, so one tolerance number means roughly the same thing everywhere
 * in the image.
 *
 * Tolerance and feather are kept strictly separate. Tolerance decides what is
 * selected; feather decides how the edge falls off, applied to the selection's
 * alpha *after* the binary selection is computed. Conflating them is the usual
 * mistake and makes both unusable.
 */

/** sRGB 0-255 to CIE L*a*b* (D65). */
export function srgbToLab(r: number, g: number, b: number): [number, number, number] {
  const linear = (channel: number) => {
    const value = channel / 255
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4
  }
  const lr = linear(r)
  const lg = linear(g)
  const lb = linear(b)

  // sRGB D65 primaries.
  const x = (lr * 0.4124564 + lg * 0.3575761 + lb * 0.1804375) / 0.95047
  const y = lr * 0.2126729 + lg * 0.7151522 + lb * 0.072175
  const z = (lr * 0.0193339 + lg * 0.119192 + lb * 0.9503041) / 1.08883

  const f = (t: number) => (t > 216 / 24389 ? Math.cbrt(t) : (841 / 108) * t + 4 / 29)
  const fx = f(x)
  const fy = f(y)
  const fz = f(z)

  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]
}

/**
 * Precomputed Lab for a whole image, as a flat [L, a, b, L, a, b, …].
 *
 * The wand is re-run on every tolerance change while dragging a slider, so the
 * colour conversion must not be inside that loop.
 */
export function buildLabCache(data: Uint8ClampedArray): Float32Array {
  const pixels = data.length / 4
  const lab = new Float32Array(pixels * 3)
  for (let index = 0; index < pixels; index += 1) {
    const [l, a, b] = srgbToLab(data[index * 4], data[index * 4 + 1], data[index * 4 + 2])
    lab[index * 3] = l
    lab[index * 3 + 1] = a
    lab[index * 3 + 2] = b
  }
  return lab
}

/** CIE76 difference. Adequate here, and cheap enough to run per pixel per frame. */
export function deltaE(lab: Float32Array, indexA: number, indexB: number): number {
  const dl = lab[indexA * 3] - lab[indexB * 3]
  const da = lab[indexA * 3 + 1] - lab[indexB * 3 + 1]
  const db = lab[indexA * 3 + 2] - lab[indexB * 3 + 2]
  return Math.sqrt(dl * dl + da * da + db * db)
}

export interface WandOptions {
  lab: Float32Array
  width: number
  height: number
  x: number
  y: number
  /** Maximum CIE76 distance, in Lab units. Roughly 0-100 is the useful range. */
  tolerance: number
  /** False turns the same tool into a global chroma key (SS8.2). */
  contiguous: boolean
}

/**
 * Select similar. Returns 0/1 per pixel.
 *
 * Contiguous mode is a 4-connected flood fill from the clicked pixel; with
 * `contiguous` off, every pixel in the image within tolerance of the clicked
 * colour is selected regardless of position.
 */
export function magicWand(options: WandOptions): Uint8Array {
  const { lab, width, height, x, y, tolerance, contiguous } = options
  const mask = new Uint8Array(width * height)
  if (x < 0 || y < 0 || x >= width || y >= height) return mask
  const seed = y * width + x

  if (!contiguous) {
    for (let index = 0; index < mask.length; index += 1) {
      if (deltaE(lab, index, seed) <= tolerance) mask[index] = 1
    }
    return mask
  }

  // An explicit stack rather than recursion: a large flat region overflows the
  // call stack long before it exhausts memory.
  const stack = [seed]
  mask[seed] = 1
  while (stack.length > 0) {
    const index = stack.pop()!
    const px = index % width
    const py = (index - px) / width
    if (px > 0) visit(index - 1)
    if (px < width - 1) visit(index + 1)
    if (py > 0) visit(index - width)
    if (py < height - 1) visit(index + width)
  }
  return mask

  function visit(neighbour: number) {
    if (mask[neighbour]) return
    if (deltaE(lab, neighbour, seed) > tolerance) return
    mask[neighbour] = 1
    stack.push(neighbour)
  }
}

export type MaskOp = 'replace' | 'add' | 'subtract'

/** Shift-click adds another cluster; alt-click subtracts (SS8.2). */
export function combineMasks(base: Uint8Array, addition: Uint8Array, op: MaskOp): Uint8Array {
  if (op === 'replace') return addition.slice()
  const result = base.slice()
  for (let index = 0; index < result.length; index += 1) {
    if (!addition[index]) continue
    result[index] = op === 'add' ? 1 : 0
  }
  return result
}

export function invertMask(mask: Uint8Array): Uint8Array {
  const result = new Uint8Array(mask.length)
  for (let index = 0; index < mask.length; index += 1) result[index] = mask[index] ? 0 : 1
  return result
}

export function countMask(mask: Uint8Array): number {
  let total = 0
  for (let index = 0; index < mask.length; index += 1) total += mask[index]
  return total
}

/**
 * Feather: blur the binary selection's alpha, producing 0-255 coverage.
 *
 * Three box passes approximate a Gaussian closely enough for an edge falloff and
 * cost O(n) per pass rather than O(n·r²).
 */
export function featherMask(
  mask: Uint8Array,
  width: number,
  height: number,
  radius: number,
): Uint8ClampedArray {
  const alpha = new Uint8ClampedArray(mask.length)
  for (let index = 0; index < mask.length; index += 1) alpha[index] = mask[index] ? 255 : 0
  if (radius <= 0) return alpha

  let buffer = Float32Array.from(alpha)
  let scratch = new Float32Array(buffer.length)
  const passRadius = Math.max(1, Math.round(radius / 3))
  for (let pass = 0; pass < 3; pass += 1) {
    boxBlurH(buffer, scratch, width, height, passRadius)
    boxBlurV(scratch, buffer, width, height, passRadius)
  }
  scratch = new Float32Array(0)

  const feathered = new Uint8ClampedArray(mask.length)
  for (let index = 0; index < feathered.length; index += 1) feathered[index] = buffer[index]
  return feathered
}

function boxBlurH(
  source: Float32Array,
  target: Float32Array,
  width: number,
  height: number,
  radius: number,
) {
  const span = radius * 2 + 1
  for (let y = 0; y < height; y += 1) {
    const row = y * width
    let sum = 0
    for (let x = -radius; x <= radius; x += 1) sum += source[row + clampIndex(x, width)]
    for (let x = 0; x < width; x += 1) {
      target[row + x] = sum / span
      sum -= source[row + clampIndex(x - radius, width)]
      sum += source[row + clampIndex(x + radius + 1, width)]
    }
  }
}

function boxBlurV(
  source: Float32Array,
  target: Float32Array,
  width: number,
  height: number,
  radius: number,
) {
  const span = radius * 2 + 1
  for (let x = 0; x < width; x += 1) {
    let sum = 0
    for (let y = -radius; y <= radius; y += 1) sum += source[clampIndex(y, height) * width + x]
    for (let y = 0; y < height; y += 1) {
      target[y * width + x] = sum / span
      sum -= source[clampIndex(y - radius, height) * width + x]
      sum += source[clampIndex(y + radius + 1, height) * width + x]
    }
  }
}

/** Clamp-to-edge, so blurring does not darken the image border. */
function clampIndex(value: number, limit: number): number {
  return value < 0 ? 0 : value >= limit ? limit - 1 : value
}

export interface BrushOptions {
  width: number
  height: number
  x: number
  y: number
  radius: number
  /** 0 = fully soft, 1 = hard edge. */
  hardness: number
}

/** Round brush coverage, 0-1 per pixel, for both selection and density painting. */
export function brushStamp(options: BrushOptions): { index: number; coverage: number }[] {
  const { width, height, x, y, radius, hardness } = options
  const stamped: { index: number; coverage: number }[] = []
  const inner = radius * Math.min(0.999, Math.max(0, hardness))
  const minX = Math.max(0, Math.floor(x - radius))
  const maxX = Math.min(width - 1, Math.ceil(x + radius))
  const minY = Math.max(0, Math.floor(y - radius))
  const maxY = Math.min(height - 1, Math.ceil(y + radius))

  for (let py = minY; py <= maxY; py += 1) {
    for (let px = minX; px <= maxX; px += 1) {
      const distance = Math.hypot(px - x, py - y)
      if (distance > radius) continue
      const coverage =
        distance <= inner ? 1 : 1 - (distance - inner) / Math.max(1e-6, radius - inner)
      if (coverage > 0) stamped.push({ index: py * width + px, coverage })
    }
  }
  return stamped
}

/**
 * Marching ants: the boundary pixels of a selection.
 *
 * Drawn as a 1px outline rather than a filled wash, because a wash hides the
 * thing being judged -- whether the edge landed where you wanted it.
 */
export function maskOutline(mask: Uint8Array, width: number, height: number): Uint8Array {
  const outline = new Uint8Array(mask.length)
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x
      if (!mask[index]) continue
      const edge =
        x === 0 ||
        y === 0 ||
        x === width - 1 ||
        y === height - 1 ||
        !mask[index - 1] ||
        !mask[index + 1] ||
        !mask[index - width] ||
        !mask[index + width]
      if (edge) outline[index] = 1
    }
  }
  return outline
}
