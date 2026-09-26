/**
 * The landmark editor's model (Spec 7 SS6): points, the merge rule that lets
 * auto-detect fill in without overwriting what the user placed, the file the
 * run reads, and the view-box arithmetic for zoom and pan.
 *
 * Pure functions only, so all of it is tested without a browser.
 */

export type LandmarkSource = 'mesh' | 'pose' | 'silhouette' | 'manual'

export interface EditorPoint {
  /** Stable key for React and selection; never written to the file. */
  id: string
  name: string
  /** Canvas pixels; null for a template row not yet placed. */
  xy: [number, number] | null
  weight: number
  source: LandmarkSource
  /** Moved, renamed or re-weighted by the user: detection leaves it alone. */
  edited: boolean
}

export interface DetectedPoint {
  name: string
  xy: [number, number]
  weight: number
  source?: string
}

export interface LandmarkView {
  kind: 'frontal' | 'turned' | 'profile' | null
  yaw: number | null
  method: 'mesh' | 'pose'
  facing: 'left' | 'right' | null
}

/** Profile names a user can place by hand (Spec 7 SS6.4), with their weights. */
export const PROFILE_TEMPLATE: { name: string; weight: number }[] = [
  { name: 'brow_ridge', weight: 1.5 },
  { name: 'nasion', weight: 2 },
  { name: 'eye', weight: 3 },
  { name: 'pupil', weight: 3 },
  { name: 'nose_tip', weight: 3 },
  { name: 'nostril', weight: 2 },
  { name: 'subnasale', weight: 1.5 },
  { name: 'upper_lip', weight: 2 },
  { name: 'mouth_corner', weight: 2 },
  { name: 'stomion', weight: 1.5 },
  { name: 'lower_lip', weight: 2 },
  { name: 'chin_front', weight: 1 },
  { name: 'jaw_angle', weight: 0.5 },
  { name: 'ear', weight: 1 },
]

export const WEIGHT_MIN = 0
export const WEIGHT_MAX = 5

let counter = 0
/** A fresh id. Deterministic within a session, which is all React needs. */
export function newId(): string {
  counter += 1
  return `lm${counter}`
}

function toSource(value: string | undefined): LandmarkSource {
  return value === 'pose' || value === 'silhouette' || value === 'manual' ? value : 'mesh'
}

/** Detection output as editor points. */
export function fromDetected(points: DetectedPoint[]): EditorPoint[] {
  return points.map((point) => ({
    id: newId(),
    name: point.name,
    xy: [point.xy[0], point.xy[1]],
    weight: point.weight,
    source: toSource(point.source),
    edited: false,
  }))
}

export interface MergeReport {
  points: EditorPoint[]
  added: number
  updated: number
  kept: number
  removed: number
}

/**
 * Auto-detect's merge rule (Spec 7 SS6.4). The user's work always wins:
 *
 * - a manual or edited point is kept untouched, and blocks the detected point
 *   of the same name;
 * - an unplaced template row takes the detected position of its name;
 * - a detected, never-edited point is updated in place;
 * - a detected, never-edited point the new detection no longer reports (it is
 *   now culled as hidden) is removed;
 * - any other detected name is added.
 */
export function mergeDetected(current: EditorPoint[], detected: DetectedPoint[]): MergeReport {
  const byName = new Map(detected.map((point) => [point.name, point]))
  const used = new Set<string>()
  let updated = 0
  let kept = 0
  let removed = 0
  const points: EditorPoint[] = []
  for (const point of current) {
    const found = byName.get(point.name)
    if (point.source === 'manual' || point.edited) {
      if (point.xy === null && found) {
        points.push({ ...point, xy: [found.xy[0], found.xy[1]] })
        used.add(point.name)
        updated += 1
      } else {
        points.push(point)
        if (found) used.add(point.name)
        kept += 1
      }
    } else if (found) {
      points.push({
        ...point,
        xy: [found.xy[0], found.xy[1]],
        weight: found.weight,
        source: toSource(found.source),
      })
      used.add(point.name)
      updated += 1
    } else {
      removed += 1
    }
  }
  const fresh = fromDetected(detected.filter((point) => !used.has(point.name)))
  return { points: [...points, ...fresh], added: fresh.length, updated, kept, removed }
}

/** Template rows for the profile names not already in the table. */
export function withTemplate(current: EditorPoint[]): EditorPoint[] {
  const have = new Set(current.map((point) => point.name))
  const rows = PROFILE_TEMPLATE.filter((entry) => !have.has(entry.name)).map((entry) => ({
    id: newId(),
    name: entry.name,
    xy: null,
    weight: entry.weight,
    source: 'manual' as const,
    edited: false,
  }))
  return [...current, ...rows]
}

/** `p1`, `p2`, ... -- the first free one. */
export function nextName(current: EditorPoint[]): string {
  const have = new Set(current.map((point) => point.name))
  let index = 1
  while (have.has(`p${index}`)) index += 1
  return `p${index}`
}

export function clampWeight(value: number): number {
  if (!Number.isFinite(value)) return 1
  return Math.min(WEIGHT_MAX, Math.max(WEIGHT_MIN, value))
}

/** The file the run reads (Spec 7 SS3). Unplaced rows are left out. */
export function serialize(points: EditorPoint[], imageSize: [number, number]) {
  return {
    space: 'canvas',
    image_size: imageSize,
    preset: 'edited',
    landmarks: points
      .filter((point) => point.xy !== null)
      .map((point) => ({
        name: point.name,
        xy: [round2(point.xy![0]), round2(point.xy![1])],
        weight: point.weight,
        source: point.source,
        edited: point.edited,
      })),
  }
}

function round2(value: number): number {
  return Math.round(value * 100) / 100
}

/** A landmark file (any preset) back into editor points, or an error message. */
export function parseFile(
  payload: unknown,
): { points: EditorPoint[]; imageSize: [number, number] } | string {
  if (!payload || typeof payload !== 'object') return 'not a landmark file'
  const data = payload as Record<string, unknown>
  if (data.space !== 'canvas') return 'the file is not in canvas space'
  const size = data.image_size as number[] | undefined
  if (!Array.isArray(size) || size.length !== 2) return 'the file has no image_size'
  const entries = Array.isArray(data.landmarks) ? data.landmarks : []
  const points: EditorPoint[] = []
  for (const entry of entries as Record<string, unknown>[]) {
    const xy = entry?.xy as number[] | undefined
    if (!Array.isArray(xy) || xy.length !== 2 || !xy.every(Number.isFinite)) continue
    points.push({
      id: newId(),
      name: String(entry.name ?? `p${points.length + 1}`),
      xy: [Number(xy[0]), Number(xy[1])],
      weight: clampWeight(Number(entry.weight ?? 1)),
      source: toSource(entry.source as string | undefined),
      edited: Boolean(entry.edited),
    })
  }
  return { points, imageSize: [Number(size[0]), Number(size[1])] }
}

/** How many points the run would actually use. */
export function placedCount(points: EditorPoint[]): number {
  return points.filter((point) => point.xy !== null && point.weight > 0).length
}

// -- view box ------------------------------------------------------------------

/** The visible part of the canvas, in canvas pixels. */
export interface ViewBox {
  x: number
  y: number
  w: number
  h: number
}

export const MAX_ZOOM = 16

export function fitView(imageSize: [number, number]): ViewBox {
  return { x: 0, y: 0, w: imageSize[0], h: imageSize[1] }
}

export function zoomLevel(view: ViewBox, imageSize: [number, number]): number {
  return imageSize[0] / view.w
}

/** Keep the view inside the canvas and between 1x and MAX_ZOOM. */
export function clampView(view: ViewBox, imageSize: [number, number]): ViewBox {
  const [width, height] = imageSize
  const w = Math.min(width, Math.max(width / MAX_ZOOM, view.w))
  const h = (w / width) * height
  const x = Math.min(width - w, Math.max(0, view.x))
  const y = Math.min(height - h, Math.max(0, view.y))
  return { x, y, w, h }
}

/** Zoom by `factor` (> 1 zooms in) keeping the canvas point `at` under the cursor. */
export function zoomAbout(
  view: ViewBox,
  factor: number,
  at: [number, number],
  imageSize: [number, number],
): ViewBox {
  const w = view.w / factor
  const h = view.h / factor
  const fx = (at[0] - view.x) / view.w
  const fy = (at[1] - view.y) / view.h
  return clampView({ x: at[0] - fx * w, y: at[1] - fy * h, w, h }, imageSize)
}

export function panBy(
  view: ViewBox,
  dx: number,
  dy: number,
  imageSize: [number, number],
): ViewBox {
  return clampView({ ...view, x: view.x - dx, y: view.y - dy }, imageSize)
}

/** A normalised box from two corners, clipped to the canvas. */
export function boxFrom(
  a: [number, number],
  b: [number, number],
  imageSize: [number, number],
): [number, number, number, number] {
  const clip = (value: number, max: number) => Math.min(max, Math.max(0, value))
  return [
    clip(Math.min(a[0], b[0]), imageSize[0]),
    clip(Math.min(a[1], b[1]), imageSize[1]),
    clip(Math.max(a[0], b[0]), imageSize[0]),
    clip(Math.max(a[1], b[1]), imageSize[1]),
  ]
}

/** Dot radius in canvas pixels, from the weight -- the same reading as Spec 6. */
export function dotRadius(weight: number): number {
  return 1.5 + weight
}
