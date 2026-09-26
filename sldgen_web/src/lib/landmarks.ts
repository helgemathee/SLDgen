/**
 * The landmark editor's model (Spec 7 SS6): points, the merge rule that lets
 * auto-detect fill in without overwriting what the user placed, the file the
 * run reads, and the view-box arithmetic for zoom and pan.
 *
 * Pure functions only, so all of it is tested without a browser.
 */

export type LandmarkSource = 'mesh' | 'model' | 'pose' | 'silhouette' | 'manual'

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

/** A polyline landmark (Spec 7 addendum SS5): a hard edge such as a glasses rim. */
export interface EditorLine {
  id: string
  name: string
  /** Vertices in canvas pixels; the run densifies them (about one point per 8 px). */
  xy: [number, number][]
  closed: boolean
  /** The whole line's weight: it pulls as much as one landmark of this weight. */
  weight: number
  source: string
  edited: boolean
}

export interface DetectedLine {
  name: string
  xy: [number, number][]
  closed: boolean
  weight: number
  source?: string
}

/** The rigid-fit head pose (Spec 7 addendum SS3.4); a diagnostic only. */
export interface PoseReport {
  yaw: number
  pitch: number | null
  roll: number | null
  method: 'rigid-fit' | 'pose'
  residual_px?: number
}

/** `--landmark-set` values (Spec 7 addendum SS2), with what each is for. */
export const LANDMARK_SETS: { value: string; label: string; help: string }[] = [
  {
    value: 'sparse',
    label: 'sparse (~20)',
    help: 'Eyes, nose, mouth corners, brows, a few outline points. Holds the features; leaves the drawing loose.',
  },
  {
    value: 'standard',
    label: 'standard (~50)',
    help: 'Adds brow arcs, lids, nose, lip contour and more outline. Better feature fidelity; pose still loosely held.',
  },
  {
    value: 'dense',
    label: 'dense (120)',
    help: 'Points spread evenly over the face. Head angle and proportions become hard to get wrong.',
  },
  {
    value: 'pose-locked',
    label: 'pose-locked (120)',
    help: 'Dense, checked against a rigid 3D face fitted to the photo: points the detector got wrong on a turned face are re-placed where the fitted head puts them.',
  },
]

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
  return value === 'model' || value === 'pose' || value === 'silhouette' || value === 'manual'
    ? value
    : 'mesh'
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

/** Lines the run accepts: enough vertices for their kind (Spec 7 addendum SS6). */
export function lineIsValid(line: EditorLine): boolean {
  return line.xy.length >= (line.closed ? 3 : 2)
}

/**
 * The file the run reads (Spec 7 SS3, addendum SS5). Unplaced rows and lines
 * with too few vertices are left out; the pose report is carried along.
 */
export function serialize(
  points: EditorPoint[],
  imageSize: [number, number],
  extras: { lines?: EditorLine[]; pose?: PoseReport | null } = {},
) {
  const lines = (extras.lines ?? []).filter(lineIsValid)
  return {
    space: 'canvas',
    image_size: imageSize,
    preset: 'edited',
    ...(extras.pose ? { pose: extras.pose } : {}),
    landmarks: points
      .filter((point) => point.xy !== null)
      .map((point) => ({
        name: point.name,
        xy: [round2(point.xy![0]), round2(point.xy![1])],
        weight: point.weight,
        source: point.source,
        edited: point.edited,
      })),
    ...(lines.length
      ? {
          polylines: lines.map((line) => ({
            name: line.name,
            closed: line.closed,
            weight: line.weight,
            source: line.source,
            edited: line.edited,
            xy: line.xy.map(([x, y]) => [round2(x), round2(y)]),
          })),
        }
      : {}),
  }
}

function round2(value: number): number {
  return Math.round(value * 100) / 100
}

function toLine(entry: Record<string, unknown>, index: number): EditorLine | null {
  const raw = Array.isArray(entry?.xy) ? (entry.xy as unknown[]) : []
  const xy: [number, number][] = []
  for (const vertex of raw) {
    if (Array.isArray(vertex) && vertex.length === 2 && vertex.every(Number.isFinite))
      xy.push([Number(vertex[0]), Number(vertex[1])])
  }
  const line: EditorLine = {
    id: newId(),
    name: String(entry?.name ?? `line${index + 1}`),
    xy,
    closed: Boolean(entry?.closed),
    weight: clampWeight(Number(entry?.weight ?? 1)),
    source: String(entry?.source ?? 'manual'),
    edited: Boolean(entry?.edited),
  }
  return lineIsValid(line) ? line : null
}

/** A landmark file (any preset) back into editor points and lines, or an error message. */
export function parseFile(payload: unknown):
  | {
      points: EditorPoint[]
      lines: EditorLine[]
      pose: PoseReport | null
      imageSize: [number, number]
    }
  | string {
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
  const rawLines = Array.isArray(data.polylines) ? (data.polylines as Record<string, unknown>[]) : []
  const lines = rawLines.map(toLine).filter((line): line is EditorLine => line !== null)
  const pose =
    data.pose && typeof data.pose === 'object' && Number.isFinite((data.pose as PoseReport).yaw)
      ? (data.pose as PoseReport)
      : null
  return { points, lines, pose, imageSize: [Number(size[0]), Number(size[1])] }
}

/** How many points the run would actually use. */
export function placedCount(points: EditorPoint[]): number {
  return points.filter((point) => point.xy !== null && point.weight > 0).length
}

/** How many lines the run would actually use. */
export function lineCount(lines: EditorLine[]): number {
  return lines.filter((line) => lineIsValid(line) && line.weight > 0).length
}

/** The attached input's label: `21 landmarks (edited)`, `21 landmarks + 2 lines (edited)`. */
export function savedLabel(points: EditorPoint[], lines: EditorLine[]): string {
  const count = lineCount(lines)
  const extra = count ? ` + ${count} line${count === 1 ? '' : 's'}` : ''
  return `${placedCount(points)} landmarks${extra} (edited)`
}

// -- polylines ------------------------------------------------------------------

/** Detected polylines as editor lines. */
export function fromDetectedLines(lines: DetectedLine[]): EditorLine[] {
  return lines.map((line) => ({
    id: newId(),
    name: line.name,
    xy: line.xy.map(([x, y]) => [x, y] as [number, number]),
    closed: line.closed,
    weight: line.weight,
    source: line.source ?? 'manual',
    edited: false,
  }))
}

export interface LineMergeReport {
  lines: EditorLine[]
  added: number
  updated: number
  kept: number
  removed: number
}

/**
 * The merge rule for lines, by name (Spec 7 addendum SS8): a manual or edited
 * line is kept untouched; a detected, never-edited line is updated, or removed
 * when detection no longer finds it; new names are added.
 */
export function mergeLines(current: EditorLine[], detected: DetectedLine[]): LineMergeReport {
  const byName = new Map(detected.map((line) => [line.name, line]))
  const used = new Set<string>()
  let updated = 0
  let kept = 0
  let removed = 0
  const lines: EditorLine[] = []
  for (const line of current) {
    const found = byName.get(line.name)
    if (line.source === 'manual' || line.edited) {
      lines.push(line)
      if (found) used.add(line.name)
      kept += 1
    } else if (found) {
      lines.push({
        ...line,
        xy: found.xy.map(([x, y]) => [x, y] as [number, number]),
        closed: found.closed,
        weight: found.weight,
        source: found.source ?? line.source,
      })
      used.add(line.name)
      updated += 1
    } else {
      removed += 1
    }
  }
  const fresh = fromDetectedLines(detected.filter((line) => !used.has(line.name)))
  return { lines: [...lines, ...fresh], added: fresh.length, updated, kept, removed }
}

/** `line1`, `line2`, ... -- the first free one. */
export function nextLineName(lines: EditorLine[]): string {
  const have = new Set(lines.map((line) => line.name))
  let index = 1
  while (have.has(`line${index}`)) index += 1
  return `line${index}`
}

/** An empty open line, ready for vertices (double-click adds them). */
export function newLine(lines: EditorLine[]): EditorLine {
  return {
    id: newId(),
    name: nextLineName(lines),
    xy: [],
    closed: false,
    weight: 1,
    source: 'manual',
    edited: true,
  }
}

function segmentDistance(p: [number, number], a: [number, number], b: [number, number]): number {
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  const length = dx * dx + dy * dy
  const t = length === 0 ? 0 : Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length))
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))
}

/**
 * The vertices with `at` inserted on the nearest segment (the closing one too,
 * for a closed line); appended while the line has fewer than two vertices.
 * Returns the new vertices and the inserted index.
 */
export function insertVertex(
  xy: [number, number][],
  closed: boolean,
  at: [number, number],
): { xy: [number, number][]; index: number } {
  if (xy.length < 2) return { xy: [...xy, at], index: xy.length }
  const segments = closed && xy.length > 2 ? xy.length : xy.length - 1
  let best = 0
  let bestDistance = Infinity
  for (let k = 0; k < segments; k += 1) {
    const distance = segmentDistance(at, xy[k], xy[(k + 1) % xy.length])
    if (distance < bestDistance) {
      best = k
      bestDistance = distance
    }
  }
  // An open line grows at an end when the click lies beyond that end.
  if (!closed && best === 0 && beyond(at, xy[0], xy[1])) return { xy: [at, ...xy], index: 0 }
  if (!closed && best === segments - 1 && beyond(at, xy[xy.length - 1], xy[xy.length - 2]))
    return { xy: [...xy, at], index: xy.length }
  const next = [...xy]
  next.splice(best + 1, 0, at)
  return { xy: next, index: best + 1 }
}

/** Is `p` past `end`, seen from `other` (outside the segment's end)? */
function beyond(p: [number, number], end: [number, number], other: [number, number]): boolean {
  return (p[0] - end[0]) * (end[0] - other[0]) + (p[1] - end[1]) * (end[1] - other[1]) > 0
}

function ellipse(
  centre: [number, number],
  rx: number,
  ry: number,
  count = 12,
): [number, number][] {
  return Array.from({ length: count }, (_, k) => {
    const angle = (2 * Math.PI * k) / count
    return [round2(centre[0] + rx * Math.cos(angle)), round2(centre[1] + ry * Math.sin(angle))] as [
      number,
      number,
    ]
  })
}

function average(points: ([number, number] | null | undefined)[]): [number, number] | null {
  const placed = points.filter((point): point is [number, number] => Boolean(point))
  if (!placed.length) return null
  return [
    placed.reduce((sum, p) => sum + p[0], 0) / placed.length,
    placed.reduce((sum, p) => sum + p[1], 0) / placed.length,
  ]
}

/**
 * Glasses rims to drag onto the frames (Spec 7 addendum SS8): a closed
 * 12-vertex ellipse around each eye and an open bridge between them, skipping
 * names already present. The eyes come from the table (pupils or eye corners);
 * without them the rims start at the canvas centre.
 */
export function glassesTemplate(
  points: EditorPoint[],
  lines: EditorLine[],
  imageSize: [number, number],
): EditorLine[] {
  const at = (name: string) => points.find((point) => point.name === name)?.xy ?? null
  const eye = (side: 'right' | 'left') =>
    at(`${side}_pupil`) ?? average([at(`${side}_eye_outer`), at(`${side}_eye_inner`)])
  const widthOf = (side: 'right' | 'left') => {
    const outer = at(`${side}_eye_outer`)
    const inner = at(`${side}_eye_inner`)
    return outer && inner ? Math.hypot(outer[0] - inner[0], outer[1] - inner[1]) : null
  }
  let right = eye('right')
  let left = eye('left')
  const width =
    widthOf('right') ?? widthOf('left') ?? (right && left ? Math.abs(left[0] - right[0]) / 2.2 : imageSize[0] * 0.06)
  // The subject's right eye is on the image's left. A missing eye mirrors the other.
  if (right && !left) left = [right[0] + 2.2 * width, right[1]]
  if (left && !right) right = [left[0] - 2.2 * width, left[1]]
  if (!right || !left) {
    const centre: [number, number] = [imageSize[0] / 2, imageSize[1] / 2]
    right = [centre[0] - 1.1 * width, centre[1]]
    left = [centre[0] + 1.1 * width, centre[1]]
  }
  const rx = 0.85 * width
  const ry = 0.6 * width
  const bridgeY = Math.min(right[1], left[1]) - 0.15 * width
  const templates: Omit<EditorLine, 'id'>[] = [
    { name: 'glasses_right_rim', xy: ellipse(right, rx, ry), closed: true, weight: 3, source: 'manual', edited: true },
    { name: 'glasses_left_rim', xy: ellipse(left, rx, ry), closed: true, weight: 3, source: 'manual', edited: true },
    {
      name: 'glasses_bridge',
      xy: [
        [round2(right[0] + rx), round2(right[1])],
        [round2((right[0] + left[0]) / 2), round2(bridgeY)],
        [round2(left[0] - rx), round2(left[1])],
      ],
      closed: false,
      weight: 1,
      source: 'manual',
      edited: true,
    },
  ]
  const have = new Set(lines.map((line) => line.name))
  return [
    ...lines,
    ...templates.filter((line) => !have.has(line.name)).map((line) => ({ ...line, id: newId() })),
  ]
}

/** `yaw −18°, pitch 6°, roll −3°` (or the coarse Pose yaw). */
export function describePose(pose: PoseReport | null | undefined): string {
  if (!pose) return ''
  const deg = (value: number) => `${Math.round(value)}°`
  if (pose.method !== 'rigid-fit' || pose.pitch === null || pose.roll === null)
    return `yaw ${deg(pose.yaw)} (coarse)`
  return `yaw ${deg(pose.yaw)}, pitch ${deg(pose.pitch)}, roll ${deg(pose.roll)}`
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

// -- placement hints -------------------------------------------------------------

const HOW =
  'The line is pulled to pass through this point, so put it exactly on the edge you want drawn.'

/** Where each named landmark goes (Spec 7 SS6.3). Side-view names first. */
export const PLACEMENT_HINT: Record<string, string> = {
  brow_ridge: 'Side view: the most forward point of the brow bone on the outline, just above the eye.',
  nasion: 'The deepest point of the dip between forehead and nose (between the eyes); in a side view, on the outline, about level with the upper eyelid.',
  eye: 'The outer corner of the visible eye, where the upper and lower lids meet (the corner toward the ear).',
  pupil: 'The centre of the visible pupil. In a side view, the front edge of the iris.',
  nose_tip: 'The tip of the nose; in a side view, its most forward point on the outline.',
  nostril: 'The back curve of the nostril wing, where it meets the cheek.',
  subnasale: 'Where the underside of the nose meets the upper lip; in a side view, that inner corner on the outline.',
  upper_lip: 'The top edge of the upper lip at the middle; in a side view, its most forward point on the outline.',
  mouth_corner: 'The corner of the mouth, where the upper and lower lips meet at the side.',
  stomion: 'Where the lips meet, at the middle of the mouth; in a side view, that point on the outline.',
  lower_lip: 'The bottom edge of the lower lip at the middle; in a side view, its most forward point on the outline.',
  chin_front: 'Side view: the most forward point of the chin on the outline (with a beard, the beard’s outline).',
  jaw_angle: 'The corner of the jaw below the ear, where the jawline turns upward.',
  ear: 'On the outer rim of the ear: its back edge at about half the ear’s height, roughly level with the eye. For more of the ear’s shape, add a few low-weight points along the rim.',
  // Front and three-quarter names (from detection). Left/right are the subject's own.
  eye_outer: 'The outer corner of the eye, where the lids meet on the temple side.',
  eye_inner: 'The inner corner of the eye, next to the nose.',
  brow_outer: 'The outer end of the eyebrow.',
  brow_mid: 'The top of the eyebrow’s arch.',
  brow_inner: 'The inner end of the eyebrow, near the nose.',
  mouth: 'The corner of the mouth, where the upper and lower lips meet.',
  chin: 'The lowest point of the chin, on the face outline.',
  jaw: 'On the jawline, about halfway between the chin and the corner of the jaw.',
  cheek: 'On the face outline at the cheekbone, about level with the eyes.',
  // Standard set (Spec 7 addendum SS2).
  brow_outer_mid: 'On the top edge of the eyebrow, between its outer end and the top of the arch.',
  brow_inner_mid: 'On the top edge of the eyebrow, between the top of the arch and its inner end.',
  eye_upper: 'The middle of the upper eyelid’s edge, where it meets the eye.',
  eye_lower: 'The middle of the lower eyelid’s edge, where it meets the eye.',
  ala: 'The outer edge of the nostril wing, where it curves into the cheek.',
  lip_peak: 'The peak of the upper lip’s bow (the cupid’s bow), on its top edge.',
  temple: 'On the face outline at the temple, beside the eye.',
  jaw_high: 'On the face outline below the cheekbone, toward the corner of the jaw.',
  jaw_low: 'On the jawline, past the corner of the jaw, toward the chin.',
  chin_side: 'On the jawline just beside the chin.',
}

/** Hints for polylines (Spec 7 addendum SS7). */
const LINE_HINT: Record<string, string> = {
  glasses_right_rim:
    'The subject’s right lens rim (on the image’s left for a face looking at the camera): drag the squares onto the frame’s outline.',
  glasses_left_rim:
    'The subject’s left lens rim (on the image’s right for a face looking at the camera): drag the squares onto the frame’s outline.',
  glasses_bridge: 'The bridge of the glasses, from one rim over the nose to the other.',
  hairline: 'Where the hair meets the forehead. Detected from skin colour: check it, drag the squares where it strays.',
}

/** The hint for a polyline by name. */
export function lineHint(name: string): string {
  return (
    LINE_HINT[name] ??
    'Your own line. The line is pulled along its whole length; put the squares exactly on the edge you want drawn.'
  )
}

/** The placement hint for a name, handling `left_`/`_right` sides and `p1` points. */
export function placementHint(name: string): string {
  let side = ''
  let part = name
  for (const candidate of ['left', 'right']) {
    if (name.startsWith(`${candidate}_`)) {
      side = candidate
      part = name.slice(candidate.length + 1)
    } else if (name.endsWith(`_${candidate}`)) {
      side = candidate
      part = name.slice(0, -candidate.length - 1)
    }
  }
  const mesh = /^m(\d+)$/.exec(name)
  if (mesh)
    return `Face-mesh point ${mesh[1]} from the dense set: with its neighbours it holds the face’s shape and the head’s angle, so its weight is small by design. ${HOW}`
  const hint = PLACEMENT_HINT[part]
  if (!hint) return `Your own point. ${HOW}`
  const which = side
    ? ` This is the subject’s own ${side} side (on a face looking at the camera, their ${side} is on the image’s ${side === 'left' ? 'right' : 'left'}).`
    : ''
  return `${hint}${which} ${HOW}`
}

// -- references (open in a new tab) -------------------------------------------------

export interface Reference {
  label: string
  url: string
}

const REF = {
  faceMesh: {
    label: 'MediaPipe Face Landmarker',
    url: 'https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker',
  },
  meshMap: {
    label: 'Face mesh point map',
    url: 'https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png',
  },
  pose: {
    label: 'MediaPipe Pose Landmarker',
    url: 'https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker',
  },
  profile: {
    label: 'Soft-tissue profile landmarks',
    url: 'https://en.wikipedia.org/wiki/Cephalometric_analysis',
  },
  nasion: { label: 'Nasion', url: 'https://en.wikipedia.org/wiki/Nasion' },
  canthus: { label: 'Canthus (eye corners)', url: 'https://en.wikipedia.org/wiki/Canthus' },
  ear: { label: 'Helix (ear rim)', url: 'https://en.wikipedia.org/wiki/Helix_(ear)' },
} satisfies Record<string, Reference>

/** The general references shown under the editor. */
export const REFERENCES: Reference[] = [REF.faceMesh, REF.meshMap, REF.pose, REF.profile]

const PROFILE_PARTS = new Set([
  'brow_ridge', 'nose_tip', 'subnasale', 'upper_lip', 'stomion', 'lower_lip', 'chin_front', 'jaw_angle',
])

/** The most useful page about one landmark, or null for the user's own points. */
export function referenceFor(name: string): Reference | null {
  const part = name.replace(/^(left|right)_/, '').replace(/_(left|right)$/, '')
  if (part === 'nasion') return REF.nasion
  if (part === 'ear') return REF.ear
  if (part === 'eye' || part === 'eye_outer' || part === 'eye_inner') return REF.canthus
  if (PROFILE_PARTS.has(part)) return REF.profile
  if (part in PLACEMENT_HINT || /^m\d+$/.test(name)) return REF.meshMap
  return null
}
