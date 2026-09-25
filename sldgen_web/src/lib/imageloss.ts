import { IMAGE_LOSS_DEFAULT_START } from './params'

/**
 * Image fidelity loss helpers (Spec 6 SS11): the schedule, for the panel's
 * sparkline, and the diagnostics CSV the run writes, for the job page.
 * Pure, so every piece is unit-tested without a DOM.
 */

/** Mirror of `SLDgen/image_loss.py::alpha_at`. */
export function alphaAt(
  epoch: number,
  numIter: number,
  schedule: string,
  weight: number,
  start: number | null = null,
): number {
  if (!(schedule in IMAGE_LOSS_DEFAULT_START)) return weight
  const begin = start ?? IMAGE_LOSS_DEFAULT_START[schedule]
  const t = epoch / Math.max(numIter - 1, 1)
  return begin + (weight - begin) * t
}

export interface ImageLossRow {
  epoch: number
  alpha: number
  sds_norm: number | null
  img_norm: number | null
  cosine: number | null
  /** 0 blended, 1 image gradient vanished, 2 SDS gradient vanished. */
  skipped: number
  loss_sds: number | null
  loss_img: number | null
  chamfer: number | null
  pyramid: number | null
  landmark: number | null
}

const NUMERIC = [
  'sds_norm',
  'img_norm',
  'cosine',
  'loss_sds',
  'loss_img',
  'chamfer',
  'pyramid',
  'landmark',
] as const

/** Parse `image_loss_log.csv`. Empty cells are null; malformed rows are dropped. */
export function parseImageLossLog(text: string): ImageLossRow[] {
  const lines = text.split(/\r?\n/).filter((line) => line.trim() !== '')
  if (lines.length === 0) return []
  const header = lines[0].split(',')
  const column = (name: string) => header.indexOf(name)
  if (column('epoch') < 0 || column('alpha') < 0) return []

  const cell = (cells: string[], name: string): number | null => {
    const index = column(name)
    if (index < 0 || cells[index] === undefined || cells[index] === '') return null
    const value = Number(cells[index])
    return Number.isFinite(value) ? value : null
  }

  const rows: ImageLossRow[] = []
  for (const line of lines.slice(1)) {
    const cells = line.split(',')
    const epoch = cell(cells, 'epoch')
    const alpha = cell(cells, 'alpha')
    if (epoch === null || alpha === null) continue
    const row: ImageLossRow = {
      epoch,
      alpha,
      skipped: cell(cells, 'skipped') ?? 0,
      sds_norm: null,
      img_norm: null,
      cosine: null,
      loss_sds: null,
      loss_img: null,
      chamfer: null,
      pyramid: null,
      landmark: null,
    }
    for (const name of NUMERIC) row[name] = cell(cells, name)
    rows.push(row)
  }
  return rows
}

export interface ImageLossSummary {
  rows: number
  lastEpoch: number | null
  alpha: number | null
  cosineMean: number | null
  cosineNegativeShare: number | null
  sdsNorm: { min: number; median: number; max: number } | null
  skippedShare: number
  latest: { chamfer: number | null; pyramid: number | null; landmark: number | null }
}

function median(sorted: number[]): number {
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
}

/** What the diagnostics panel shows. */
export function summarizeImageLoss(rows: ImageLossRow[]): ImageLossSummary {
  const last = rows.length ? rows[rows.length - 1] : null
  const cosines = rows.map((row) => row.cosine).filter((value): value is number => value !== null)
  const norms = rows
    .map((row) => row.sds_norm)
    .filter((value): value is number => value !== null)
    .sort((a, b) => a - b)
  return {
    rows: rows.length,
    lastEpoch: last?.epoch ?? null,
    alpha: last?.alpha ?? null,
    cosineMean: cosines.length ? cosines.reduce((a, b) => a + b, 0) / cosines.length : null,
    cosineNegativeShare: cosines.length
      ? cosines.filter((value) => value < 0).length / cosines.length
      : null,
    sdsNorm: norms.length
      ? { min: norms[0], median: median(norms), max: norms[norms.length - 1] }
      : null,
    skippedShare: rows.length ? rows.filter((row) => row.skipped !== 0).length / rows.length : 0,
    latest: {
      chamfer: last?.chamfer ?? null,
      pyramid: last?.pyramid ?? null,
      landmark: last?.landmark ?? null,
    },
  }
}

/**
 * An SVG polyline `points` string for a sparkline of `values` in a
 * `width` x `height` box. At most `maxPoints` points (evenly strided) so a
 * 8000-row log does not become an 8000-vertex path. Nulls are skipped.
 */
export function sparkline(
  values: (number | null)[],
  width: number,
  height: number,
  range?: [number, number],
  maxPoints = 200,
): string {
  const indexed = values
    .map((value, index) => [index, value] as const)
    .filter((entry): entry is readonly [number, number] => entry[1] !== null)
  if (indexed.length === 0) return ''
  const stride = Math.max(1, Math.ceil(indexed.length / maxPoints))
  const picked = indexed.filter((_entry, position) => position % stride === 0)
  const lastEntry = indexed[indexed.length - 1]
  if (picked[picked.length - 1] !== lastEntry) picked.push(lastEntry)

  const numbers = indexed.map((entry) => entry[1])
  const [low, high] = range ?? [Math.min(...numbers), Math.max(...numbers)]
  const span = high - low || 1
  const lastIndex = Math.max(values.length - 1, 1)
  return picked
    .map(([index, value]) => {
      const x = (index / lastIndex) * width
      const y = height - ((Math.min(Math.max(value, low), high) - low) / span) * height
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
}

/** Alpha over the whole run, for the panel's schedule sparkline. */
export function scheduleSeries(
  numIter: number,
  schedule: string,
  weight: number,
  start: number | null,
  samples = 50,
): number[] {
  const last = Math.max(numIter - 1, 0)
  return Array.from({ length: samples }, (_unused, index) =>
    alphaAt(Math.round((index / (samples - 1)) * last), numIter, schedule, weight, start),
  )
}
