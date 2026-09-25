import type { JobSummary, Params } from '../api/types'
import { sameRun } from './paramdiff'

/**
 * The variant table (Spec 3 SS6.6).
 *
 * The dominant exploration pattern is not one edited copy but a handful of
 * variants queued together: five runs differing only by seed, then later another
 * five differing by seed *and* caption.
 */

export interface Variant {
  /** Stable key for React; not sent to the server. */
  key: string
  enabled: boolean
  params: Params
  title?: string
}

/**
 * Sequential from the parent's seed, not random.
 *
 * It is reproducible, you can tell at a glance which family a run belongs to,
 * and re-running the batch tomorrow does not silently collide.
 */
export function sequentialSeeds(parentSeed: number, count: number): number[] {
  return Array.from({ length: count }, (_unused, index) => parentSeed + index + 1)
}

/**
 * A random block, for when the sequential range has been used up.
 *
 * Contiguous rather than N independent draws: a block still reads as one family
 * in the rail, which is the property sequential seeds were chosen for.
 */
export function randomSeedBlock(count: number, random: () => number = Math.random): number[] {
  const start = Math.floor(random() * 900_000) + 1000
  return Array.from({ length: count }, (_unused, index) => start + index)
}

/** Resize the table, keeping edits already made to the rows that survive. */
export function resizeVariants(
  variants: Variant[],
  count: number,
  makeRow: (index: number) => Variant,
): Variant[] {
  if (count <= variants.length) return variants.slice(0, count)
  return [
    ...variants,
    ...Array.from({ length: count - variants.length }, (_unused, index) =>
      makeRow(variants.length + index),
    ),
  ]
}

/**
 * Existing jobs that this variant would duplicate.
 *
 * Not blocked -- re-running a seed is occasionally deliberate -- but it should
 * never happen by accident, so the row says so inline.
 */
export function findDuplicates(variant: Params, existing: JobSummary[]): JobSummary[] {
  return existing.filter((job) => job.params && sameRun(variant, job.params))
}

const SEED_PART = /^s\d+$/i
const NUMBER_PART = /^v\d+$/i

/**
 * The name somebody chose, with the parts that only told runs apart removed.
 *
 * A seed (`· s4050`, or `S4050` typed at the end) and a rename's `· v02` both
 * describe the parent, not the family, so carrying them into a child would
 * stack one seed per generation: `helge · s4050 · s4051 · s4052`.
 */
export function baseTitle(title: string): string {
  const parts = title
    .split(' · ')
    .map((part) => part.trim())
    .filter((part) => part && !SEED_PART.test(part) && !NUMBER_PART.test(part))
  let base = parts.join(' · ')
  // A seed typed by hand rather than by a rename: `helge S4050`.
  let trimmed = base.replace(/\s+s\d+$/i, '')
  while (trimmed !== base) {
    base = trimmed
    trimmed = base.replace(/\s+s\d+$/i, '')
  }
  return base || title.trim()
}

/**
 * `car_03 · s1042`, with the caption difference appended where one exists.
 *
 * Derived automatically and legibly, so a batch of five is readable in the rail
 * without opening anything. Built on the parent's chosen name rather than its
 * whole title, so the parent's own seed is not carried along.
 */
export function variantTitle(
  parentTitle: string,
  params: Params,
  parentParams: Params,
): string {
  const parts = [baseTitle(parentTitle)]
  if (params.seed !== parentParams.seed) parts.push(`s${params.seed}`)
  if (params.caption !== parentParams.caption) {
    const caption = String(params.caption ?? '').trim()
    parts.push(caption ? truncate(caption, 28) : 'auto caption')
  }
  if (parts.length === 1) parts.push(`s${params.seed}`)
  return parts.join(' · ')
}

function truncate(text: string, limit: number): string {
  return text.length <= limit ? text : `${text.slice(0, limit - 1)}…`
}

/**
 * Wall-clock cost of the batch, from the running average it/s.
 *
 * FIFO, one job at a time (Spec 2 SS6), so the batch's time is the sum of its
 * runs rather than the longest of them.
 */
export function estimateBatchSeconds(
  variants: Variant[],
  targetEpoch: number,
  itersPerSec: number | null,
): number | null {
  if (!itersPerSec || itersPerSec <= 0) return null
  const enabled = variants.filter((variant) => variant.enabled).length
  return (enabled * targetEpoch) / itersPerSec
}
