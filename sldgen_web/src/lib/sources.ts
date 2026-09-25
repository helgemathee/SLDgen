import type { JobSummary, UploadResult } from '../api/types'
import type { InputRef } from './formstate'
import { jobLabel } from './format'

/**
 * Where the files behind `--avoid`, `--attract` and `--init-points` come from.
 *
 * Three kinds of source, all resolving to the same `InputRef` the service
 * copies at submission time (Spec 2 SS4.3):
 *
 *   * a **job's final SVG** — the original offer, and the only one for a job
 *     whose frames are in a different coordinate space;
 *   * a **frame** of any job that has drawn one, including a job still running.
 *     "Avoid that other run at iteration 1700" is the thing you actually want
 *     when the interesting curve was not the final one;
 *   * an **uploaded SVG**, for geometry that came from somewhere else entirely.
 *
 * The one thing every consumer of these must agree on is the *coordinate
 * space*: everything the curve is pushed away from is read in canvas pixels at
 * `--render-size`. Frames and `final_sld.svg` are both written at that size —
 * the doubling in `run.py` happens after `save_svg`, for the PNG only, and the
 * `--object-size-ratio` rescale never reaches either file — so they register
 * with each other. `FramesResponse.rescaled` reports the exception, measured
 * from the canvases the files declare rather than assumed from a flag. An
 * upload is the one case nobody can check, which is why the picker says so.
 */

/** Jobs that have an SVG to offer: a final export, or at least one frame. */
export function canSupplySvg(job: JobSummary): boolean {
  if (job.state === 'deleting') return false
  return job.state === 'complete' || job.current_epoch > 0
}

/**
 * The rail's search, over the jobs that can supply one.
 *
 * Same three fields the rail filter matches on, because after fifty jobs the
 * grid of thumbnails is no longer a way to find the one you mean, and the
 * thing you remember is its title.
 */
export function searchSources(jobs: JobSummary[], query: string): JobSummary[] {
  const needle = query.trim().toLowerCase()
  const candidates = jobs.filter(canSupplySvg)
  if (!needle) return candidates
  return candidates.filter(
    (job) =>
      (job.title ?? '').toLowerCase().includes(needle) ||
      (job.resolved_caption ?? '').toLowerCase().includes(needle) ||
      job.id.toLowerCase().includes(needle),
  )
}

/** One frame's SVG, relative to the run directory. run.py leaves it unpadded. */
export function frameSvgPath(epoch: number): string {
  return `svg_logs/svg_iter${epoch}.svg`
}

export function frameRef(job: JobSummary, epoch: number): InputRef {
  return {
    source_kind: 'job',
    source_job_id: job.id,
    path: frameSvgPath(epoch),
    label: `${jobLabel(job)} · epoch ${epoch}`,
  }
}

export function finalRef(job: JobSummary): InputRef {
  return {
    source_kind: 'job',
    source_job_id: job.id,
    path: 'final_sld.svg',
    label: `${jobLabel(job)} · final_sld.svg`,
  }
}

export function uploadRef(upload: UploadResult, filename: string): InputRef {
  return { source_kind: 'upload', sha256: upload.sha256, label: filename }
}

/**
 * Where to fetch a reference for the prep canvas overlay, or null for one that
 * cannot be drawn (a non-SVG, or a reference missing the id it needs).
 *
 * Uploads answer from the content-addressed store rather than a job directory,
 * which is why this is a function and not a template literal at the call site.
 */
export function overlayUrl(reference: InputRef): string | null {
  if (reference.source_kind === 'upload') {
    if (!reference.sha256) return null
    return (reference.label ?? '').toLowerCase().endsWith('.svg')
      ? `/api/uploads/${reference.sha256}`
      : null
  }
  if (!(reference.path ?? '').endsWith('.svg')) return null
  if (reference.source_kind === 'job') {
    return reference.source_job_id
      ? `/api/jobs/${reference.source_job_id}/files/target/run/${reference.path}`
      : null
  }
  return reference.source_partition_id
    ? `/api/partitions/${reference.source_partition_id}/files/${reference.path}`
    : null
}

/** What a chosen reference is called in the list of what you picked. */
export function refLabel(reference: InputRef): string {
  return reference.label ?? reference.path ?? reference.sha256?.slice(0, 12) ?? 'source'
}

/** Stable identity for a reference, so picking the same frame twice is visible. */
export function refKey(reference: InputRef): string {
  return [
    reference.source_kind,
    reference.source_job_id ?? reference.source_partition_id ?? reference.sha256 ?? '',
    reference.path ?? '',
  ].join('|')
}
