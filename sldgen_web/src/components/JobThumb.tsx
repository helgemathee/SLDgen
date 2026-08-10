import { useState } from 'react'
import type { JobSummary } from '../api/types'

/**
 * The cache key for a job's preview image.
 *
 * `current_epoch` advances one step per saved frame -- the worker only learns a
 * new epoch when SLDgen writes its heartbeat, which is the same moment a new
 * frame lands -- so it is exactly the version of what `/preview` will return.
 * The state is in the key too, for the one swap the epoch does not cover: at the
 * end of a run `final_sld.png` replaces the last iteration frame while the epoch
 * stays put, and without this the thumbnail would keep showing the second-to-last
 * picture forever.
 *
 * Deliberately not `updated_at`: the worker rewrites that on every poll, so it
 * would refetch every thumbnail of every running job once a second.
 *
 * `viewed_epoch` is in the key because a job parked on a frame serves *that*
 * frame: parking, unparking and moving the mark all change the picture without
 * changing anything else about the job.
 */
export function previewSrc(job: JobSummary): string {
  return `${job.preview_url}?v=${job.state}-${job.current_epoch}-${job.viewed_epoch ?? 'live'}`
}

/**
 * A job's latest picture, retried as the job progresses.
 *
 * The failure is remembered per URL rather than as a boolean, and that is the
 * whole point of this component: a job that has not written its first frame yet
 * 404s, and a sticky "broken" flag left the row blank for the rest of the
 * session -- so a rail full of jobs that were queued when the page opened looked
 * like a rail full of jobs doing nothing, long after their artwork existed.
 */
export function JobThumb({
  job,
  className = '',
  emptyClassName,
  alt = '',
}: {
  job: JobSummary
  className?: string
  /** Rendered in place of the image before the first frame exists. */
  emptyClassName?: string
  alt?: string
}) {
  const [failed, setFailed] = useState<string | null>(null)
  const src = previewSrc(job)

  if (failed === src) return <span className={emptyClassName ?? className} />
  return (
    <img
      className={className}
      src={src}
      alt={alt}
      loading="lazy"
      onError={() => setFailed(src)}
    />
  )
}
