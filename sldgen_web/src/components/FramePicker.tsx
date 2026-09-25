import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { FramesResponse, JobSummary } from '../api/types'
import { jobLabel } from '../lib/format'
import type { InputRef } from '../lib/formstate'
import { finalRef, frameRef } from '../lib/sources'

/**
 * Which iteration of another job to use as a constraint source.
 *
 * The contact sheet again, at picker size: the whole reason to reference
 * another job's geometry is usually that a particular moment in it was the good
 * one, and "the final SVG" is only occasionally that moment. Favourites are
 * marked, because a star is exactly the record of "this epoch was the one" —
 * made earlier, on the job page, by whoever was watching it emerge.
 *
 * A running job is offered too: its frames up to now are finished geometry and
 * the file will not change under you (the service copies it at submission).
 *
 * The one refusal is `rescaled`: a job whose intermediates declare a different
 * canvas than its `final_sld.svg` has them in a different coordinate space, and
 * geometry that does not register would push the new curve away from the wrong
 * place. The API asks the files the same question at submission, so what is
 * offered here is exactly what will be accepted.
 */
export function FramePicker({
  job,
  onPick,
  onBack,
}: {
  job: JobSummary
  onPick: (reference: InputRef) => void
  onBack: () => void
}) {
  const [frames, setFrames] = useState<FramesResponse | null>(null)
  const [favorites, setFavorites] = useState<number[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setFrames(null)
    setError(null)
    api
      .frames(job.id)
      .then((result) => {
        if (!cancelled) setFrames(result)
      })
      .catch((problem: unknown) => {
        if (!cancelled) setError(problem instanceof Error ? problem.message : 'Could not read frames')
      })
    api
      .favorites(job.id)
      .then((result) => {
        if (!cancelled) setFavorites(result.favorites.map((entry) => entry.epoch))
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [job.id])

  const starred = new Set(favorites)
  const usable = frames && !frames.rescaled ? frames.frames : []

  return (
    <div className="picker">
      <div className="picker__head">
        <button type="button" className="btn btn--small" onClick={onBack}>
          ← Jobs
        </button>
        <strong>{jobLabel(job)}</strong>
        <span className="note">
          {job.state === 'complete'
            ? `${job.num_iter} iterations`
            : `${job.current_epoch}/${job.num_iter} · ${job.state}`}
        </span>
      </div>

      {error && <div className="warn">{error}</div>}
      {!frames && !error && <div className="note">Reading frames…</div>}

      {frames && (
        <>
          <div className="btn-row" style={{ marginBottom: 8 }}>
            <button
              type="button"
              className="btn btn--small btn--primary"
              disabled={!frames.final_svg_url}
              title={
                frames.final_svg_url
                  ? 'Use the finished curve'
                  : 'This job has not written a final SVG yet'
              }
              onClick={() => onPick(finalRef(job))}
            >
              Final SVG
            </button>
            {favorites.length > 0 && <span className="eyebrow">Starred</span>}
            {favorites.map((epoch) => (
              <button
                key={epoch}
                type="button"
                className="chip"
                disabled={!usable.some((frame) => frame.epoch === epoch)}
                title={
                  usable.some((frame) => frame.epoch === epoch)
                    ? `Use epoch ${epoch}`
                    : 'That frame is no longer on disk'
                }
                onClick={() => onPick(frameRef(job, epoch))}
              >
                ★ {epoch}
              </button>
            ))}
          </div>

          {frames.rescaled ? (
            <p className="warn" style={{ margin: 0 }}>
              <strong>Only the final SVG, for this job.</strong> Its frames declare a different
              canvas than <span className="mono">final_sld.svg</span>, so they do not register
              with it and would move your curve by an unknown offset. The API refuses them.
            </p>
          ) : frames.frames.length === 0 ? (
            <div className="note">
              No frames yet. They appear every {frames.save_interval ?? '—'} iterations.
            </div>
          ) : (
            <div className="picker__frames">
              {frames.frames.map((frame) => (
                <figure
                  key={frame.epoch}
                  className={`filmstrip__frame${starred.has(frame.epoch) ? ' filmstrip__frame--starred' : ''}`}
                  role="button"
                  tabIndex={0}
                  title={`Use epoch ${frame.epoch}`}
                  aria-disabled={!frame.svg}
                  onClick={() => frame.svg && onPick(frameRef(job, frame.epoch))}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && frame.svg) onPick(frameRef(job, frame.epoch))
                  }}
                >
                  <img src={frame.png_url} alt={`epoch ${frame.epoch}`} loading="lazy" />
                  {starred.has(frame.epoch) && <span className="filmstrip__star">★</span>}
                  <figcaption>{frame.svg ? frame.epoch : `${frame.epoch} · no svg`}</figcaption>
                </figure>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}
