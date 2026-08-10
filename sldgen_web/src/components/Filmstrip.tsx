import { useEffect, useRef } from 'react'
import type { FramesResponse } from '../api/types'

/**
 * The contact sheet (Spec 3 SS3, SS6.2) — the signature element.
 *
 * The iteration history as a horizontal strip of the actual frames: the drawing
 * emerging from noise into structure, scrubbable. It is the most characteristic
 * thing this system produces and it is also exactly the tool needed for the
 * problem that a job takes twenty minutes and the interesting result is often
 * not the final one.
 *
 * The mp4 is offered as a download but is deliberately not the scrubbing
 * mechanism: frames seek instantly and video does not.
 */
export function Filmstrip({
  frames,
  index,
  onIndex,
  playing,
  onPlaying,
  favorites,
  onToggleFavorite,
}: {
  frames: FramesResponse
  index: number
  onIndex: (index: number) => void
  playing: boolean
  onPlaying: (playing: boolean) => void
  /** Epochs starred on this job, ascending -- kept on the server (SS6.2). */
  favorites: number[]
  onToggleFavorite: (epoch: number) => void
}) {
  const track = useRef<HTMLDivElement>(null)
  const count = frames.frames.length
  const current = frames.frames[index]
  const starred = new Set(favorites)
  const isStarred = current ? starred.has(current.epoch) : false

  // 10 fps, matching the mp4, so the two show the same motion.
  useEffect(() => {
    if (!playing || count === 0) return
    const timer = setInterval(() => {
      onIndex(index + 1 >= count ? 0 : index + 1)
    }, 100)
    return () => clearInterval(timer)
  }, [playing, index, count, onIndex])

  // Keep the selected frame in view while scrubbing or playing.
  useEffect(() => {
    const element = track.current?.children[index] as HTMLElement | undefined
    element?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }, [index])

  if (count === 0) {
    return (
      <div className="panel">
        <div className="panel__head">
          <span className="eyebrow">History</span>
        </div>
        <div className="panel__body note">
          No frames yet. They appear every {frames.save_interval ?? '—'} iterations once the job
          starts.
        </div>
      </div>
    )
  }

  return (
    <div className="panel">
      <div className="panel__head">
        <span className="eyebrow">History · contact sheet</span>
        <span className="mono">
          {index + 1}/{count} · epoch {current?.epoch}
        </span>
        {frames.video_url && (
          <a className="btn btn--small" href={frames.video_url} download>
            mp4
          </a>
        )}
      </div>

      {/* Above the sheet, not below it: the slider drives the artwork at the top
        * of the page, and on a laptop screen the sheet is tall enough to push a
        * scrubber under it off-screen -- leaving you dragging one thing while
        * watching another you cannot see. Everything here acts on the selected
        * frame, so the buttons travel with the slider rather than staying put. */}
      <div className="filmstrip__controls">
        <button
          type="button"
          className="btn btn--small"
          onClick={() => onPlaying(!playing)}
          aria-pressed={playing}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <input
          className="filmstrip__slider"
          type="range"
          min={0}
          max={count - 1}
          step={1}
          value={index}
          aria-label="Frame"
          onChange={(event) => {
            onPlaying(false)
            onIndex(Number(event.target.value))
          }}
        />
        <span className="mono">epoch {current?.epoch}</span>
        {current && (
          <button
            type="button"
            className={`btn btn--small${isStarred ? ' btn--starred' : ''}`}
            aria-pressed={isStarred}
            title={
              isStarred
                ? `Unstar epoch ${current.epoch}`
                : `Star epoch ${current.epoch} (f) — starred frames download together`
            }
            onClick={() => onToggleFavorite(current.epoch)}
          >
            {isStarred ? '★' : '☆'}
          </button>
        )}
        {current?.svg_url && (
          <>
            <a
              className="btn btn--small"
              href={current.svg_url}
              target="_blank"
              rel="noreferrer"
              title="Open this frame's SVG"
            >
              svg
            </a>
            <a className="btn btn--small" href={current.svg_url} download>
              ↓
            </a>
          </>
        )}
      </div>

      <div className="filmstrip__track" ref={track}>
        {frames.frames.map((frame, position) => (
          <figure
            key={frame.epoch}
            className={`filmstrip__frame${starred.has(frame.epoch) ? ' filmstrip__frame--starred' : ''}`}
            aria-current={position === index}
            onClick={() => {
              onPlaying(false)
              onIndex(position)
            }}
            role="button"
            tabIndex={-1}
          >
            <img src={frame.png_url} alt={`epoch ${frame.epoch}`} loading="lazy" />
            {starred.has(frame.epoch) && <span className="filmstrip__star">★</span>}
            <figcaption>{frame.epoch}</figcaption>
          </figure>
        ))}
      </div>

      {favorites.length > 0 && (
        <div className="filmstrip__favorites">
          <span className="eyebrow">Starred</span>
          {favorites.map((epoch) => {
            const position = frames.frames.findIndex((frame) => frame.epoch === epoch)
            return (
              <button
                key={epoch}
                type="button"
                className="chip"
                aria-pressed={position === index}
                // A frame can be starred and then pruned; the epoch is still the
                // answer to "which one was good", so it stays listed, just dead.
                disabled={position === -1}
                title={position === -1 ? 'That frame is no longer on disk' : `Jump to epoch ${epoch}`}
                onClick={() => {
                  onPlaying(false)
                  onIndex(position)
                }}
              >
                ★ {epoch}
              </button>
            )
          })}
          <a className="btn btn--small" href={`/api/jobs/${frames.job_id}/favorites.zip`} download>
            ↓ SVGs
          </a>
        </div>
      )}

      <div className="panel__body" style={{ paddingTop: 0, display: 'grid', gap: 6 }}>
        <p className="note" style={{ margin: 0 }}>
          Frames exist only every {frames.save_interval ?? '—'} iterations, so this slider steps
          rather than sliding.
        </p>
        {frames.rescaled && (
          <p className="warn" style={{ margin: 0 }}>
            <strong>These intermediates are in a different coordinate space.</strong> This job
            rescaled its object, and intermediate SVGs are written before that rescale runs — so
            they do not register with <span className="mono">final_sld.svg</span>. They can be
            viewed and downloaded, but only the final SVG may be used as an avoid, attract or init
            source, or as a partition source. The API refuses the others rather than misregistering
            them silently.
          </p>
        )}
      </div>
    </div>
  )
}
