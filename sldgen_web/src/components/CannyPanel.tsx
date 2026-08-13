import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { CannyPreview, ParamValue, Params } from '../api/types'
import { SPEC_BY_NAME } from '../lib/params'

const KNOBS = [
  'attract_canny_low',
  'attract_canny_high',
  'attract_canny_blur',
  'attract_canny_simplify',
  'attract_canny_min_length',
  'attract_canny_max_points',
] as const

/**
 * Canny-derived attraction, with a preview (Spec 5).
 *
 * The run generates its own edge SVG, because canvas space only exists once
 * SLDgen has masked, padded, resized and rescaled the target -- so there is
 * nothing to upload and nothing to register. The cost of that correctness is
 * that the thresholds are invisible until a job starts, which is what this panel
 * buys back: it runs the same code over a **previous run of the same image**,
 * whose `input.png` is that canvas, and overlays the result.
 *
 * No previous run means no preview, and that is stated rather than hidden: the
 * parameters still apply, they just cannot be shown yet.
 */
export function CannyPanel({
  params,
  targetSha256,
  onChange,
}: {
  params: Params
  targetSha256: string | null
  onChange: (name: string, value: ParamValue) => void
}) {
  const enabled = Boolean(params.attract_canny)
  const [preview, setPreview] = useState<CannyPreview | null>(null)
  const [problem, setProblem] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const generation = useRef(0)

  const knobValues = KNOBS.map((name) => params[name]).join(',')

  const refresh = useCallback(async () => {
    if (!enabled || !targetSha256) return
    const ticket = ++generation.current
    setBusy(true)
    try {
      const result = await api.cannyPreview({
        target_sha256: targetSha256,
        params: {
          low: params.attract_canny_low,
          high: params.attract_canny_high,
          blur: params.attract_canny_blur,
          simplify: params.attract_canny_simplify,
          min_length: params.attract_canny_min_length,
          max_points: params.attract_canny_max_points,
        },
      })
      // A slower earlier request must not overwrite a newer answer.
      if (ticket !== generation.current) return
      setPreview(result)
      setProblem(null)
    } catch (error) {
      if (ticket !== generation.current) return
      setPreview(null)
      setProblem(error instanceof Error ? error.message : 'Could not build the preview')
    } finally {
      if (ticket === generation.current) setBusy(false)
    }
  }, [enabled, targetSha256, knobValues]) // eslint-disable-line react-hooks/exhaustive-deps

  // Debounced: the knobs are number inputs and every keystroke is a change.
  useEffect(() => {
    if (!enabled || !targetSha256) return
    const timer = window.setTimeout(refresh, 350)
    return () => window.clearTimeout(timer)
  }, [enabled, targetSha256, knobValues, refresh])

  const controlPoints = Number(params.n_control_points)
  const tooMany = preview?.points != null && preview.points > controlPoints

  return (
    <div className={`optional${enabled ? '' : ' optional--off'}`}>
      <input
        type="checkbox"
        checked={enabled}
        aria-label="Use Canny attraction"
        onChange={(event) => onChange('attract_canny', event.target.checked)}
      />
      <div className="optional__body">
        <strong>{SPEC_BY_NAME.attract_canny.label}</strong>
        <div className="note">{SPEC_BY_NAME.attract_canny.hint}</div>

        {enabled && (
          <>
            <div className="grid-knobs">
              {KNOBS.map((name) => (
                <div className="field" key={name}>
                  <label htmlFor={`canny-${name}`}>{SPEC_BY_NAME[name].label}</label>
                  <input
                    id={`canny-${name}`}
                    type="number"
                    step={SPEC_BY_NAME[name].step ?? 1}
                    min={SPEC_BY_NAME[name].min}
                    value={Number(params[name])}
                    onChange={(event) => onChange(name, Number(event.target.value))}
                  />
                </div>
              ))}
            </div>

            <div
              className="canvas-wrap"
              style={{ position: 'relative', minHeight: 160, opacity: busy ? 0.55 : 1 }}
              aria-busy={busy}
            >
              {preview ? (
                // The inner box is sized by the image itself, so the overlay's
                // inset:0 is the image's box and not the panel's -- both are the
                // canvas, at whatever size it ends up displayed.
                <div style={{ position: 'relative', display: 'inline-block', lineHeight: 0 }}>
                  <img
                    src={preview.image_url}
                    alt="Canvas-space target"
                    style={{ maxWidth: '100%', maxHeight: 320, display: 'block' }}
                  />
                  {/* The SVG is in canvas pixels and declares the canvas as its
                      viewBox, so stretching it over the image registers exactly. */}
                  <img
                    src={`${preview.svg_url}?v=${knobValues}`}
                    alt="Canny attraction targets"
                    style={{
                      position: 'absolute',
                      inset: 0,
                      width: '100%',
                      height: '100%',
                      pointerEvents: 'none',
                    }}
                  />
                </div>
              ) : (
                <span className="muted note">
                  {busy
                    ? 'Tracing…'
                    : (problem ??
                      'No preview yet — pick a source image that has been run before.')}
                </span>
              )}
            </div>

            {preview && (
              <div className="note mono">
                {preview.summary}
                {preview.source_job_id && ` · from job ${preview.source_job_id.slice(-6)}`}
              </div>
            )}

            {tooMany && (
              <div className="warn">
                {preview?.points} targets for {controlPoints} control points. The attraction
                loss sums its coverage term over every target, so more targets than control
                points means a pull the curve cannot satisfy — it will fight the SDS gradient
                rather than guide it. Lower Canny max points.
              </div>
            )}

            {problem && preview && <div className="warn">{problem}</div>}
          </>
        )}
      </div>
    </div>
  )
}
