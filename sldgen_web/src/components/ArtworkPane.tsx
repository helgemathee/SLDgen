import { useCallback, useEffect, useRef, useState } from 'react'
import { fileUrl } from '../api/client'
import type { JobDetail } from '../api/types'
import { formatBytes } from '../lib/format'
import { dotRadius, lineIsValid, parseFile, type EditorLine, type EditorPoint } from '../lib/landmarks'
import { canOverlay, overlayStyle, withViewBox, type OverlayBlend } from '../lib/overlay'
import { ZOOM_STEP, actualSizeView, fitView, rescaleView, zoomCentered } from '../lib/zoom'
import { previewSrc } from './JobThumb'

export type ArtworkTab =
  | 'result'
  | 'preview'
  | 'input'
  | 'mask'
  | 'condition'
  | 'weight'
  | 'edges'
  | 'landmarks'

interface Available {
  result: string | null
  preview: string | null
  input: string | null
  mask: string | null
  condition: string | null
  weight: string | null
  edges: string | null
  /** The landmark JSON the run was given (an input, like the weight map). */
  landmarks: string | null
}

export function availableArtwork(job: JobDetail, frameUrl: string | null): Available {
  const has = (path: string) => job.artifacts.some((artifact) => artifact.path === path)
  const run = (name: string) => (has(`target/run/${name}`) ? fileUrl(job.id, `target/run/${name}`) : null)
  const condition = job.artifacts.find((artifact) =>
    /^target\/run\/condition_\w+\.png$/.test(artifact.path),
  )
  // The weight map is an *input*, not something the run produced, so it lives in
  // inputs/ rather than target/run/. Shown raw, as painted: white is full ink,
  // black is none. In multiply mode the field that actually seeds the stipple is
  // this times the RMBG mask, and that product is never written to disk -- the
  // Mask tab next to it is the other half.
  const weight = job.inputs.find((input) => input.role === 'stipple_weight')
  const landmarks = job.inputs.find((input) => input.role === 'image_loss_landmarks')
  const inputUrl = (input: { stored_path: string }) =>
    fileUrl(job.id, `inputs/${input.stored_path.split('/').pop()}`)
  return {
    result: run('final_sld.svg'),
    preview: frameUrl ?? (job.current_epoch > 0 ? previewSrc(job) : null),
    input: run('input.png'),
    mask: run('mask.png'),
    condition: condition ? fileUrl(job.id, condition.path) : null,
    weight: weight ? inputUrl(weight) : null,
    // What --image-loss actually compared against, after the mask: written by
    // the run whether the map was derived or supplied.
    edges: run('image_loss_target.png'),
    landmarks: landmarks ? inputUrl(landmarks) : null,
  }
}

/** The most advanced artefact available -- what the page should open on. */
export function defaultTab(available: Available): ArtworkTab {
  if (available.result) return 'result'
  if (available.preview) return 'preview'
  if (available.input) return 'input'
  return 'result'
}

const TAB_LABELS: Record<ArtworkTab, string> = {
  result: 'Result',
  preview: 'Preview',
  input: 'Input',
  mask: 'Mask',
  condition: 'Condition',
  weight: 'Stipple weight',
  edges: 'Edge target',
  landmarks: 'Landmarks',
}

/** Tabs only a few jobs have: hidden rather than shown disabled on the rest. */
const OPT_IN_TABS: ArtworkTab[] = ['edges', 'landmarks']

/**
 * The artwork viewer (Spec 3 SS6.1).
 *
 * All tabs share one pan/zoom state, because toggling between Input, Mask and
 * Result at the same zoom is how you diagnose a bad run -- if each tab reset the
 * view, the comparison the tabs exist for would be impossible.
 */
export function ArtworkPane({
  job,
  frameUrl,
  frameEpoch,
  tab,
  onTab,
}: {
  job: JobDetail
  frameUrl: string | null
  frameEpoch: number | null
  tab: ArtworkTab
  onTab: (tab: ArtworkTab) => void
}) {
  const available = availableArtwork(job, frameUrl)
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 })
  const [dragging, setDragging] = useState(false)
  const [paperBackground, setPaperBackground] = useState(true)
  const [strokeBoost, setStrokeBoost] = useState(false)
  const [svgMarkup, setSvgMarkup] = useState<string | null>(null)
  const [svgStats, setSvgStats] = useState<{ length: number; segments: number } | null>(null)
  const [landmarkFile, setLandmarkFile] = useState<LandmarkFile | null>(null)
  // The current tab drawn over the input, to judge how well it matches. Shared
  // by all tabs, like the view: flipping tabs with it on compares each in turn.
  const [overlay, setOverlay] = useState(false)
  const [overlayOpacity, setOverlayOpacity] = useState(0.5)
  const [overlayBlend, setOverlayBlend] = useState<OverlayBlend>('normal')
  const container = useRef<HTMLDivElement>(null)
  const stage = useRef<HTMLDivElement>(null)
  const origin = useRef({ x: 0, y: 0, viewX: 0, viewY: 0 })
  // Measuring the content needs the scale it is currently drawn at, and the
  // measurement happens in callbacks that would otherwise close over a stale one.
  const viewRef = useRef(view)
  viewRef.current = view
  const fittedOnce = useRef(false)
  // The natural size of whatever the stage last showed, so a tab switch can
  // tell "same picture, re-measured" from "a different-sized artefact".
  const lastContent = useRef<{ width: number; height: number } | null>(null)

  const url = available[tab]
  const overlaid = overlay && canOverlay(tab, Boolean(available.input))

  /**
   * The untransformed size of whatever is on the stage, and of the viewport.
   *
   * The stage carries a CSS transform, so its rect is divided back out rather
   * than read from `offsetWidth` -- an inlined SVG's layout box does not always
   * agree with what it actually paints.
   */
  const measure = useCallback(() => {
    const viewportRect = container.current?.getBoundingClientRect()
    const content = stage.current?.firstElementChild as HTMLElement | undefined
    const contentRect = content?.getBoundingClientRect()
    if (!viewportRect || !contentRect) return null
    const scale = viewRef.current.scale || 1
    return {
      viewport: { width: viewportRect.width, height: viewportRect.height },
      content: { width: contentRect.width / scale, height: contentRect.height / scale },
    }
  }, [])

  const fit = useCallback(() => {
    const measured = measure()
    if (!measured) return false
    const fitted = fitView(measured.content, measured.viewport)
    if (!fitted) return false
    lastContent.current = measured.content
    setView(fitted)
    return true
  }, [measure])

  /**
   * Take the view across a change of content (SS6.1).
   *
   * The first measurable content is fitted; after that the view belongs to the
   * user, and a tab whose artefact has a different pixel size gets the same
   * framing rather than the same raw scale -- see `rescaleView`. Returns false
   * when nothing is measurable yet, so the caller can retry.
   */
  const syncContent = useCallback(() => {
    const measured = measure()
    if (!measured) return false
    const { content, viewport } = measured
    if (!(content.width > 0) || !(content.height > 0)) return false

    if (!fittedOnce.current) {
      const fitted = fitView(content, viewport)
      if (!fitted) return false
      lastContent.current = content
      fittedOnce.current = true
      setView(fitted)
      return true
    }

    const previous = lastContent.current
    lastContent.current = content
    // Sub-pixel differences are measurement noise, not a new artefact.
    if (
      previous &&
      (Math.abs(previous.width - content.width) > 0.5 ||
        Math.abs(previous.height - content.height) > 0.5)
    ) {
      setView((current) => rescaleView(current, previous, content, viewport))
    }
    return true
  }, [measure])

  const actualSize = useCallback(() => {
    const measured = measure()
    setView(
      measured
        ? actualSizeView(measured.content, measured.viewport)
        : { scale: 1, x: 0, y: 0 },
    )
  }, [measure])

  const step = useCallback(
    (factor: number) => {
      const measured = measure()
      const viewport = measured?.viewport ?? { width: 0, height: 0 }
      setView((current) => zoomCentered(current, factor, viewport))
    },
    [measure],
  )

  // Fit is the opening view, but only once: after that the view is the user's,
  // and the tabs exist to be compared at a shared framing (see above). The frame
  // retry covers content that has not been laid out yet on the first pass;
  // an <img> that is still loading is caught by its onLoad instead.
  useEffect(() => {
    if (syncContent()) return
    const frame = requestAnimationFrame(syncContent)
    return () => cancelAnimationFrame(frame)
  }, [syncContent, tab, url, svgMarkup, landmarkFile, overlaid])

  // The landmarks the run was given, drawn on the canvas they were placed on.
  useEffect(() => {
    if (tab !== 'landmarks' || !available.landmarks) {
      setLandmarkFile(null)
      return
    }
    let cancelled = false
    fetch(available.landmarks)
      .then((response) => response.json())
      .then((payload) => {
        if (cancelled) return
        const parsed = parseFile(payload)
        setLandmarkFile(typeof parsed === 'string' ? { error: parsed } : parsed)
      })
      .catch((error) => !cancelled && setLandmarkFile({ error: String(error) }))
    return () => {
      cancelled = true
    }
  }, [tab, available.landmarks])

  // The result is inlined rather than put in an <img> so it can be zoomed
  // without resampling and so its geometry can be measured (SS6.1).
  useEffect(() => {
    if (tab !== 'result' || !available.result) {
      setSvgMarkup(null)
      setSvgStats(null)
      return
    }
    let cancelled = false
    fetch(available.result)
      .then((response) => response.text())
      .then((text) => {
        if (cancelled) return
        setSvgMarkup(text)
        setSvgStats(measureSvg(text))
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [tab, available.result])

  return (
    <div className="panel">
      <div className="panel__head" style={{ padding: '4px 8px 0', background: 'var(--paper-sunk)' }}>
        <div className="tabs" role="tablist">
          {(Object.keys(TAB_LABELS) as ArtworkTab[])
            .filter((name) => available[name] || !OPT_IN_TABS.includes(name))
            .map((name) => (
              <button
                key={name}
                type="button"
                role="tab"
                className="tab"
                aria-selected={tab === name}
                disabled={!available[name]}
                onClick={() => onTab(name)}
              >
                {TAB_LABELS[name]}
              </button>
            ))}
        </div>
      </div>

      <div
        ref={container}
        className={`artwork${dragging ? ' artwork--grabbing' : ''}${
          paperBackground ? '' : ' artwork--checker'
        }`}
        onPointerDown={(event) => {
          setDragging(true)
          ;(event.target as HTMLElement).setPointerCapture?.(event.pointerId)
          origin.current = { x: event.clientX, y: event.clientY, viewX: view.x, viewY: view.y }
        }}
        onPointerMove={(event) => {
          if (!dragging) return
          setView((current) => ({
            ...current,
            x: origin.current.viewX + (event.clientX - origin.current.x),
            y: origin.current.viewY + (event.clientY - origin.current.y),
          }))
        }}
        onPointerUp={() => setDragging(false)}
        onPointerLeave={() => setDragging(false)}
      >
        <div
          ref={stage}
          className="artwork__stage"
          style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})` }}
        >
          {(() => {
            const top =
              tab === 'landmarks' ? (
                landmarkFile && !('error' in landmarkFile) ? (
                  <LandmarkOverlay file={landmarkFile} imageUrl={available.input} />
                ) : null
              ) : tab === 'result' && svgMarkup ? (
                <div
                  // eslint-disable-next-line react/no-danger -- our own API, our own file
                  dangerouslySetInnerHTML={{ __html: overlaid ? withViewBox(svgMarkup) : svgMarkup }}
                  style={
                    strokeBoost
                      ? ({ ['--boost' as string]: '1', strokeWidth: 2 } as React.CSSProperties)
                      : undefined
                  }
                  className={strokeBoost ? 'svg-boost' : undefined}
                />
              ) : url ? (
                <img
                  src={url}
                  alt={TAB_LABELS[tab]}
                  // An image has no size until it decodes, so both the opening fit
                  // and the framing carried over from the last tab wait for this
                  // rather than for React.
                  onLoad={syncContent}
                />
              ) : null
            if (!overlaid || !top || !available.input) return top
            // The input sets the size; the tab is stretched over it (every
            // canvas-space artefact is drawn at the input's size or a multiple).
            return (
              <div className="artwork__stack">
                <img src={available.input} alt="Input" onLoad={syncContent} />
                <div
                  className="artwork__overlay"
                  style={overlayStyle(overlayOpacity, overlayBlend)}
                >
                  {top}
                </div>
              </div>
            )
          })()}
        </div>

        {tab === 'landmarks' && landmarkFile && 'error' in landmarkFile && (
          <div className="empty" style={{ position: 'absolute', inset: 0 }}>
            <span className="muted">The landmark file could not be read: {landmarkFile.error}</span>
          </div>
        )}

        {!url && (
          <div className="empty" style={{ position: 'absolute', inset: 0 }}>
            <span className="muted">
              {tab === 'result'
                ? 'No final SVG yet — this job has not reached its horizon.'
                : `No ${TAB_LABELS[tab].toLowerCase()} for this job.`}
            </span>
          </div>
        )}

        <div className="artwork__hud mono">
          {tab === 'preview' && frameEpoch !== null && <span>epoch {frameEpoch}</span>}
          <span>{Math.round(view.scale * 100)}%</span>
          <button
            type="button"
            className="btn btn--small btn--ghost"
            onClick={() => step(1 / ZOOM_STEP)}
            title="Zoom out 10%"
            aria-label="Zoom out"
          >
            −
          </button>
          <button
            type="button"
            className="btn btn--small btn--ghost"
            onClick={() => step(ZOOM_STEP)}
            title="Zoom in 10%"
            aria-label="Zoom in"
          >
            +
          </button>
          <button
            type="button"
            className="btn btn--small btn--ghost"
            onClick={() => fit()}
            title="Fit the whole drawing in the frame"
          >
            fit
          </button>
          <button
            type="button"
            className="btn btn--small btn--ghost"
            onClick={actualSize}
            title="Actual size (one SVG unit per screen pixel)"
          >
            100%
          </button>
          <button
            type="button"
            className="btn btn--small btn--ghost"
            aria-pressed={!paperBackground}
            onClick={() => setPaperBackground((value) => !value)}
            title="White / checkerboard background"
          >
            bg
          </button>
          {canOverlay(tab, Boolean(available.input)) && (
            <button
              type="button"
              className="btn btn--small btn--ghost"
              aria-pressed={overlay}
              onClick={() => setOverlay((value) => !value)}
              title="Draw this over the input, to see how well it lines up"
            >
              overlay
            </button>
          )}
          {overlaid && (
            <>
              <input
                type="range"
                min={0.1}
                max={1}
                step={0.05}
                value={overlayOpacity}
                onChange={(event) => setOverlayOpacity(Number(event.target.value))}
                onPointerDown={(event) => event.stopPropagation()}
                aria-label="Overlay opacity"
                title={`Overlay opacity ${Math.round(overlayOpacity * 100)}%`}
                style={{ width: 70 }}
              />
              <span>{Math.round(overlayOpacity * 100)}%</span>
              <button
                type="button"
                className="btn btn--small btn--ghost"
                aria-pressed={overlayBlend === 'multiply'}
                onClick={() => setOverlayBlend((value) => (value === 'multiply' ? 'normal' : 'multiply'))}
                title="Multiply: white turns transparent, so only the lines (or dark areas) sit on the input"
              >
                multiply
              </button>
            </>
          )}
          {tab === 'result' && (
            <button
              type="button"
              className="btn btn--small btn--ghost"
              aria-pressed={strokeBoost}
              onClick={() => setStrokeBoost((value) => !value)}
              title="Thicken strokes for legibility at small zoom"
            >
              stroke
            </button>
          )}
        </div>
      </div>

      {tab === 'landmarks' && landmarkFile && !('error' in landmarkFile) && (
        <div className="panel__body mono" style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <span>{landmarkFile.points.length} landmarks</span>
          {landmarkFile.lines.length > 0 && <span>{landmarkFile.lines.length} lines</span>}
          <span className="note">Dot size is weight; hover a dot for its name.</span>
          {available.landmarks && (
            <a href={available.landmarks} download>
              download JSON
            </a>
          )}
        </div>
      )}

      {tab === 'result' && svgStats && (
        <div className="panel__body mono" style={{ display: 'flex', gap: 16 }}>
          <span>path length {Math.round(svgStats.length).toLocaleString()} units</span>
          <span>{svgStats.segments} segments</span>
          {available.result && (
            <a href={available.result} download>
              download SVG
            </a>
          )}
        </div>
      )}

      {tab === 'result' && svgMarkup === null && available.result === null && (
        <div className="panel__body note">
          The final SVG appears when the job reaches its horizon. Until then the Preview tab shows
          the most recent frame.
        </div>
      )}

      {job.artifacts.some((artifact) => artifact.name === 'final_sld.png') && tab === 'result' && (
        <div className="panel__body note" style={{ paddingTop: 0 }}>
          Also available:{' '}
          <a href={fileUrl(job.id, 'target/run/final_sld.png')} download>
            final PNG ·{' '}
            {formatBytes(
              job.artifacts.find((artifact) => artifact.name === 'final_sld.png')?.bytes ?? null,
            )}
          </a>
        </div>
      )}
    </div>
  )
}

type LandmarkFile =
  | { points: EditorPoint[]; lines: EditorLine[]; imageSize: [number, number] }
  | { error: string }

/**
 * A job's landmark file on the canvas it was placed on: the input under the
 * points and lines, at the canvas's own pixel size (so the stage measures it
 * like an image and the shared view carries over from the other tabs).
 */
function LandmarkOverlay({
  file,
  imageUrl,
}: {
  file: { points: EditorPoint[]; lines: EditorLine[]; imageSize: [number, number] }
  imageUrl: string | null
}) {
  const [width, height] = file.imageSize
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      style={{ display: 'block' }}
      role="img"
      aria-label={`${file.points.length} landmarks`}
    >
      {imageUrl ? (
        <image href={imageUrl} x={0} y={0} width={width} height={height} />
      ) : (
        <rect width={width} height={height} fill="var(--paper)" />
      )}
      {file.lines.filter(lineIsValid).map((line) => (
        <path
          key={line.id}
          className="lm-line lm-line--view"
          d={`M${line.xy.map(([x, y]) => `${x},${y}`).join('L')}${line.closed && line.xy.length > 2 ? 'Z' : ''}`}
          strokeWidth={1.5}
        >
          <title>{`${line.name} · weight ${line.weight}`}</title>
        </path>
      ))}
      {file.points.map((point) =>
        point.xy ? (
          <circle
            key={point.id}
            className={`lm-dot lm-dot--${point.source}${point.edited ? ' lm-dot--edited' : ''}`}
            cx={point.xy[0]}
            cy={point.xy[1]}
            r={dotRadius(point.weight)}
            strokeWidth={1}
          >
            <title>{`${point.name} · weight ${point.weight} · ${point.source}`}</title>
          </circle>
        ) : null,
      )}
    </svg>
  )
}

/**
 * Total drawn length, summed over every path in the SVG.
 *
 * Measured with the browser's own `getTotalLength`, so curves are measured
 * rather than approximated by their control polygon. Spec 3 SS17.3 asks whether
 * to show an estimated pen travel *time*; that needs a speed constant nobody
 * has established, so this reports length only and leaves the plotter to the
 * plotter.
 */
function measureSvg(markup: string): { length: number; segments: number } | null {
  try {
    const parsed = new DOMParser().parseFromString(markup, 'image/svg+xml')
    const paths = Array.from(parsed.querySelectorAll('path'))
    if (paths.length === 0) return null
    // getTotalLength needs the node to be in a rendered document.
    const holder = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    holder.setAttribute('style', 'position:absolute;width:0;height:0;overflow:hidden')
    document.body.appendChild(holder)
    let length = 0
    for (const path of paths) {
      const clone = path.cloneNode(false) as SVGPathElement
      holder.appendChild(clone)
      length += clone.getTotalLength()
    }
    document.body.removeChild(holder)
    return { length, segments: paths.length }
  } catch {
    return null
  }
}
