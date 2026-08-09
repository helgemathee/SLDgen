import { useEffect, useRef, useState } from 'react'
import { fileUrl } from '../api/client'
import type { JobDetail } from '../api/types'
import { formatBytes } from '../lib/format'

export type ArtworkTab = 'result' | 'preview' | 'input' | 'mask' | 'condition'

interface Available {
  result: string | null
  preview: string | null
  input: string | null
  mask: string | null
  condition: string | null
}

export function availableArtwork(job: JobDetail, frameUrl: string | null): Available {
  const has = (path: string) => job.artifacts.some((artifact) => artifact.path === path)
  const run = (name: string) => (has(`target/run/${name}`) ? fileUrl(job.id, `target/run/${name}`) : null)
  const condition = job.artifacts.find((artifact) =>
    /^target\/run\/condition_\w+\.png$/.test(artifact.path),
  )
  return {
    result: run('final_sld.svg'),
    preview: frameUrl ?? (job.current_epoch > 0 ? job.preview_url : null),
    input: run('input.png'),
    mask: run('mask.png'),
    condition: condition ? fileUrl(job.id, condition.path) : null,
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
}

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
  const container = useRef<HTMLDivElement>(null)
  const origin = useRef({ x: 0, y: 0, viewX: 0, viewY: 0 })

  const url = available[tab]

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

  const onWheel = (event: React.WheelEvent) => {
    event.preventDefault()
    const rect = container.current?.getBoundingClientRect()
    if (!rect) return
    const pointerX = event.clientX - rect.left
    const pointerY = event.clientY - rect.top
    const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15
    setView((current) => {
      const scale = Math.min(24, Math.max(0.1, current.scale * factor))
      const ratio = scale / current.scale
      return {
        scale,
        // Keep the pixel under the cursor fixed; anything else makes zooming in
        // on a detail a hunting exercise.
        x: pointerX - (pointerX - current.x) * ratio,
        y: pointerY - (pointerY - current.y) * ratio,
      }
    })
  }

  return (
    <div className="panel">
      <div className="panel__head" style={{ padding: '4px 8px 0', background: 'var(--paper-sunk)' }}>
        <div className="tabs" role="tablist">
          {(Object.keys(TAB_LABELS) as ArtworkTab[]).map((name) => (
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
        onWheel={onWheel}
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
          className="artwork__stage"
          style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})` }}
        >
          {tab === 'result' && svgMarkup ? (
            <div
              // eslint-disable-next-line react/no-danger -- our own API, our own file
              dangerouslySetInnerHTML={{ __html: svgMarkup }}
              style={
                strokeBoost
                  ? ({ ['--boost' as string]: '1', strokeWidth: 2 } as React.CSSProperties)
                  : undefined
              }
              className={strokeBoost ? 'svg-boost' : undefined}
            />
          ) : url ? (
            <img src={url} alt={TAB_LABELS[tab]} />
          ) : null}
        </div>

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
            onClick={() => setView({ scale: 1, x: 0, y: 0 })}
          >
            reset
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
