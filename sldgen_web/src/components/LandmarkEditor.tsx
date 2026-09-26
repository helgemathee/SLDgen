import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { CanvasInfo, LandmarkExtract } from '../api/types'
import type { InputRef } from '../lib/formstate'
import { UI_HELP } from '../lib/help'
import {
  LANDMARK_SETS,
  WEIGHT_MAX,
  boxFrom,
  clampWeight,
  describePose,
  dotRadius,
  fitView,
  fromDetected,
  fromDetectedLines,
  glassesTemplate,
  insertVertex,
  lineCount,
  lineHint,
  lineIsValid,
  mergeDetected,
  mergeLines,
  newLine,
  newId,
  nextName,
  panBy,
  parseFile,
  placedCount,
  placementHint,
  REFERENCES,
  referenceFor,
  savedLabel,
  serialize,
  withTemplate,
  zoomAbout,
  zoomLevel,
  type EditorLine,
  type EditorPoint,
  type PoseReport,
  type ViewBox,
} from '../lib/landmarks'

type Mode = 'select' | 'box'

type Drag =
  | { kind: 'move'; id: string }
  | { kind: 'vertex'; id: string; index: number }
  | { kind: 'pan'; startX: number; startY: number; view: ViewBox; moved: boolean }
  | { kind: 'box'; from: [number, number]; to: [number, number] }

/**
 * The landmark editor (Spec 7 SS6): the canvas image with the points on it,
 * zoom and pan, and a table beside it. Every change is saved as a canvas-space
 * JSON upload and attached as the `image_loss_landmarks` input, so what the
 * run reads is exactly what is on screen.
 */
export function LandmarkEditor({
  targetSha256,
  attachedSha256,
  edgeUrl,
  termWeight,
  renderSize,
  fallbackCanvas,
  onAttach,
  onPending,
  onEnableTerm,
}: {
  targetSha256: string | null
  /** The upload currently attached as the landmark input, if any. */
  attachedSha256: string | null
  /** The edge map preview, shown under the points on request. */
  edgeUrl: string | null
  termWeight: number
  renderSize: number
  /** The edge preview's canvas, used if the canvas endpoint is unavailable. */
  fallbackCanvas: CanvasInfo | null
  onAttach: (reference: InputRef | null) => void
  onPending: (pending: boolean) => void
  onEnableTerm: () => void
}) {
  const [canvas, setCanvas] = useState<CanvasInfo | null>(null)
  const [canvasProblem, setCanvasProblem] = useState<string | null>(null)
  const [points, setPoints] = useState<EditorPoint[]>([])
  const [lines, setLines] = useState<EditorLine[]>([])
  const [pose, setPose] = useState<PoseReport | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  /** The selected polyline, and one of its vertices (Spec 7 addendum SS8). */
  const [selectedLine, setSelectedLine] = useState<string | null>(null)
  const [selectedVertex, setSelectedVertex] = useState<number | null>(null)
  const [landmarkSet, setLandmarkSet] = useState('sparse')
  const [hairline, setHairline] = useState(false)
  const [view, setView] = useState<ViewBox | null>(null)
  const [mode, setMode] = useState<Mode>('select')
  const [drag, setDrag] = useState<Drag | null>(null)
  const [showEdges, setShowEdges] = useState(false)
  const [detecting, setDetecting] = useState(false)
  const [status, setStatus] = useState<string | null>(null)
  const [problem, setProblem] = useState<string | null>(null)
  const svg = useRef<SVGSVGElement>(null)
  const jsonInput = useRef<HTMLInputElement>(null)
  /** The sha256 the table was loaded from or last saved as: no re-upload of it. */
  const savedSha = useRef<string | null>(null)
  const dirty = useRef(false)
  /** The latest save: an older upload finishing late must not attach stale content. */
  const saveSeq = useRef(0)
  /** The last pointerdown landed on a dot or square: its double-click adds nothing. */
  const downOnHandle = useRef(false)

  const imageSize: [number, number] = canvas?.image_size ?? [512, 512]

  // The canvas: the latest run of this image.
  useEffect(() => {
    if (!targetSha256) return
    let live = true
    api
      .imageLossCanvas(targetSha256)
      .then((info) => {
        if (!live) return
        setCanvas(info)
        setView(fitView(info.image_size))
        setCanvasProblem(null)
      })
      .catch((error) => {
        if (!live) return
        setCanvas(null)
        setCanvasProblem(error instanceof Error ? error.message : 'No canvas for this image yet')
      })
    return () => {
      live = false
    }
  }, [targetSha256])

  // An API that predates /api/image-loss/canvas still serves the edge preview,
  // which knows the same canvas.
  useEffect(() => {
    if (canvas || !canvasProblem || !fallbackCanvas) return
    setCanvas(fallbackCanvas)
    setView(fitView(fallbackCanvas.image_size))
    setCanvasProblem(null)
  }, [canvas, canvasProblem, fallbackCanvas])

  // An attached file (a restored form, an upload) becomes the table.
  useEffect(() => {
    if (!attachedSha256 || attachedSha256 === savedSha.current) return
    let live = true
    api
      .uploadJson<unknown>(attachedSha256)
      .then((raw) => {
        if (!live) return
        const parsed = parseFile(typeof raw === 'string' ? JSON.parse(raw) : raw)
        if (typeof parsed === 'string') {
          setProblem(`The attached landmark file: ${parsed}.`)
          return
        }
        savedSha.current = attachedSha256
        dirty.current = false
        setPoints(parsed.points)
        setLines(parsed.lines)
        setPose(parsed.pose)
      })
      .catch((error) => live && setProblem(error instanceof Error ? error.message : String(error)))
    return () => {
      live = false
    }
  }, [attachedSha256])

  // Save: debounced, content-addressed, attached.
  useEffect(() => {
    if (!dirty.current) return
    onPending(true)
    const timer = window.setTimeout(async () => {
      saveSeq.current += 1
      const seq = saveSeq.current
      try {
        if (placedCount(points) + lineCount(lines) === 0) {
          savedSha.current = null
          onAttach(null)
          return
        }
        const body = JSON.stringify(serialize(points, imageSize, { lines, pose }))
        const result = await api.upload(new Blob([body], { type: 'application/json' }), 'landmarks.json')
        if (seq !== saveSeq.current) return // a newer save owns the attachment
        savedSha.current = result.sha256
        onAttach({
          source_kind: 'upload',
          sha256: result.sha256,
          label: savedLabel(points, lines),
        })
        setProblem(null)
      } catch (error) {
        if (seq === saveSeq.current)
          setProblem(error instanceof Error ? error.message : 'Could not save the landmarks')
      } finally {
        if (seq === saveSeq.current) onPending(false)
      }
    }, 400)
    return () => window.clearTimeout(timer)
  }, [points, lines]) // eslint-disable-line react-hooks/exhaustive-deps

  // Never leave the form blocked on a save that can no longer happen.
  useEffect(() => () => onPending(false), []) // eslint-disable-line react-hooks/exhaustive-deps

  const change = useCallback((next: EditorPoint[] | ((current: EditorPoint[]) => EditorPoint[])) => {
    dirty.current = true
    setPoints(next)
  }, [])

  const update = (id: string, patch: Partial<EditorPoint>) =>
    change((current) =>
      current.map((point) => (point.id === id ? { ...point, ...patch, edited: true } : point)),
    )

  const remove = (id: string) => {
    change((current) => current.filter((point) => point.id !== id))
    if (selected === id) setSelected(null)
  }

  const changeLines = useCallback(
    (next: EditorLine[] | ((current: EditorLine[]) => EditorLine[])) => {
      dirty.current = true
      setLines(next)
    },
    [],
  )

  const updateLine = (id: string, patch: Partial<EditorLine>) =>
    changeLines((current) =>
      current.map((line) => (line.id === id ? { ...line, ...patch, edited: true } : line)),
    )

  const removeLine = (id: string) => {
    changeLines((current) => current.filter((line) => line.id !== id))
    if (selectedLine === id) {
      setSelectedLine(null)
      setSelectedVertex(null)
    }
  }

  /** Select a point (or nothing), dropping any line selection. */
  const selectPoint = (id: string | null) => {
    setSelected(id)
    setSelectedLine(null)
    setSelectedVertex(null)
  }

  const selectLine = (id: string | null, vertex: number | null = null) => {
    setSelectedLine(id)
    setSelectedVertex(vertex)
    setSelected(null)
  }

  // -- detection ------------------------------------------------------------

  const detect = async (how: 'fill' | 'replace', box?: [number, number, number, number]) => {
    if (!targetSha256) return
    setDetecting(true)
    setProblem(null)
    try {
      const result: LandmarkExtract = await api.extractLandmarks({
        target_sha256: targetSha256,
        preset: 'portrait',
        landmark_set: landmarkSet,
        include_hairline: hairline,
        pose_report: true,
        ...(box ? { box } : {}),
      })
      const where = box ? ' in the box' : ''
      const seen = describeView(result)
      const detectedLines = result.polylines ?? []
      setPose(result.pose ?? null)
      const extra = `${poseNote(result)}${setNote(result, landmarkSet)}${hairlineNote(result, hairline)}${droppedNote(result)}`
      if (how === 'replace') {
        change(fromDetected(result.landmarks))
        changeLines(fromDetectedLines(detectedLines))
        selectPoint(null)
        setStatus(`${seen}${where}: ${result.landmarks.length} points.${extra}`)
      } else {
        const report = mergeDetected(points, result.landmarks)
        const lineReport = mergeLines(lines, detectedLines)
        change(report.points)
        changeLines(lineReport.lines)
        // A merged line may have changed length or gone: keep the selection valid.
        const still = lineReport.lines.find((entry) => entry.id === selectedLine)
        if (!still) selectLine(null)
        else if (selectedVertex !== null && selectedVertex >= still.xy.length) setSelectedVertex(null)
        setStatus(
          `${seen}${where}: ${report.added} added, ${report.updated} updated, ${report.kept} of yours kept` +
            `${report.removed ? `, ${report.removed} removed as hidden` : ''}.${extra}`,
        )
      }
    } catch (error) {
      setProblem(error instanceof Error ? error.message : 'Detection failed')
    } finally {
      setDetecting(false)
    }
  }

  const receiveJson = async (file: File) => {
    try {
      const parsed = parseFile(JSON.parse(await file.text()))
      if (typeof parsed === 'string') throw new Error(parsed)
      if (parsed.imageSize[0] !== imageSize[0] || parsed.imageSize[1] !== imageSize[1])
        throw new Error(
          `the file was made at ${parsed.imageSize.join('×')}, this canvas is ${imageSize.join('×')}`,
        )
      change(parsed.points)
      changeLines(parsed.lines)
      setPose(parsed.pose)
      selectPoint(null)
      setStatus(
        `Loaded ${parsed.points.length} points${parsed.lines.length ? ` and ${parsed.lines.length} lines` : ''} from ${file.name}.`,
      )
    } catch (error) {
      setProblem(`${file.name}: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      if (jsonInput.current) jsonInput.current.value = ''
    }
  }

  // -- pointer and wheel -----------------------------------------------------

  const toCanvas = (clientX: number, clientY: number): [number, number] => {
    const element = svg.current
    const matrix = element?.getScreenCTM()
    if (!element || !matrix) return [0, 0]
    const point = element.createSVGPoint()
    point.x = clientX
    point.y = clientY
    const local = point.matrixTransform(matrix.inverse())
    return [local.x, local.y]
  }

  // React's wheel listener is passive; zooming must prevent the page scroll.
  useEffect(() => {
    const element = svg.current
    if (!element) return
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const at = toCanvas(event.clientX, event.clientY)
      setView((current) =>
        current ? zoomAbout(current, event.deltaY < 0 ? 1.25 : 0.8, at, imageSize) : current,
      )
    }
    element.addEventListener('wheel', onWheel, { passive: false })
    return () => element.removeEventListener('wheel', onWheel)
  }) // re-bound each render: it closes over imageSize

  const onBackgroundDown = (event: React.PointerEvent) => {
    downOnHandle.current = false
    if (!view) return
    svg.current?.setPointerCapture(event.pointerId)
    const at = toCanvas(event.clientX, event.clientY)
    const target = points.find((point) => point.id === selected)
    if (mode === 'box') {
      setDrag({ kind: 'box', from: at, to: at })
    } else if (target && target.xy === null) {
      update(target.id, { xy: at })
    } else {
      setDrag({ kind: 'pan', startX: event.clientX, startY: event.clientY, view, moved: false })
    }
  }

  const onPointDown = (event: React.PointerEvent, id: string) => {
    event.stopPropagation()
    if (mode === 'box') return onBackgroundDown(event)
    downOnHandle.current = true
    svg.current?.setPointerCapture(event.pointerId)
    selectPoint(id)
    setDrag({ kind: 'move', id })
  }

  const onVertexDown = (event: React.PointerEvent, id: string, index: number) => {
    event.stopPropagation()
    if (mode === 'box') return onBackgroundDown(event)
    downOnHandle.current = true
    svg.current?.setPointerCapture(event.pointerId)
    selectLine(id, index)
    setDrag({ kind: 'vertex', id, index })
  }

  const onLineDown = (event: React.PointerEvent, id: string) => {
    if (mode === 'box') return
    event.stopPropagation()
    selectLine(id)
  }

  const onMove = (event: React.PointerEvent) => {
    if (!drag || !view) return
    if (drag.kind === 'move') {
      const [x, y] = toCanvas(event.clientX, event.clientY)
      update(drag.id, {
        xy: [
          Math.min(imageSize[0], Math.max(0, x)),
          Math.min(imageSize[1], Math.max(0, y)),
        ],
      })
    } else if (drag.kind === 'vertex') {
      const [x, y] = toCanvas(event.clientX, event.clientY)
      const at: [number, number] = [
        Math.min(imageSize[0], Math.max(0, x)),
        Math.min(imageSize[1], Math.max(0, y)),
      ]
      const line = lines.find((entry) => entry.id === drag.id)
      if (line)
        updateLine(line.id, { xy: line.xy.map((vertex, k) => (k === drag.index ? at : vertex)) })
    } else if (drag.kind === 'pan') {
      const rect = svg.current!.getBoundingClientRect()
      const dx = ((event.clientX - drag.startX) * drag.view.w) / rect.width
      const dy = ((event.clientY - drag.startY) * drag.view.h) / rect.height
      if (Math.abs(event.clientX - drag.startX) + Math.abs(event.clientY - drag.startY) > 3)
        setDrag({ ...drag, moved: true })
      setView(panBy(drag.view, dx, dy, imageSize))
    } else {
      setDrag({ ...drag, to: toCanvas(event.clientX, event.clientY) })
    }
  }

  const onUp = () => {
    // A plain click on empty canvas deselects a point. A selected line stays
    // selected (Esc ends it): double-clicking to add its vertices starts with
    // two such clicks.
    if (drag?.kind === 'pan' && !drag.moved && !selectedLine) selectPoint(null)
    if (drag?.kind === 'box') {
      const box = boxFrom(drag.from, drag.to, imageSize)
      setMode('select')
      if (box[2] - box[0] > 8 && box[3] - box[1] > 8) detect('fill', box)
    }
    setDrag(null)
  }

  const onDoubleClick = (event: React.MouseEvent) => {
    // Double-clicking a dot or a square grabs it; it must not add a point or
    // vertex on top of it (the dblclick reaches the svg despite stopPropagation
    // on pointerdown, and pointer capture retargets it there anyway).
    if (mode !== 'select' || downOnHandle.current) return
    const at = toCanvas(event.clientX, event.clientY)
    const line = lines.find((entry) => entry.id === selectedLine)
    if (line) {
      const inserted = insertVertex(line.xy, line.closed, at)
      updateLine(line.id, { xy: inserted.xy })
      setSelectedVertex(inserted.index)
      return
    }
    const point: EditorPoint = {
      id: newId(),
      name: nextName(points),
      xy: at,
      weight: 1,
      source: 'manual',
      edited: true,
    }
    change((current) => [...current, point])
    selectPoint(point.id)
  }

  const onKeyDown = (event: React.KeyboardEvent) => {
    if ((event.target as HTMLElement).closest('input, select, textarea, button, [contenteditable]'))
      return
    const target = points.find((point) => point.id === selected)
    if (event.key === 'Escape') {
      selectPoint(null)
      setMode('select')
      return
    }
    const line = lines.find((entry) => entry.id === selectedLine)
    if (line) {
      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault()
        if (selectedVertex === null) {
          removeLine(line.id)
        } else if (line.xy[selectedVertex]) {
          const rest = line.xy.filter((_vertex, k) => k !== selectedVertex)
          updateLine(line.id, { xy: rest })
          // Select a neighbour, so repeated Delete keeps removing vertices
          // rather than falling through to the whole line.
          setSelectedVertex(rest.length ? Math.min(selectedVertex, rest.length - 1) : null)
        } else {
          setSelectedVertex(null)
        }
        return
      }
      const step = event.shiftKey ? 5 : 0.5
      const move: Record<string, [number, number]> = {
        ArrowLeft: [-step, 0],
        ArrowRight: [step, 0],
        ArrowUp: [0, -step],
        ArrowDown: [0, step],
      }
      if (move[event.key] && selectedVertex !== null && line.xy[selectedVertex]) {
        event.preventDefault()
        const [dx, dy] = move[event.key]
        updateLine(line.id, {
          xy: line.xy.map((vertex, k) =>
            k === selectedVertex ? ([vertex[0] + dx, vertex[1] + dy] as [number, number]) : vertex,
          ),
        })
      }
      return
    }
    if (!target) return
    if (event.key === 'Delete' || event.key === 'Backspace') {
      event.preventDefault()
      remove(target.id)
      return
    }
    const step = event.shiftKey ? 5 : 0.5
    const nudge: Record<string, [number, number]> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    }
    if (nudge[event.key] && target.xy) {
      event.preventDefault()
      update(target.id, { xy: [target.xy[0] + nudge[event.key][0], target.xy[1] + nudge[event.key][1]] })
    }
  }

  // -- render -----------------------------------------------------------------

  if (!targetSha256) return <div className="note">Choose a source image first.</div>

  const zoom = view ? zoomLevel(view, imageSize) : 1
  const scale = view ? view.w / imageSize[0] : 1 // canvas px per fit px: dots keep their screen size
  const placing = points.find((point) => point.id === selected && point.xy === null)
  const chosen = points.find((point) => point.id === selected) ?? null
  const usable = placedCount(points) + lineCount(lines)
  const chosenLine = lines.find((line) => line.id === selectedLine) ?? null
  const setHelp = LANDMARK_SETS.find((entry) => entry.value === landmarkSet)?.help ?? ''

  return (
    <div className="lm-editor" onKeyDown={onKeyDown} tabIndex={-1}>
      <div className="btn-row" style={{ flexWrap: 'wrap' }}>
        <label className="note" title={`${UI_HELP.lmSet}\n\n${setHelp}`}>
          set{' '}
          <select
            value={landmarkSet}
            aria-label="Landmark set"
            onChange={(event) => setLandmarkSet(event.target.value)}
          >
            {LANDMARK_SETS.map((entry) => (
              <option key={entry.value} value={entry.value} title={entry.help}>
                {entry.label}
              </option>
            ))}
          </select>
        </label>
        <label className="note" title={UI_HELP.lmHairline}>
          <input
            type="checkbox"
            checked={hairline}
            onChange={(event) => setHairline(event.target.checked)}
          />{' '}
          hairline
        </label>
        <button
          type="button"
          className="btn btn--small"
          disabled={!canvas || detecting}
          title={UI_HELP.lmAutoDetect}
          onClick={() => detect('fill')}
        >
          {detecting ? 'Detecting…' : 'Auto-detect'}
        </button>
        <button
          type="button"
          className="btn btn--small"
          disabled={!canvas || detecting}
          title={UI_HELP.lmReplace}
          onClick={() => detect('replace')}
        >
          Replace all
        </button>
        <button
          type="button"
          className="btn btn--small"
          aria-pressed={mode === 'box'}
          disabled={!canvas || detecting}
          title={UI_HELP.lmBox}
          onClick={() => setMode(mode === 'box' ? 'select' : 'box')}
        >
          Detect in box
        </button>
        <button
          type="button"
          className="btn btn--small"
          title={UI_HELP.lmTemplate}
          onClick={() => change(withTemplate)}
        >
          Profile template
        </button>
        <button
          type="button"
          className="btn btn--small"
          title={UI_HELP.lmGlasses}
          disabled={!canvas}
          onClick={() => changeLines((current) => glassesTemplate(points, current, imageSize))}
        >
          Glasses
        </button>
        <button
          type="button"
          className="btn btn--small"
          title={UI_HELP.lmLine}
          disabled={!canvas}
          onClick={() => {
            const line = newLine(lines)
            changeLines((current) => [...current, line])
            selectLine(line.id)
          }}
        >
          Line
        </button>
        <button
          type="button"
          className="btn btn--small"
          title={UI_HELP.lmUpload}
          disabled={!canvas}
          onClick={() => jsonInput.current?.click()}
        >
          Upload JSON…
        </button>
        <input
          ref={jsonInput}
          type="file"
          accept=".json,application/json"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0]
            if (file) receiveJson(file)
          }}
        />
        <button
          type="button"
          className="btn btn--small"
          disabled={points.length === 0 && lines.length === 0}
          title={UI_HELP.lmClear}
          onClick={() => {
            change([])
            changeLines([])
            setPose(null)
            selectPoint(null)
            setStatus(null)
          }}
        >
          Clear
        </button>
      </div>

      {canvasProblem && (
        <div className="warn">
          {canvasProblem} The editor places points on a previous run&rsquo;s canvas: run this image
          once (a short budget is enough), then come back.
        </div>
      )}
      {canvas && renderSize !== imageSize[0] && (
        <div className="warn">
          This canvas is {imageSize.join('×')} but the render size is {renderSize}: the run will
          refuse these landmarks. Use render size {imageSize[0]}, or run the image at {renderSize}{' '}
          first.
        </div>
      )}

      {canvas && view && (
        <div className="lm-editor__body">
          <div className="lm-editor__stage">
            <svg
              ref={svg}
              className={`lm-editor__svg${mode === 'box' ? ' lm-editor__svg--box' : ''}${placing || chosenLine ? ' lm-editor__svg--place' : ''}`}
              viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
              onPointerDown={onBackgroundDown}
              onPointerMove={onMove}
              onPointerUp={onUp}
              onDoubleClick={onDoubleClick}
              role="img"
              aria-label="Landmark canvas"
            >
              <image href={canvas.image_url} x={0} y={0} width={imageSize[0]} height={imageSize[1]} />
              {showEdges && edgeUrl && (
                <image
                  href={edgeUrl}
                  x={0}
                  y={0}
                  width={imageSize[0]}
                  height={imageSize[1]}
                  style={{ mixBlendMode: 'multiply', opacity: 0.6, filter: 'invert(1)' }}
                />
              )}
              {lines.map((line) => (
                <g key={line.id}>
                  {line.xy.length >= 2 && (
                    <path
                      className={`lm-line lm-line--${line.source === 'manual' || line.edited ? 'manual' : 'detected'}${line.id === selectedLine ? ' lm-line--selected' : ''}`}
                      d={`M${line.xy.map(([x, y]) => `${x},${y}`).join('L')}${line.closed && line.xy.length > 2 ? 'Z' : ''}`}
                      strokeWidth={(line.id === selectedLine ? 2.5 : 1.5) * scale}
                      onPointerDown={(event) => onLineDown(event, line.id)}
                    >
                      <title>{`${line.name} · weight ${line.weight} · ${line.xy.length} vertices\n\n${lineHint(line.name)}`}</title>
                    </path>
                  )}
                  {(line.id === selectedLine || line.xy.length < 2) &&
                    line.xy.map(([x, y], index) => (
                      <rect
                        key={index}
                        className={`lm-vertex${line.id === selectedLine && index === selectedVertex ? ' lm-vertex--selected' : ''}`}
                        x={x - 3 * scale}
                        y={y - 3 * scale}
                        width={6 * scale}
                        height={6 * scale}
                        strokeWidth={1.2 * scale}
                        onPointerDown={(event) => onVertexDown(event, line.id, index)}
                      />
                    ))}
                </g>
              ))}
              {points.map((point) =>
                point.xy ? (
                  <g key={point.id}>
                    <circle
                      className={`lm-dot lm-dot--${point.source}${point.edited ? ' lm-dot--edited' : ''}${point.id === selected ? ' lm-dot--selected' : ''}`}
                      cx={point.xy[0]}
                      cy={point.xy[1]}
                      r={dotRadius(point.weight) * scale}
                      strokeWidth={1.5 * scale}
                      onPointerDown={(event) => onPointDown(event, point.id)}
                    >
                      <title>
                        {`${point.name} · weight ${point.weight} · ${point.source}\n\n${placementHint(point.name)}`}
                      </title>
                    </circle>
                    {point.id === selected && (
                      <text
                        x={point.xy[0] + (dotRadius(point.weight) + 3) * scale}
                        y={point.xy[1] - (dotRadius(point.weight) + 3) * scale}
                        fontSize={11 * scale}
                        className="lm-label"
                      >
                        {point.name}
                      </text>
                    )}
                  </g>
                ) : null,
              )}
              {drag?.kind === 'box' && (
                <rect
                  className="lm-box"
                  x={Math.min(drag.from[0], drag.to[0])}
                  y={Math.min(drag.from[1], drag.to[1])}
                  width={Math.abs(drag.to[0] - drag.from[0])}
                  height={Math.abs(drag.to[1] - drag.from[1])}
                  strokeWidth={1.5 * scale}
                />
              )}
            </svg>
            <div className="lm-editor__zoom">
              <button
                type="button"
                className="btn btn--small"
                title="Zoom out"
                onClick={() => setView(zoomAbout(view, 0.8, center(view), imageSize))}
              >
                −
              </button>
              <span className="mono">{zoom.toFixed(1)}×</span>
              <button
                type="button"
                className="btn btn--small"
                title="Zoom in"
                onClick={() => setView(zoomAbout(view, 1.25, center(view), imageSize))}
              >
                +
              </button>
              <button
                type="button"
                className="btn btn--small"
                title="Show the whole canvas"
                onClick={() => setView(fitView(imageSize))}
              >
                Fit
              </button>
              {edgeUrl && (
                <label className="note" title={UI_HELP.lmShowEdges}>
                  <input
                    type="checkbox"
                    checked={showEdges}
                    onChange={(event) => setShowEdges(event.target.checked)}
                  />{' '}
                  edges
                </label>
              )}
            </div>
          </div>

          <div className="lm-editor__table">
            <table className="mono">
              <thead>
                <tr>
                  <th title={UI_HELP.lmName}>name</th>
                  <th title={UI_HELP.lmWeight}>weight</th>
                  <th title={UI_HELP.lmXY}>x</th>
                  <th title={UI_HELP.lmXY}>y</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {points.map((point) => (
                  <tr
                    key={point.id}
                    aria-selected={point.id === selected}
                    onClick={() => selectPoint(point.id)}
                    title={`${placementHint(point.name)}\n\n(${point.source}${point.edited ? ', edited' : ''})`}
                  >
                    <td>
                      <span className={`lm-swatch lm-dot--${point.source}`} />
                      <input
                        type="text"
                        value={point.name}
                        aria-label="Landmark name"
                        onChange={(event) => update(point.id, { name: event.target.value })}
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        min={0}
                        max={WEIGHT_MAX}
                        step="any"
                        value={point.weight}
                        aria-label="Landmark weight"
                        onChange={(event) =>
                          update(point.id, { weight: clampWeight(Number(event.target.value)) })
                        }
                      />
                    </td>
                    {point.xy ? (
                      <>
                        <td>
                          <input
                            type="number"
                            step={0.5}
                            value={point.xy[0]}
                            aria-label="x"
                            onChange={(event) =>
                              update(point.id, { xy: [Number(event.target.value), point.xy![1]] })
                            }
                          />
                        </td>
                        <td>
                          <input
                            type="number"
                            step={0.5}
                            value={point.xy[1]}
                            aria-label="y"
                            onChange={(event) =>
                              update(point.id, { xy: [point.xy![0], Number(event.target.value)] })
                            }
                          />
                        </td>
                      </>
                    ) : (
                      <td colSpan={2} className="muted">
                        {point.id === selected ? 'click the canvas' : 'not placed'}
                      </td>
                    )}
                    <td>
                      <button
                        type="button"
                        className="btn btn--small"
                        title="Delete this landmark"
                        aria-label={`Delete ${point.name}`}
                        onClick={(event) => {
                          event.stopPropagation()
                          remove(point.id)
                        }}
                      >
                        ×
                      </button>
                    </td>
                  </tr>
                ))}
                {points.length === 0 && (
                  <tr>
                    <td colSpan={5} className="muted">
                      No landmarks yet. Auto-detect, or double-click the canvas.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
            {lines.length > 0 && (
              <table className="mono lm-editor__lines">
                <thead>
                  <tr>
                    <th title={UI_HELP.lmLineName}>line</th>
                    <th title={UI_HELP.lmLineWeight}>weight</th>
                    <th title={UI_HELP.lmClosed}>closed</th>
                    <th title="Vertices">pts</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {lines.map((line) => (
                    <tr
                      key={line.id}
                      aria-selected={line.id === selectedLine}
                      onClick={() => selectLine(line.id, line.id === selectedLine ? selectedVertex : null)}
                      title={`${lineHint(line.name)}\n\n(${line.source}${line.edited ? ', edited' : ''})`}
                    >
                      <td>
                        <span className={`lm-swatch lm-line-swatch${line.source === 'manual' || line.edited ? '' : ' lm-line-swatch--detected'}`} />
                        <input
                          type="text"
                          value={line.name}
                          aria-label="Line name"
                          onChange={(event) => updateLine(line.id, { name: event.target.value })}
                        />
                      </td>
                      <td>
                        <input
                          type="number"
                          min={0}
                          max={WEIGHT_MAX}
                          step="any"
                          value={line.weight}
                          aria-label="Line weight"
                          onChange={(event) =>
                            updateLine(line.id, { weight: clampWeight(Number(event.target.value)) })
                          }
                        />
                      </td>
                      <td>
                        <input
                          type="checkbox"
                          checked={line.closed}
                          aria-label="Closed"
                          onChange={(event) => updateLine(line.id, { closed: event.target.checked })}
                        />
                      </td>
                      <td className={lineIsValid(line) ? undefined : 'muted'}>{line.xy.length}</td>
                      <td>
                        <button
                          type="button"
                          className="btn btn--small"
                          title="Delete this line"
                          aria-label={`Delete ${line.name}`}
                          onClick={(event) => {
                            event.stopPropagation()
                            removeLine(line.id)
                          }}
                        >
                          ×
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}

      <div className="note">
        {placing
          ? `Click the canvas to place ${placing.name}: ${placementHint(placing.name)}`
          : chosenLine
            ? `${chosenLine.name}: double-click to add a vertex on its nearest edge${chosenLine.xy.length < 2 ? ' (the first ones are appended)' : ''}, drag a square to move it, arrow keys nudge the selected one, Delete removes it (with no square selected, the whole line). Esc deselects.${lineIsValid(chosenLine) ? '' : ` Needs ${chosenLine.closed ? 3 : 2} vertices before the run uses it.`}`
            : mode === 'box'
            ? 'Drag a box around the face; detection runs inside it.'
            : 'Wheel to zoom, drag to pan, double-click to add, drag a dot to move it, arrow keys nudge the selected one (Shift for 5 px), Delete removes it.'}
      </div>
      {chosenLine && <div className="note">{lineHint(chosenLine.name)}</div>}
      {chosen && !placing && (
        <div className="note">
          <strong>{chosen.name}</strong>: {placementHint(chosen.name)}
          {referenceFor(chosen.name) && (
            <>
              {' '}
              <a href={referenceFor(chosen.name)!.url} target="_blank" rel="noopener noreferrer">
                {referenceFor(chosen.name)!.label} ↗
              </a>
            </>
          )}
        </div>
      )}
      {status && <div className="note mono">{status}</div>}
      {problem && <div className="warn">{problem}</div>}
      <div className="note lm-editor__refs">
        Reference:{' '}
        {REFERENCES.map((reference, index) => (
          <span key={reference.url}>
            {index > 0 && ' · '}
            <a href={reference.url} target="_blank" rel="noopener noreferrer">
              {reference.label} ↗
            </a>
          </span>
        ))}
      </div>
      {usable > 0 && termWeight <= 0 && (
        <div className="warn">
          The landmark term is 0, so these {usable} points do nothing.{' '}
          <button type="button" className="btn btn--small" onClick={onEnableTerm}>
            Turn it on
          </button>
        </div>
      )}
    </div>
  )
}

function center(view: ViewBox): [number, number] {
  return [view.x + view.w / 2, view.y + view.h / 2]
}

function describeView(result: LandmarkExtract): string {
  const view = result.view
  if (!view) return 'Detected'
  const yaw = view.yaw === null ? '' : `, turned ${Math.abs(Math.round(view.yaw))}°`
  const how = view.method === 'pose' ? ' (face mesh failed; body pose + outline)' : ''
  return `${view.kind ?? 'face'}${yaw}${view.facing ? ` to the ${view.facing}` : ''}${how}`
}

function droppedNote(result: LandmarkExtract): string {
  const dropped = result.dropped ?? []
  if (!dropped.length) return ''
  // The dense sets drop dozens of mesh points: name the named ones, count the rest.
  const named = dropped.filter((name) => !/^m\d+$/.test(name))
  const mesh = dropped.length - named.length
  const list = [...named, ...(mesh ? [`${mesh} mesh points`] : [])]
  return ` Hidden side left out: ${list.join(', ')}.`
}

function poseNote(result: LandmarkExtract): string {
  const text = describePose(result.pose)
  const model = result.landmarks.filter((point) => point.source === 'model').length
  return `${text ? ` Head: ${text}.` : ''}${model ? ` ${model} points placed by the fitted head.` : ''}`
}

function setNote(result: LandmarkExtract, asked: string): string {
  return asked !== 'sparse' && result.landmark_set === 'sparse'
    ? ` The ${asked} set needs the face mesh, which found nothing here: these are the side-view points.`
    : ''
}

function hairlineNote(result: LandmarkExtract, asked: boolean): string {
  if (!asked) return ''
  return (result.polylines ?? []).some((line) => line.name === 'hairline')
    ? ''
    : ' No hairline found (it needs a frontal or turned face and hair that differs from the skin).'
}
