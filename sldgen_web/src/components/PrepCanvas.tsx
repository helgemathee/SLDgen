import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  brushStamp,
  buildLabCache,
  combineMasks,
  countMask,
  featherMask,
  invertMask,
  magicWand,
  maskOutline,
} from '../lib/lab'
import type { MaskMode } from '../lib/formstate'
import { MASK_MODES } from '../lib/formstate'

/** Interaction speed matters more than fidelity while selecting (Spec 3 SS8.2). */
const WORKING_MAX = 2048

export interface PrepResult {
  /** target.png, at the source's full resolution. */
  target: Blob
  /** weight.png, only for the guide and control mask modes. */
  weight: Blob | null
}

type Tool = 'wand' | 'brush-add' | 'brush-remove' | 'density'

interface Snapshot {
  mask: Uint8Array
  density: Float32Array
}

export interface PrepCanvasHandle {
  /** Produce the export. Async because it upsamples through a canvas. */
  export: () => Promise<PrepResult>
  hasSelection: boolean
}

export function PrepCanvas({
  imageUrl,
  maskMode,
  origin,
  onOrigin,
  overlays,
  onReady,
}: {
  imageUrl: string
  maskMode: MaskMode
  /** Normalised origin, or null when the pin is off. */
  origin: [number, number] | null
  onOrigin: (origin: [number, number]) => void
  /** Hairline SVG overlays for avoid / attract / init_points sources. */
  overlays: string[]
  onReady: (handle: PrepCanvasHandle) => void
}) {
  const display = useRef<HTMLCanvasElement>(null)
  const source = useRef<{ full: HTMLCanvasElement; work: HTMLCanvasElement } | null>(null)
  const lab = useRef<Float32Array | null>(null)
  const mask = useRef<Uint8Array>(new Uint8Array(0))
  const density = useRef<Float32Array>(new Float32Array(0))
  const history = useRef<{ past: Snapshot[]; future: Snapshot[] }>({ past: [], future: [] })
  const painting = useRef(false)

  const [size, setSize] = useState({ width: 0, height: 0 })
  const [tool, setTool] = useState<Tool>('wand')
  const [tolerance, setTolerance] = useState(18)
  const [feather, setFeather] = useState(0)
  const [contiguous, setContiguous] = useState(true)
  const [brushRadius, setBrushRadius] = useState(28)
  const [hardness, setHardness] = useState(0.6)
  const [densityValue, setDensityValue] = useState(0.25)
  const [showWash, setShowWash] = useState(true)
  const [showKnockout, setShowKnockout] = useState(false)
  const [selected, setSelected] = useState(0)
  const [placingOrigin, setPlacingOrigin] = useState(false)
  const [version, setVersion] = useState(0)
  const [overlayImages, setOverlayImages] = useState<HTMLImageElement[]>([])

  const densityAvailable = maskMode !== 'clean'

  // -- load ----------------------------------------------------------------

  useEffect(() => {
    let cancelled = false
    const image = new Image()
    image.crossOrigin = 'anonymous'
    image.onload = () => {
      if (cancelled) return
      const full = document.createElement('canvas')
      full.width = image.naturalWidth
      full.height = image.naturalHeight
      full.getContext('2d')!.drawImage(image, 0, 0)

      const scale = Math.min(1, WORKING_MAX / Math.max(image.naturalWidth, image.naturalHeight))
      const work = document.createElement('canvas')
      work.width = Math.max(1, Math.round(image.naturalWidth * scale))
      work.height = Math.max(1, Math.round(image.naturalHeight * scale))
      work.getContext('2d')!.drawImage(image, 0, 0, work.width, work.height)

      source.current = { full, work }
      const pixels = work.getContext('2d')!.getImageData(0, 0, work.width, work.height)
      lab.current = buildLabCache(pixels.data)
      mask.current = new Uint8Array(work.width * work.height)
      density.current = new Float32Array(work.width * work.height).fill(1)
      history.current = { past: [], future: [] }
      setSize({ width: work.width, height: work.height })
      setSelected(0)
      setVersion((value) => value + 1)
    }
    image.src = imageUrl
    return () => {
      cancelled = true
    }
  }, [imageUrl])

  useEffect(() => {
    let cancelled = false
    Promise.all(
      overlays.map(
        (url) =>
          new Promise<HTMLImageElement | null>((resolve) => {
            const image = new Image()
            image.onload = () => resolve(image)
            image.onerror = () => resolve(null)
            image.src = url
          }),
      ),
    ).then((images) => {
      if (!cancelled) setOverlayImages(images.filter((image): image is HTMLImageElement => !!image))
    })
    return () => {
      cancelled = true
    }
  }, [overlays.join(',')])

  // -- history -------------------------------------------------------------

  const snapshot = useCallback(() => {
    history.current.past.push({
      mask: mask.current.slice(),
      density: density.current.slice(),
    })
    // Bounded: a full-resolution selection is a megabyte or two, and an
    // unbounded stack of them is how a long editing session runs out of memory.
    if (history.current.past.length > 30) history.current.past.shift()
    history.current.future = []
  }, [])

  const restore = useCallback((snap: Snapshot) => {
    mask.current = snap.mask
    density.current = snap.density
    setSelected(countMask(mask.current))
    setVersion((value) => value + 1)
  }, [])

  const undo = useCallback(() => {
    const previous = history.current.past.pop()
    if (!previous) return
    history.current.future.push({ mask: mask.current.slice(), density: density.current.slice() })
    restore(previous)
  }, [restore])

  const redo = useCallback(() => {
    const next = history.current.future.pop()
    if (!next) return
    history.current.past.push({ mask: mask.current.slice(), density: density.current.slice() })
    restore(next)
  }, [restore])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey)) return
      if (event.key.toLowerCase() !== 'z') return
      event.preventDefault()
      if (event.shiftKey) redo()
      else undo()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [undo, redo])

  // -- painting ------------------------------------------------------------

  const toWorking = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = display.current!
    const rect = canvas.getBoundingClientRect()
    return {
      x: Math.floor(((event.clientX - rect.left) / rect.width) * size.width),
      y: Math.floor(((event.clientY - rect.top) / rect.height) * size.height),
    }
  }

  const applyBrush = (x: number, y: number) => {
    const stamp = brushStamp({
      width: size.width,
      height: size.height,
      x,
      y,
      radius: brushRadius,
      hardness,
    })
    if (tool === 'density') {
      for (const { index, coverage } of stamp) {
        // Paint towards the chosen density rather than setting it, so repeated
        // strokes build up the way a real brush does.
        density.current[index] += (densityValue - density.current[index]) * coverage * 0.5
      }
    } else {
      const value = tool === 'brush-add' ? 1 : 0
      for (const { index, coverage } of stamp) {
        if (coverage > 0.5) mask.current[index] = value
      }
      setSelected(countMask(mask.current))
    }
    setVersion((value) => value + 1)
  }

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!source.current || !lab.current) return
    const { x, y } = toWorking(event)

    if (placingOrigin) {
      onOrigin([
        Math.min(1, Math.max(0, x / size.width)),
        Math.min(1, Math.max(0, y / size.height)),
      ])
      setPlacingOrigin(false)
      return
    }

    event.currentTarget.setPointerCapture(event.pointerId)
    snapshot()

    if (tool === 'wand') {
      const addition = magicWand({
        lab: lab.current,
        width: size.width,
        height: size.height,
        x,
        y,
        tolerance,
        contiguous,
      })
      mask.current = combineMasks(
        mask.current,
        addition,
        event.shiftKey ? 'add' : event.altKey ? 'subtract' : 'replace',
      )
      setSelected(countMask(mask.current))
      setVersion((value) => value + 1)
      return
    }

    painting.current = true
    applyBrush(x, y)
  }

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!painting.current) return
    const { x, y } = toWorking(event)
    applyBrush(x, y)
  }

  const stopPainting = () => {
    painting.current = false
  }

  // -- render --------------------------------------------------------------

  const outline = useMemo(
    () => (size.width ? maskOutline(mask.current, size.width, size.height) : new Uint8Array(0)),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `version` is the signal
    [version, size.width, size.height],
  )

  useEffect(() => {
    const canvas = display.current
    const current = source.current
    if (!canvas || !current || size.width === 0) return
    canvas.width = size.width
    canvas.height = size.height
    const context = canvas.getContext('2d')!
    context.clearRect(0, 0, size.width, size.height)
    context.drawImage(current.work, 0, 0)

    const frame = context.getImageData(0, 0, size.width, size.height)
    const pixels = frame.data
    const alpha = feather > 0 ? featherMask(mask.current, size.width, size.height, feather) : null

    for (let index = 0; index < mask.current.length; index += 1) {
      const coverage = alpha ? alpha[index] / 255 : mask.current[index]
      if (showKnockout && coverage > 0) {
        // The checkerboard preview: what will actually be removed.
        pixels[index * 4 + 3] = Math.round(255 * (1 - coverage))
        continue
      }
      if (showWash && coverage > 0) {
        pixels[index * 4] = pixels[index * 4] * (1 - 0.5 * coverage) + 255 * 0.5 * coverage
        pixels[index * 4 + 1] = pixels[index * 4 + 1] * (1 - 0.5 * coverage) + 255 * 0.5 * coverage
        pixels[index * 4 + 2] = pixels[index * 4 + 2] * (1 - 0.5 * coverage) + 255 * 0.5 * coverage
      }
      if (densityAvailable) {
        // Show painted density as a red-free darkening, so it reads as "more
        // ink here" without introducing decorative colour.
        const value = density.current[index]
        if (value < 0.999) {
          const tint = 1 - (1 - value) * 0.45
          pixels[index * 4] *= tint
          pixels[index * 4 + 1] *= tint
          pixels[index * 4 + 2] *= tint
        }
      }
    }

    // Marching ants, drawn last so nothing above covers them.
    for (let index = 0; index < outline.length; index += 1) {
      if (!outline[index]) continue
      const dark = ((index % size.width) + Math.floor(index / size.width)) % 8 < 4
      pixels[index * 4] = dark ? 20 : 255
      pixels[index * 4 + 1] = dark ? 20 : 255
      pixels[index * 4 + 2] = dark ? 20 : 255
      pixels[index * 4 + 3] = 255
    }
    context.putImageData(frame, 0, 0)

    // Reference overlays: the spatial relationship between the new curve's
    // constraints and the image, which is what makes those flags usable rather
    // than guesswork (SS8.2).
    context.save()
    context.globalAlpha = 0.5
    for (const image of overlayImages) {
      context.drawImage(image, 0, 0, size.width, size.height)
    }
    context.restore()

    if (origin) {
      const x = origin[0] * size.width
      const y = origin[1] * size.height
      const arm = Math.max(10, size.width * 0.02)
      context.save()
      context.strokeStyle = '#14140f'
      context.lineWidth = Math.max(1, size.width / 700)
      context.beginPath()
      context.moveTo(x - arm, y)
      context.lineTo(x + arm, y)
      context.moveTo(x, y - arm)
      context.lineTo(x, y + arm)
      context.stroke()
      context.beginPath()
      context.arc(x, y, arm * 0.45, 0, Math.PI * 2)
      context.stroke()
      context.restore()
    }
  }, [version, size, showWash, showKnockout, feather, outline, origin, overlayImages, densityAvailable])

  // -- export --------------------------------------------------------------

  const exportResult = useCallback(async (): Promise<PrepResult> => {
    const current = source.current
    if (!current) throw new Error('No image loaded')
    const { full } = current

    // The selection was made at working resolution; the export is produced from
    // the full-resolution original using that same selection, upsampled (SS8.2).
    const coverage = upscale(
      feather > 0
        ? featherMask(mask.current, size.width, size.height, feather)
        : Uint8ClampedArray.from(mask.current, (value) => (value ? 255 : 0)),
      size.width,
      size.height,
      full.width,
      full.height,
    )

    const target = document.createElement('canvas')
    target.width = full.width
    target.height = full.height
    const context = target.getContext('2d')!
    // White, not transparent: SLDgen's target is an opaque image, and a
    // transparent knockout would composite as black in most readers.
    context.fillStyle = '#ffffff'
    context.fillRect(0, 0, full.width, full.height)
    context.drawImage(full, 0, 0)
    const frame = context.getImageData(0, 0, full.width, full.height)
    for (let index = 0; index < coverage.length; index += 1) {
      const value = coverage[index] / 255
      if (value <= 0) continue
      frame.data[index * 4] = frame.data[index * 4] * (1 - value) + 255 * value
      frame.data[index * 4 + 1] = frame.data[index * 4 + 1] * (1 - value) + 255 * value
      frame.data[index * 4 + 2] = frame.data[index * 4 + 2] * (1 - value) + 255 * value
      frame.data[index * 4 + 3] = 255
    }
    context.putImageData(frame, 0, 0)

    let weight: Blob | null = null
    if (densityAvailable) {
      const densityBytes = Uint8ClampedArray.from(density.current, (value) =>
        Math.round(Math.min(1, Math.max(0, value)) * 255),
      )
      const upsampled = upscale(densityBytes, size.width, size.height, full.width, full.height)
      const weightCanvas = document.createElement('canvas')
      weightCanvas.width = full.width
      weightCanvas.height = full.height
      const weightContext = weightCanvas.getContext('2d')!
      const weightFrame = weightContext.createImageData(full.width, full.height)
      for (let index = 0; index < upsampled.length; index += 1) {
        // Grayscale: darker means more ink, which is what --stipple-weight reads.
        const value = upsampled[index]
        weightFrame.data[index * 4] = value
        weightFrame.data[index * 4 + 1] = value
        weightFrame.data[index * 4 + 2] = value
        weightFrame.data[index * 4 + 3] = 255
      }
      weightContext.putImageData(weightFrame, 0, 0)
      weight = await toBlob(weightCanvas)
    }

    return { target: await toBlob(target), weight }
  }, [size, feather, densityAvailable])

  useEffect(() => {
    onReady({ export: exportResult, hasSelection: selected > 0 })
  }, [exportResult, selected, onReady])

  const modeCopy = MASK_MODES.find((entry) => entry.mode === maskMode)!

  return (
    <div className="prep">
      <div className="prep__tools">
        <div className="toolgrid">
          <button
            type="button"
            className="btn"
            aria-pressed={tool === 'wand'}
            onClick={() => setTool('wand')}
          >
            Select similar
          </button>
          <button
            type="button"
            className="btn"
            aria-pressed={tool === 'brush-add'}
            onClick={() => setTool('brush-add')}
          >
            Brush add
          </button>
          <button
            type="button"
            className="btn"
            aria-pressed={tool === 'brush-remove'}
            onClick={() => setTool('brush-remove')}
          >
            Brush remove
          </button>
          <button
            type="button"
            className="btn"
            aria-pressed={tool === 'density'}
            disabled={!densityAvailable}
            title={
              densityAvailable
                ? 'Paint darker where you want more ink'
                : 'Choose Guide or Control the ink to paint density'
            }
            onClick={() => setTool('density')}
          >
            Density brush
          </button>
        </div>

        {tool === 'wand' && (
          <>
            <div className="slider-field">
              <label htmlFor="tolerance">Tolerance — what gets selected</label>
              <input
                id="tolerance"
                type="range"
                min={1}
                max={60}
                value={tolerance}
                onChange={(event) => setTolerance(Number(event.target.value))}
              />
              <span className="mono">{tolerance}</span>
            </div>
            <label className="note" style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <input
                type="checkbox"
                checked={contiguous}
                onChange={(event) => setContiguous(event.target.checked)}
              />
              contiguous {contiguous ? '' : '(global chroma key)'}
            </label>
            <p className="note">Shift-click adds a cluster. Alt-click subtracts one.</p>
          </>
        )}

        {(tool === 'brush-add' || tool === 'brush-remove' || tool === 'density') && (
          <>
            <div className="slider-field">
              <label htmlFor="radius">Radius</label>
              <input
                id="radius"
                type="range"
                min={2}
                max={200}
                value={brushRadius}
                onChange={(event) => setBrushRadius(Number(event.target.value))}
              />
              <span className="mono">{brushRadius}</span>
            </div>
            <div className="slider-field">
              <label htmlFor="hardness">Hardness</label>
              <input
                id="hardness"
                type="range"
                min={0}
                max={100}
                value={Math.round(hardness * 100)}
                onChange={(event) => setHardness(Number(event.target.value) / 100)}
              />
              <span className="mono">{hardness.toFixed(2)}</span>
            </div>
          </>
        )}

        {tool === 'density' && (
          <div className="slider-field">
            <label htmlFor="density-value">Density — 0 is no ink, 1 is full</label>
            <input
              id="density-value"
              type="range"
              min={0}
              max={100}
              value={Math.round(densityValue * 100)}
              onChange={(event) => setDensityValue(Number(event.target.value) / 100)}
            />
            <span className="mono">{densityValue.toFixed(2)}</span>
          </div>
        )}

        <div className="slider-field">
          <label htmlFor="feather">Feather — how the edge falls off</label>
          <input
            id="feather"
            type="range"
            min={0}
            max={40}
            value={feather}
            onChange={(event) => setFeather(Number(event.target.value))}
          />
          <span className="mono">{feather}</span>
        </div>

        <div className="btn-row">
          <button type="button" className="btn btn--small" onClick={undo}>
            Undo
          </button>
          <button type="button" className="btn btn--small" onClick={redo}>
            Redo
          </button>
          <button
            type="button"
            className="btn btn--small"
            onClick={() => {
              snapshot()
              mask.current = invertMask(mask.current)
              setSelected(countMask(mask.current))
              setVersion((value) => value + 1)
            }}
          >
            Invert
          </button>
          <button
            type="button"
            className="btn btn--small"
            onClick={() => {
              snapshot()
              mask.current = new Uint8Array(mask.current.length)
              setSelected(0)
              setVersion((value) => value + 1)
            }}
          >
            Clear
          </button>
        </div>

        <div className="btn-row">
          <button
            type="button"
            className="btn btn--small"
            aria-pressed={showWash}
            onClick={() => setShowWash((value) => !value)}
          >
            wash
          </button>
          <button
            type="button"
            className="btn btn--small"
            aria-pressed={showKnockout}
            onClick={() => setShowKnockout((value) => !value)}
          >
            preview removal
          </button>
          <button
            type="button"
            className="btn btn--small"
            aria-pressed={placingOrigin}
            onClick={() => setPlacingOrigin((value) => !value)}
          >
            {origin ? 'move origin' : 'place origin'}
          </button>
        </div>

        <div className="mono note">
          {size.width}×{size.height} working · {selected.toLocaleString()} px selected
          {origin && ` · origin ${origin[0].toFixed(3)}, ${origin[1].toFixed(3)}`}
        </div>

        <p className="note">
          <strong>{modeCopy.name}.</strong> {modeCopy.effect}
        </p>
      </div>

      <div className={`canvas-wrap${showKnockout ? ' canvas-wrap--checker' : ''}`}>
        {size.width === 0 ? (
          <span className="muted note">Loading image…</span>
        ) : (
          <canvas
            ref={display}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={stopPainting}
            onPointerLeave={stopPainting}
            style={{ cursor: placingOrigin ? 'crosshair' : undefined }}
          />
        )}
      </div>
    </div>
  )
}

/** Nearest-free upsample of an 8-bit plane, through the browser's own scaler. */
function upscale(
  plane: Uint8ClampedArray,
  width: number,
  height: number,
  targetWidth: number,
  targetHeight: number,
): Uint8ClampedArray {
  if (width === targetWidth && height === targetHeight) return plane
  const small = document.createElement('canvas')
  small.width = width
  small.height = height
  const smallContext = small.getContext('2d')!
  const frame = smallContext.createImageData(width, height)
  for (let index = 0; index < plane.length; index += 1) {
    frame.data[index * 4] = plane[index]
    frame.data[index * 4 + 1] = plane[index]
    frame.data[index * 4 + 2] = plane[index]
    frame.data[index * 4 + 3] = 255
  }
  smallContext.putImageData(frame, 0, 0)

  const large = document.createElement('canvas')
  large.width = targetWidth
  large.height = targetHeight
  const largeContext = large.getContext('2d')!
  // Smoothing on: a hard mask upsamples to a hard mask anyway, and a feathered
  // one should interpolate rather than gain stair-steps.
  largeContext.imageSmoothingEnabled = true
  largeContext.imageSmoothingQuality = 'high'
  largeContext.drawImage(small, 0, 0, targetWidth, targetHeight)

  const scaled = largeContext.getImageData(0, 0, targetWidth, targetHeight)
  const out = new Uint8ClampedArray(targetWidth * targetHeight)
  for (let index = 0; index < out.length; index += 1) out[index] = scaled.data[index * 4]
  return out
}

function toBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob)
      else reject(new Error('Could not encode the canvas as PNG'))
    }, 'image/png')
  })
}
