import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type {
  ImageLossPreview,
  JobInput,
  JobSummary,
  ParamValue,
  Params,
} from '../api/types'
import type { InputRef, OptionalField } from '../lib/formstate'
import { scheduleSeries, sparkline } from '../lib/imageloss'
import { IMAGE_LOSS_DEFAULT_START, SPEC_BY_NAME } from '../lib/params'
import { jobLabel } from '../lib/format'
import { UI_HELP, paramTooltip } from '../lib/help'
import { LandmarkEditor } from './LandmarkEditor'

type EdgeSource = 'derived' | 'prepared' | 'file'

const PREPARED_LABEL = 'prepared edge map'
const EDGE_BUDGET: [number, number] = [2000, 200000]

/**
 * Image fidelity (Spec 6 SS11.2): the gate, the strength and its schedule, the
 * three term weights, and where the edge target comes from.
 *
 * Stateless over the form like `CannyPanel`: everything that must survive a
 * reload is a parameter or an optional input field. What is local is only the
 * preview and the knobs of a *prepared* map, which exist to produce an upload
 * whose sha256 is then the persisted part.
 *
 * In Run again there is no `onOptional`: input files are inherited from the
 * parent and shown read-only, as for every other role.
 */
export function ImageLossPanel({
  params,
  optional,
  targetSha256,
  jobs = [],
  inherited,
  onChange,
  onOptional,
  onBlock,
}: {
  params: Params
  optional?: Record<string, OptionalField>
  targetSha256: string | null
  jobs?: JobSummary[]
  inherited?: JobInput[]
  onChange: (name: string, value: ParamValue) => void
  onOptional?: (name: string, patch: Partial<OptionalField>) => void
  /** A reason submission must wait (a pending or failed prepared map), or null. */
  onBlock?: (reason: string | null) => void
}) {
  const enabled = Boolean(params.image_loss)
  const target = optional?.image_loss_target
  const [source, setSource] = useState<EdgeSource>(() => initialSource(target))
  const [prep, setPrep] = useState({ clahe_clip: 2, clahe_grid: 8, roi: '', preserve_silhouette: true })
  const [preview, setPreview] = useState<ImageLossPreview | null>(null)
  const [problem, setProblem] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const generation = useRef(0)
  const fileInput = useRef<HTMLInputElement>(null)
  const [landmarksSaving, setLandmarksSaving] = useState(false)

  const inheritedTarget = inherited?.find((input) => input.role === 'image_loss_target')
  const inheritedLandmarks = inherited?.find((input) => input.role === 'image_loss_landmarks')
  const landmarkField = optional?.image_loss_landmarks
  const wantsLandmarks = Number(params.image_loss_landmark) > 0
  const [tab, setTab] = useState<'edges' | 'landmarks'>(wantsLandmarks ? 'landmarks' : 'edges')
  const hasLandmarks = Boolean(
    inheritedLandmarks || (landmarkField?.enabled && landmarkField.inputs?.length),
  )
  const editable = Boolean(onOptional)
  const showPreview = enabled && (source === 'derived' || source === 'prepared') && !inheritedTarget

  const roi = parseRoi(prep.roi)
  const requestParams: Record<string, unknown> = {
    low: params.image_loss_canny_low,
    high: params.image_loss_canny_high,
    blur: params.image_loss_canny_blur,
    ...(source === 'prepared'
      ? {
          clahe_clip: prep.clahe_clip,
          clahe_grid: prep.clahe_grid,
          roi,
          preserve_silhouette: prep.preserve_silhouette,
        }
      : {}),
  }
  const requestKey = JSON.stringify(requestParams)

  const refresh = useCallback(async () => {
    if (!showPreview || !targetSha256) return
    const ticket = ++generation.current
    setBusy(true)
    try {
      const result = await api.imageLossPreview({ target_sha256: targetSha256, params: requestParams })
      if (ticket !== generation.current) return
      setPreview(result)
      setProblem(null)
      if (source === 'prepared' && onOptional) {
        onOptional('image_loss_target', {
          enabled: true,
          value: null,
          inputs: [{ source_kind: 'upload', sha256: result.sha256, label: PREPARED_LABEL }],
        })
      }
    } catch (error) {
      if (ticket !== generation.current) return
      setPreview(null)
      setProblem(error instanceof Error ? error.message : 'Could not build the preview')
    } finally {
      if (ticket === generation.current) setBusy(false)
    }
  }, [showPreview, targetSha256, requestKey, source]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!showPreview || !targetSha256) return
    const timer = window.setTimeout(refresh, 350)
    return () => window.clearTimeout(timer)
  }, [showPreview, targetSha256, requestKey, refresh])

  // A prepared map is only usable once its preview exists: until then the
  // attached upload is stale or missing, and submitting would run the wrong map.
  // A landmark weight needs a landmark file, which the server also enforces.
  useEffect(() => {
    if (!onBlock) return
    if (!enabled) onBlock(null)
    else if (source === 'prepared' && busy) onBlock('The prepared edge map is still being built.')
    else if (source === 'prepared' && !preview)
      onBlock(problem ?? 'The prepared edge map needs a preview first (run this image once).')
    else if (landmarksSaving) onBlock('The landmarks are still being saved.')
    else if (wantsLandmarks && !hasLandmarks)
      onBlock('The landmark term needs landmarks: detect or place them in the Landmarks tab.')
    else onBlock(null)
  }, [enabled, source, busy, preview, problem, wantsLandmarks, hasLandmarks, landmarksSaving, onBlock])

  const chooseSource = (next: EdgeSource) => {
    setSource(next)
    setPreview(null)
    if (!onOptional) return
    // Derived: nothing attached. The other two attach their own file.
    if (next === 'derived') onOptional('image_loss_target', { enabled: false })
    if (next === 'file')
      onOptional('image_loss_target', {
        enabled: Boolean(target?.inputs?.length && target.inputs[0].label !== PREPARED_LABEL),
        inputs: target?.inputs?.filter((ref) => ref.label !== PREPARED_LABEL) ?? [],
      })
  }

  const attachFile = (reference: InputRef) =>
    onOptional?.('image_loss_target', { enabled: true, value: null, inputs: [reference] })

  const receivePng = async (file: File) => {
    try {
      const result = await api.upload(file, file.name)
      attachFile({ source_kind: 'upload', sha256: result.sha256, label: file.name })
    } catch (error) {
      setProblem(error instanceof Error ? error.message : 'Upload failed')
    } finally {
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  const schedule = String(params.image_loss_schedule)
  const weight = Number(params.image_loss_weight)
  const start = params.image_loss_schedule_start == null ? null : Number(params.image_loss_schedule_start)
  const numIter = Number(params.num_iter ?? 4000)
  const series = scheduleSeries(numIter, schedule, weight, start)
  const sameTarget = jobs.filter((job) => job.target_sha256 === targetSha256)
  const edgePixels = preview?.edge_pixels ?? null

  const number = (name: string, extra?: { placeholder?: string }) => (
    <div className="field" key={name} title={paramTooltip(SPEC_BY_NAME[name])}>
      <label htmlFor={`il-${name}`}>
        {SPEC_BY_NAME[name].label}
      </label>
      <input
        id={`il-${name}`}
        type="number"
        step={SPEC_BY_NAME[name].step ?? 1}
        min={SPEC_BY_NAME[name].min}
        max={SPEC_BY_NAME[name].max}
        placeholder={extra?.placeholder}
        value={params[name] == null ? '' : String(params[name])}
        onChange={(event) =>
          onChange(
            name,
            event.target.value === '' && SPEC_BY_NAME[name].default === null
              ? null
              : Number(event.target.value),
          )
        }
      />
    </div>
  )

  return (
    <div className={`optional${enabled ? '' : ' optional--off'}`}>
      <input
        type="checkbox"
        checked={enabled}
        aria-label="Use image fidelity"
        title={paramTooltip(SPEC_BY_NAME.image_loss)}
        onChange={(event) => onChange('image_loss', event.target.checked)}
      />
      <div className="optional__body">
        <strong title={paramTooltip(SPEC_BY_NAME.image_loss)}>{SPEC_BY_NAME.image_loss.label}</strong>
        <div className="note">{SPEC_BY_NAME.image_loss.hint}</div>

        {enabled && (
          <>
            <div className="eyebrow" style={{ margin: '8px 0 4px' }} title={UI_HELP.strength}>
              Strength
            </div>
            <div className="field" title={paramTooltip(SPEC_BY_NAME.image_loss_weight)}>
              <label htmlFor="il-weight">{SPEC_BY_NAME.image_loss_weight.label}</label>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <input
                  id="il-weight"
                  type="range"
                  min={0.05}
                  max={1}
                  step={0.05}
                  value={weight}
                  onChange={(event) => onChange('image_loss_weight', Number(event.target.value))}
                  style={{ flex: 1 }}
                />
                <input
                  type="number"
                  aria-label="Fidelity weight value"
                  min={0.05}
                  max={1}
                  step={0.05}
                  value={weight}
                  onChange={(event) => onChange('image_loss_weight', Number(event.target.value))}
                  style={{ width: 70 }}
                />
              </div>
            </div>
            <div className="grid-knobs">
              <div className="field" title={paramTooltip(SPEC_BY_NAME.image_loss_schedule)}>
                <label htmlFor="il-schedule">{SPEC_BY_NAME.image_loss_schedule.label}</label>
                <select
                  id="il-schedule"
                  value={schedule}
                  onChange={(event) => {
                    onChange('image_loss_schedule', event.target.value)
                    // A start only means something for decay/ramp; the server
                    // refuses one with constant.
                    if (event.target.value === 'constant') onChange('image_loss_schedule_start', null)
                  }}
                >
                  {SPEC_BY_NAME.image_loss_schedule.choices?.map((choice) => (
                    <option key={choice} value={choice}>
                      {choice}
                    </option>
                  ))}
                </select>
              </div>
              {schedule !== 'constant' &&
                number('image_loss_schedule_start', {
                  placeholder: String(IMAGE_LOSS_DEFAULT_START[schedule] ?? ''),
                })}
            </div>
            {schedule !== 'constant' && (
              <div className="note mono" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <svg width={160} height={28} role="img" aria-label="Fidelity weight over the run">
                  <polyline
                    points={sparkline(series, 160, 28, [0, 1])}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={1.5}
                  />
                </svg>
                {series[0].toFixed(2)} → {series[series.length - 1].toFixed(2)} over {numIter} iterations
              </div>
            )}

            <div className="eyebrow" style={{ margin: '8px 0 4px' }} title={UI_HELP.terms}>
              Terms
            </div>
            <div className="grid-knobs">
              {number('image_loss_chamfer')}
              {number('image_loss_pyramid')}
              {number('image_loss_landmark')}
              {number('image_loss_curve_samples')}
            </div>
            <div className="note">
              Relative weights; 0 switches a term off. Chamfer pulls the line onto edges, pyramid
              matches where the ink sits at several scales, landmark keeps line near a few anchors.
            </div>

            <div className="tabs" role="tablist" style={{ margin: '10px 0 6px' }}>
              {(
                [
                  ['edges', 'Edge map', UI_HELP.edgeTab],
                  ['landmarks', 'Landmarks', UI_HELP.landmarksTab],
                ] as const
              ).map(([value, label, help]) => (
                <button
                  key={value}
                  type="button"
                  role="tab"
                  className="tab"
                  aria-selected={tab === value}
                  title={help}
                  onClick={() => setTab(value)}
                >
                  {label}
                  {value === 'landmarks' && hasLandmarks ? ' ·' : ''}
                </button>
              ))}
            </div>

            {/* Hidden rather than unmounted on the other tab: a debounced save
                in flight must still land. */}
            <div style={{ display: tab === 'landmarks' ? undefined : 'none' }}>
              {inheritedLandmarks ? (
                <div className="note mono">
                  Inherited from the parent: {inheritedLandmarks.stored_path.split('/').pop()}
                </div>
              ) : editable ? (
                <LandmarkEditor
                  targetSha256={targetSha256}
                  attachedSha256={
                    landmarkField?.enabled && landmarkField.inputs?.[0]?.source_kind === 'upload'
                      ? (landmarkField.inputs[0].sha256 ?? null)
                      : null
                  }
                  edgeUrl={preview ? `${preview.edge_url}?v=${encodeURIComponent(preview.sha256)}` : null}
                  termWeight={Number(params.image_loss_landmark)}
                  renderSize={Number(params.render_size)}
                  fallbackCanvas={
                    preview
                      ? {
                          source_job_id: preview.source_job_id,
                          image_url: preview.image_url,
                          image_size: [Number(params.render_size), Number(params.render_size)],
                        }
                      : null
                  }
                  onAttach={(reference) =>
                    onOptional?.(
                      'image_loss_landmarks',
                      reference
                        ? { enabled: true, value: null, inputs: [reference] }
                        : { enabled: false, inputs: [] },
                    )
                  }
                  onPending={setLandmarksSaving}
                  onEnableTerm={() => onChange('image_loss_landmark', 1)}
                />
              ) : (
                <div className="note">No landmarks.</div>
              )}
            </div>

            {tab === 'edges' && (
              <>
            {inheritedTarget ? (
              <div className="note mono">
                Inherited from the parent: {inheritedTarget.stored_path.split('/').pop()}
              </div>
            ) : (
              <>
                {editable && (
                  <div className="btn-row" role="radiogroup" aria-label="Edge map source">
                    {(
                      [
                        ['derived', 'Derived in the run'],
                        ['prepared', 'Prepared here'],
                        ['file', 'From a file'],
                      ] as const
                    ).map(([value, label]) => (
                      <label
                        key={value}
                        style={{ display: 'flex', gap: 4, alignItems: 'center' }}
                        title={EDGE_SOURCE_HELP[value]}
                      >
                        <input
                          type="radio"
                          name="il-source"
                          checked={source === value}
                          onChange={() => chooseSource(value)}
                        />
                        {label}
                      </label>
                    ))}
                  </div>
                )}

                {(source === 'derived' || source === 'prepared') && (
                  <div className="grid-knobs">
                    {number('image_loss_canny_low')}
                    {number('image_loss_canny_high')}
                    {number('image_loss_canny_blur')}
                  </div>
                )}

                {source === 'prepared' && (
                  <div className="grid-knobs">
                    <div className="field" title={UI_HELP.claheClip}>
                      <label htmlFor="il-clahe">CLAHE clip</label>
                      <input
                        id="il-clahe"
                        type="number"
                        min={0}
                        step={0.5}
                        value={prep.clahe_clip}
                        onChange={(event) => setPrep({ ...prep, clahe_clip: Number(event.target.value) })}
                      />
                    </div>
                    <div className="field" title={UI_HELP.claheGrid}>
                      <label htmlFor="il-grid">CLAHE grid</label>
                      <input
                        id="il-grid"
                        type="number"
                        min={1}
                        value={prep.clahe_grid}
                        onChange={(event) => setPrep({ ...prep, clahe_grid: Number(event.target.value) })}
                      />
                    </div>
                    <div className="field" title={UI_HELP.roi}>
                      <label htmlFor="il-roi">ROI x0 y0 x1 y1</label>
                      <input
                        id="il-roi"
                        type="text"
                        placeholder="whole canvas"
                        value={prep.roi}
                        onChange={(event) => setPrep({ ...prep, roi: event.target.value })}
                      />
                    </div>
                    <label
                      style={{ display: 'flex', gap: 4, alignItems: 'center' }}
                      title={UI_HELP.keepSilhouette}
                    >
                      <input
                        type="checkbox"
                        checked={prep.preserve_silhouette}
                        onChange={(event) =>
                          setPrep({ ...prep, preserve_silhouette: event.target.checked })
                        }
                      />
                      Keep silhouette
                    </label>
                  </div>
                )}

                {source === 'file' && editable && (
                  <div>
                    {target?.inputs?.[0] && target.enabled && (
                      <div className="note mono">
                        Attached: {target.inputs[0].label ?? target.inputs[0].path ?? 'file'}
                      </div>
                    )}
                    <div className="btn-row">
                      <button type="button" className="btn btn--small" onClick={() => fileInput.current?.click()}>
                        Upload PNG…
                      </button>
                      <input
                        ref={fileInput}
                        type="file"
                        accept="image/png"
                        hidden
                        onChange={(event) => {
                          const file = event.target.files?.[0]
                          if (file) receivePng(file)
                        }}
                      />
                      {sameTarget.length > 0 && (
                        <select
                          className="input"
                          value=""
                          aria-label="Edge map from a job on this image"
                          onChange={(event) => {
                            const [jobId, path] = event.target.value.split('|')
                            const job = sameTarget.find((entry) => entry.id === jobId)
                            if (job)
                              attachFile({
                                source_kind: 'job',
                                source_job_id: jobId,
                                path,
                                label: `${jobLabel(job)} · ${path}`,
                              })
                          }}
                        >
                          <option value="">From a run of this image…</option>
                          {sameTarget.flatMap((job) =>
                            ['image_loss_target.png', 'condition_canny.png'].map((path) => (
                              <option key={`${job.id}|${path}`} value={`${job.id}|${path}`}>
                                {jobLabel(job)} · {path}
                              </option>
                            )),
                          )}
                        </select>
                      )}
                    </div>
                    <div className="note">
                      A canvas-space PNG at the render size — a previous run's image_loss_target.png
                      or condition_canny.png, or a map made with sld_edge_target.py from its
                      input.png. Never the original photograph.
                    </div>
                  </div>
                )}

                {showPreview && (
                  <div
                    className="canvas-wrap"
                    style={{ position: 'relative', minHeight: 160, opacity: busy ? 0.55 : 1 }}
                    aria-busy={busy}
                  >
                    {preview ? (
                      <div style={{ position: 'relative', display: 'inline-block', lineHeight: 0 }}>
                        <img
                          src={preview.image_url}
                          alt="Canvas-space target"
                          style={{ maxWidth: '100%', maxHeight: 320, display: 'block', opacity: 0.45 }}
                        />
                        {/* White edges on black, screened over the canvas image:
                            the black vanishes and the edges light up. */}
                        <img
                          src={`${preview.edge_url}?v=${encodeURIComponent(preview.sha256)}`}
                          alt="Edge target"
                          style={{
                            position: 'absolute',
                            inset: 0,
                            width: '100%',
                            height: '100%',
                            mixBlendMode: 'screen',
                            pointerEvents: 'none',
                          }}
                        />
                      </div>
                    ) : (
                      <span className="muted note">
                        {busy
                          ? 'Finding edges…'
                          : (problem ??
                            'No preview yet — this image has not been run before. The run derives its own edges.')}
                      </span>
                    )}
                  </div>
                )}

                {preview && showPreview && (
                  <div className="note mono">
                    {edgePixels ?? '?'} edge px
                    {preview.derived_equivalent ? ' · the run derives this map itself' : ' · attached as a file'}
                    {` · from job ${preview.source_job_id.slice(-6)}`}
                  </div>
                )}
                {edgePixels !== null && showPreview && (edgePixels < EDGE_BUDGET[0] || edgePixels > EDGE_BUDGET[1]) && (
                  <div className="warn">
                    {edgePixels} edge pixels is outside {EDGE_BUDGET[0]}–{EDGE_BUDGET[1]}. The
                    thresholds are probably wrong: too few and the line has nothing to sit on,
                    too many and every stroke is near an edge.
                  </div>
                )}
                {problem && preview && <div className="warn">{problem}</div>}
              </>
            )}
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}

const EDGE_SOURCE_HELP: Record<EdgeSource, string> = {
  derived: UI_HELP.edgeDerived,
  prepared: UI_HELP.edgePrepared,
  file: UI_HELP.edgeFile,
}

function initialSource(target: OptionalField | undefined): EdgeSource {
  if (!target?.enabled || !target.inputs?.length) return 'derived'
  return target.inputs[0].label === PREPARED_LABEL ? 'prepared' : 'file'
}

function parseRoi(text: string): number[] | null {
  const values = text
    .split(/[\s,]+/)
    .filter(Boolean)
    .map(Number)
  return values.length === 4 && values.every(Number.isFinite) ? values : null
}
