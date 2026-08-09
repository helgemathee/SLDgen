import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { JobSummary, ParamValue, Partition, UploadResult } from '../api/types'
import { ConstraintPicker } from '../components/ConstraintPicker'
import { ParamFields } from '../components/ParamFields'
import { PrepCanvas, type PrepCanvasHandle } from '../components/PrepCanvas'
import {
  MASK_MODES,
  emptyFormState,
  hydrateFormState,
  toInputs,
  toParams,
  type FormState,
  type InputRef,
  type MaskMode,
} from '../lib/formstate'
import { formatDuration, meanItersPerSec } from '../lib/format'
import { SPEC_BY_NAME, validateParams, type ParamSection } from '../lib/params'
import { navigate } from '../router'
import { useApp } from '../state/store'

const SECTIONS: ParamSection[] = ['prompt', 'curve', 'guidance', 'losses']
const INPUT_ROLES = ['avoid', 'attract', 'init_points', 'stipple_weight'] as const

/**
 * The new-job flow (Spec 3 SS8).
 *
 * One scrolling page in four sections rather than a wizard: every part stays
 * visible and adjustable until you submit, because the parameters and the
 * selection inform each other and a wizard would hide one while you set the
 * other.
 */
export function NewJobPage() {
  const { toast, jobs } = useApp()
  const [form, setForm] = useState<FormState>(emptyFormState)
  const [loaded, setLoaded] = useState(false)
  const [upload, setUpload] = useState<UploadResult | null>(null)
  const [recent, setRecent] = useState<{ sha256: string; label: string }[]>([])
  const [partitions, setPartitions] = useState<Partition[]>([])
  const [title, setTitle] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const [busy, setBusy] = useState(false)
  const [presetName, setPresetName] = useState('')
  const [presets, setPresets] = useState<{ id: string; name: string; params: unknown }[]>([])
  const prep = useRef<PrepCanvasHandle | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  // The next job starts from the last job's settings (SS9).
  useEffect(() => {
    api
      .lastParams()
      .then((result) => setForm(hydrateFormState(result.params)))
      .catch(() => undefined)
      .finally(() => setLoaded(true))
    api.presets().then((result) => setPresets(result.presets)).catch(() => undefined)
    api.listPartitions().then((result) => setPartitions(result.partitions)).catch(() => undefined)
  }, [])

  // The same photo is often run many times with different parameters (SS8.1).
  useEffect(() => {
    const seen = new Map<string, string>()
    for (const job of jobs) {
      if (!seen.has(job.target_sha256)) seen.set(job.target_sha256, job.title ?? job.id.slice(-6))
    }
    setRecent(
      Array.from(seen.entries())
        .slice(0, 12)
        .map(([sha256, label]) => ({ sha256, label })),
    )
  }, [jobs])

  const setParam = (name: string, value: ParamValue) =>
    setForm((current) => ({ ...current, params: { ...current.params, [name]: value } }))

  const setOptional = (name: string, patch: Partial<FormState['optional'][string]>) =>
    setForm((current) => ({
      ...current,
      optional: { ...current.optional, [name]: { ...current.optional[name], ...patch } },
    }))

  const receiveFile = async (file: File) => {
    setBusy(true)
    try {
      const result = await api.upload(file, file.name)
      setUpload(result)
      if (!title) setTitle(file.name.replace(/\.[^.]+$/, ''))
    } catch (error) {
      toast(error instanceof Error ? error.message : 'Upload failed')
    } finally {
      setBusy(false)
    }
  }

  const chooseRecent = (sha256: string, label: string) => {
    setUpload({ sha256, width: null, height: null, bytes: 0, filename: label, url: `/api/uploads/${sha256}` })
    if (!title) setTitle(label)
  }

  const params = toParams(form)
  const problems = validateParams(params)
  const rate = meanItersPerSec(
    jobs.flatMap(() => []) as { start_epoch: number; end_epoch: number | null; started_at: string; finished_at: string | null }[],
  )
  const estimate = rate ? formatDuration(form.targetEpoch / rate) : null

  const overlays = INPUT_ROLES.flatMap((role) => {
    const field = form.optional[role]
    if (!field?.enabled) return []
    return (field.inputs ?? [])
      .filter((reference) => (reference.path ?? '').endsWith('.svg'))
      .map((reference) =>
        reference.source_kind === 'job'
          ? `/api/jobs/${reference.source_job_id}/files/target/run/${reference.path}`
          : `/api/partitions/${reference.source_partition_id}/files/${reference.path}`,
      )
  })

  const submit = async () => {
    if (!upload) return
    setBusy(true)
    try {
      let targetSha = upload.sha256
      const inputs = toInputs(form)

      // The prepared image is a new upload, not the original: SLDgen receives
      // exactly what the canvas produced.
      if (prep.current) {
        const result = await prep.current.export()
        const prepared = await api.upload(result.target, 'target.png')
        targetSha = prepared.sha256
        if (result.weight) {
          const weight = await api.upload(result.weight, 'weight.png')
          inputs.push({ role: 'stipple_weight', source_kind: 'upload', sha256: weight.sha256 })
        }
      }

      const submitted = { ...params }
      if (form.maskMode !== 'clean') {
        submitted.stipple_weight_mode = form.maskMode === 'control' ? 'replace' : 'multiply'
      }

      const job = await api.createJob({
        title: title || undefined,
        target_sha256: targetSha,
        params: submitted,
        target_epoch: Math.min(form.targetEpoch, form.numIter),
        inputs,
      })
      // Persist the form, not the image or the selection: those are per-job by
      // nature. The mask *mode* does persist (SS9).
      await api.saveLastParams(form).catch(() => undefined)
      toast('Queued.')
      navigate({ name: 'job', id: job.id })
    } catch (error) {
      toast(error instanceof Error ? error.message : 'Could not queue the job')
    } finally {
      setBusy(false)
    }
  }

  const onPrepReady = useCallback((handle: PrepCanvasHandle) => {
    prep.current = handle
  }, [])

  return (
    <div className="page">
      <section>
        <div className="section-head">
          <span className="eyebrow">1 · Source</span>
          {upload && <span className="mono muted">{upload.sha256.slice(0, 12)}…</span>}
        </div>
        <div
          className={`dropzone${dragOver ? ' dropzone--over' : ''}`}
          onClick={() => fileInput.current?.click()}
          onDragOver={(event) => {
            event.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(event) => {
            event.preventDefault()
            setDragOver(false)
            const file = event.dataTransfer.files[0]
            if (file) receiveFile(file)
          }}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === 'Enter') fileInput.current?.click()
          }}
        >
          {busy && !upload ? 'Uploading…' : 'Drop an image here, or click to pick one'}
          <input
            ref={fileInput}
            type="file"
            accept="image/*"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) receiveFile(file)
            }}
          />
        </div>
        {recent.length > 0 && (
          <>
            <div className="note" style={{ marginTop: 8 }}>
              Recently used
            </div>
            <div className="recent">
              {recent.map((entry) => (
                <button
                  key={entry.sha256}
                  type="button"
                  aria-pressed={upload?.sha256 === entry.sha256}
                  title={entry.label}
                  onClick={() => chooseRecent(entry.sha256, entry.label)}
                >
                  <img src={`/api/uploads/${entry.sha256}`} alt={entry.label} />
                </button>
              ))}
            </div>
          </>
        )}
      </section>

      {upload && (
        <section>
          <div className="section-head">
            <span className="eyebrow">2 · Prepare</span>
          </div>

          <table className="mode-table" style={{ marginBottom: 10 }}>
            <tbody>
              {MASK_MODES.map((entry) => (
                <tr key={entry.mode} aria-selected={form.maskMode === entry.mode}>
                  <td>
                    <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                      <input
                        type="radio"
                        name="mask-mode"
                        checked={form.maskMode === entry.mode}
                        onChange={() =>
                          setForm((current) => ({ ...current, maskMode: entry.mode as MaskMode }))
                        }
                      />
                      <strong>{entry.name}</strong>
                    </label>
                  </td>
                  <td>
                    <div>{entry.effect}</div>
                    <div className="note">Exports {entry.exports}.</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <p className="warn" style={{ marginBottom: 10 }}>
            SLDgen always runs RMBG-1.4 on the target, and that mask also drives the object
            bounding-box rescale — so a selection made here cannot simply be the mask. What it can
            do is remove distractions, and modulate where the ink goes.
          </p>

          <PrepCanvas
            imageUrl={`/api/uploads/${upload.sha256}`}
            maskMode={form.maskMode}
            origin={
              form.optional.origin?.enabled
                ? ((form.optional.origin.value as number[]) as [number, number])
                : null
            }
            onOrigin={(origin) => setOptional('origin', { enabled: true, value: origin })}
            overlays={overlays}
            onReady={onPrepReady}
          />
        </section>
      )}

      <section>
        <div className="section-head">
          <span className="eyebrow">3 · Parameters</span>
          {loaded && <span className="note">Carried over from your last submission.</span>}
          <span style={{ flex: 1 }} />
          <select
            className="input"
            style={{ width: 150 }}
            value=""
            onChange={(event) => {
              const preset = presets.find((entry) => entry.id === event.target.value)
              if (preset) setForm(hydrateFormState(preset.params))
            }}
          >
            <option value="">Load preset…</option>
            {presets.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.name}
              </option>
            ))}
          </select>
          <input
            className="input"
            style={{ width: 130 }}
            placeholder="Preset name"
            value={presetName}
            onChange={(event) => setPresetName(event.target.value)}
          />
          <button
            type="button"
            className="btn btn--small"
            disabled={!presetName.trim()}
            onClick={async () => {
              try {
                const saved = await api.savePreset(presetName.trim(), form)
                setPresets((current) => [
                  ...current.filter((entry) => entry.name !== saved.name),
                  saved,
                ])
                setPresetName('')
                toast(`Saved preset “${saved.name}”.`)
              } catch (error) {
                toast(error instanceof Error ? error.message : 'Could not save the preset')
              }
            }}
          >
            Save preset
          </button>
        </div>

        <div className="field">
          <label htmlFor="job-title">Title</label>
          <input
            id="job-title"
            type="text"
            value={title}
            placeholder="Defaults to the filename"
            onChange={(event) => setTitle(event.target.value)}
          />
        </div>

        <ParamFields params={form.params} sections={SECTIONS} onChange={setParam} />

        <details className="group" open>
          <summary>
            <span className="eyebrow">Constraints</span>
          </summary>
          <div className="group__body">
            <div className={`optional${form.optional.origin?.enabled ? '' : ' optional--off'}`}>
              <input
                type="checkbox"
                checked={form.optional.origin?.enabled ?? false}
                aria-label="Use origin"
                onChange={(event) => setOptional('origin', { enabled: event.target.checked })}
              />
              <div className="optional__body">
                <strong>{SPEC_BY_NAME.origin.label}</strong>
                <div className="note">
                  {SPEC_BY_NAME.origin.hint} Click “place origin” on the canvas above.
                </div>
                <div className="mono">
                  {(() => {
                    const value = form.optional.origin?.value as number[] | null
                    return value ? `${value[0].toFixed(3)}, ${value[1].toFixed(3)}` : 'not placed'
                  })()}
                </div>
              </div>
            </div>

            {INPUT_ROLES.map((role) => (
              <ConstraintPicker
                key={role}
                role={role}
                field={form.optional[role]}
                jobs={jobs}
                partitions={partitions}
                onChange={(patch) => setOptional(role, patch)}
              />
            ))}

            <ParamFields
              params={form.params}
              sections={['constraints']}
              collapsed={[]}
              onChange={setParam}
            />
          </div>
        </details>
      </section>

      <section>
        <div className="section-head">
          <span className="eyebrow">4 · Budget</span>
        </div>
        <div className="field">
          <label htmlFor="horizon">Horizon</label>
          <div>
            <input
              id="horizon"
              type="number"
              min={1}
              value={form.numIter}
              onChange={(event) =>
                setForm((current) => ({ ...current, numIter: Number(event.target.value) }))
              }
            />
            <div className="note">
              The schedule. Every iteration's sparse-loss ramp is defined against it, so changing it
              later means a new job.
            </div>
          </div>
        </div>
        <div className="field">
          <label htmlFor="budget">This run's budget</label>
          <div>
            <input
              id="budget"
              type="number"
              min={1}
              max={form.numIter}
              value={form.targetEpoch}
              onChange={(event) =>
                setForm((current) => ({ ...current, targetEpoch: Number(event.target.value) }))
              }
            />
            <div className="note">
              Where this run stops. Stopping at {form.targetEpoch} of {form.numIter} is a preview
              you can promote later — not a shorter job.
              {estimate && ` About ${estimate} of GPU time.`}
            </div>
          </div>
        </div>

        {problems.length > 0 && (
          <ul className="warn" style={{ margin: '10px 0' }}>
            {problems.map((problem) => (
              <li key={problem}>{problem}</li>
            ))}
          </ul>
        )}

        <div className="btn-row" style={{ marginTop: 12 }}>
          <button
            type="button"
            className="btn btn--primary"
            disabled={!upload || busy || problems.length > 0}
            onClick={submit}
          >
            {busy ? 'Queueing…' : 'Queue job'}
          </button>
          {!upload && <span className="note">Choose a source image first.</span>}
        </div>
      </section>
    </div>
  )
}

export type { InputRef, JobSummary }
