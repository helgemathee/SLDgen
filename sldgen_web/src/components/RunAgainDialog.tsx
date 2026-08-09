import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import type { JobDetail, JobSummary, ParamValue, Params } from '../api/types'
import { formatDuration, jobLabel, meanItersPerSec } from '../lib/format'
import { diffParams, paramLabel } from '../lib/paramdiff'
import { PARAM_SPECS, type ParamSection } from '../lib/params'
import {
  estimateBatchSeconds,
  findDuplicates,
  randomSeedBlock,
  sequentialSeeds,
  variantTitle,
  type Variant,
} from '../lib/variants'
import { useApp } from '../state/store'
import { ParamFields } from './ParamFields'

const SECTIONS: ParamSection[] = ['prompt', 'curve', 'guidance', 'constraints', 'losses', 'run']

/** Columns beyond seed and caption can be promoted into the table (SS6.6). */
const COLUMN_CANDIDATES = PARAM_SPECS.filter(
  (spec) => spec.group === 'structural' && !spec.optional && !['seed', 'caption'].includes(spec.name),
)

/**
 * Run again with changes (Spec 3 SS6.5) and batch variants (SS6.6).
 *
 * The most common action after looking at a result is *this, but with a
 * different caption* — and the more common case still is a handful of variants
 * queued together. So this is one panel: edit the shared parameters at the top,
 * and vary per row in the table below.
 *
 * Nothing needs retyping and nothing needs re-uploading: the new jobs reuse the
 * parent's source image, its prepared input, its mask and its inputs, which the
 * service copies again rather than referencing (Spec 2 SS4.3).
 */
export function RunAgainDialog({
  job,
  onClose,
  onQueued,
}: {
  job: JobDetail
  onClose: () => void
  onQueued: (ids: string[]) => void
}) {
  const { toast } = useApp()
  const parentParams = job.params
  const [base, setBase] = useState<Params>(() => ({ ...job.params }))
  const [targetEpoch, setTargetEpoch] = useState(job.target_epoch)
  const [columns, setColumns] = useState<string[]>(['seed', 'caption'])
  const [variants, setVariants] = useState<Variant[]>([])
  const [existing, setExisting] = useState<JobSummary[]>([])
  const [busy, setBusy] = useState(false)

  const parentSeed = Number(parentParams.seed ?? 0)

  const makeRow = (index: number, seed: number, params: Params): Variant => ({
    key: `${Date.now()}-${index}-${Math.round(seed)}`,
    enabled: true,
    params: { ...params, seed },
  })

  // Start with a single row: the plain "this, but different" case. Raising the
  // count fills the table, which is the batch case.
  useEffect(() => {
    setVariants([makeRow(0, parentSeed + 1, job.params)])
    // eslint-disable-next-line react-hooks/exhaustive-deps -- seeded once per job
  }, [job.id])

  // Needed to flag a seed that would duplicate an existing run.
  useEffect(() => {
    api
      .listJobs({ withParams: true, limit: 500 })
      .then((list) => setExisting(list.jobs))
      .catch(() => undefined)
  }, [])

  const changedNames = useMemo(() => diffParams(parentParams, base), [parentParams, base])
  const rate = useMemo(() => meanItersPerSec(job.segments), [job.segments])
  const enabledCount = variants.filter((variant) => variant.enabled).length
  const batchSeconds = estimateBatchSeconds(variants, targetEpoch, rate)

  const setCount = (count: number) => {
    const size = Math.max(1, Math.min(50, count))
    setVariants((current) => {
      if (size <= current.length) return current.slice(0, size)
      const seeds = sequentialSeeds(parentSeed, size)
      return [
        ...current,
        ...Array.from({ length: size - current.length }, (_unused, index) =>
          makeRow(current.length + index, seeds[current.length + index], base),
        ),
      ]
    })
  }

  const updateRow = (key: string, name: string, value: ParamValue) => {
    setVariants((current) =>
      current.map((variant) =>
        variant.key === key ? { ...variant, params: { ...variant.params, [name]: value } } : variant,
      ),
    )
  }

  const submit = async () => {
    setBusy(true)
    try {
      const payload = variants
        .filter((variant) => variant.enabled)
        .map((variant) => {
          // Row values override the shared edits, which override the parent.
          const params = { ...base, ...pick(variant.params, columns) }
          return {
            params,
            title: variantTitle(jobLabel(job), params, parentParams),
            target_epoch: Math.min(targetEpoch, Number(params.num_iter ?? job.num_iter)),
          }
        })
      const result = await api.runAgain(job.id, payload)
      toast(
        result.jobs.length === 1
          ? 'Queued one job.'
          : `Queued ${result.jobs.length} jobs as one batch.`,
      )
      onQueued(result.jobs.map((created) => created.id))
    } catch (error) {
      toast(error instanceof Error ? error.message : 'Could not queue')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="overlay" onClick={onClose} role="presentation">
      <div
        className="overlay__card overlay__card--wide"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-label="Run again with changes"
      >
        <div className="panel__head">
          <span className="eyebrow">Run again · from {jobLabel(job)}</span>
          <button type="button" className="btn btn--small" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="panel__body">
          <p className="note" style={{ marginTop: 0 }}>
            Same image, same prepared input, same mask and the same inputs — copied again, not
            referenced. Every new job starts from epoch 0.
          </p>

          <section style={{ marginBottom: 14 }}>
            <div className="section-head">
              <span className="eyebrow">Variants</span>
              <span className="header__spacer" style={{ flex: 1 }} />
              <label className="note">
                count{' '}
                <input
                  className="input"
                  style={{ width: 54, display: 'inline-block' }}
                  type="number"
                  min={1}
                  max={50}
                  value={variants.length}
                  onChange={(event) => setCount(Number(event.target.value))}
                />
              </label>
              <button
                type="button"
                className="btn btn--small"
                onClick={() =>
                  setVariants((current) => [
                    ...current,
                    makeRow(current.length, parentSeed + current.length + 1, base),
                  ])
                }
              >
                + Add
              </button>
              <button
                type="button"
                className="btn btn--small"
                title="A random block, for when the sequential range has been used up"
                onClick={() => {
                  const seeds = randomSeedBlock(variants.length)
                  setVariants((current) =>
                    current.map((variant, index) => ({
                      ...variant,
                      params: { ...variant.params, seed: seeds[index] },
                    })),
                  )
                }}
              >
                Regenerate seeds
              </button>
              <select
                className="input"
                style={{ width: 150 }}
                value=""
                onChange={(event) => {
                  if (event.target.value) setColumns((current) => [...current, event.target.value])
                }}
              >
                <option value="">+ Add column…</option>
                {COLUMN_CANDIDATES.filter((spec) => !columns.includes(spec.name)).map((spec) => (
                  <option key={spec.name} value={spec.name}>
                    {spec.label}
                  </option>
                ))}
              </select>
            </div>

            <table className="table variants">
              <thead>
                <tr>
                  <th />
                  {columns.map((name) => (
                    <th key={name}>
                      {paramLabel(name)}
                      {!['seed', 'caption'].includes(name) && (
                        <button
                          type="button"
                          className="btn btn--small btn--ghost"
                          onClick={() =>
                            setColumns((current) => current.filter((value) => value !== name))
                          }
                        >
                          ✕
                        </button>
                      )}
                    </th>
                  ))}
                  <th />
                </tr>
              </thead>
              <tbody>
                {variants.map((variant, index) => {
                  const merged = { ...base, ...pick(variant.params, columns) }
                  const duplicates = findDuplicates(merged, existing)
                  return (
                    <tr key={variant.key} data-disabled={!variant.enabled}>
                      <td style={{ width: 1 }}>
                        <input
                          type="checkbox"
                          checked={variant.enabled}
                          aria-label={`Include variant ${index + 1}`}
                          onChange={(event) =>
                            setVariants((current) =>
                              current.map((entry) =>
                                entry.key === variant.key
                                  ? { ...entry, enabled: event.target.checked }
                                  : entry,
                              ),
                            )
                          }
                        />
                      </td>
                      {columns.map((name) => (
                        <td key={name}>
                          <input
                            type={name === 'seed' ? 'number' : 'text'}
                            value={String(variant.params[name] ?? base[name] ?? '')}
                            placeholder={name === 'caption' ? '(BLIP-2 captions it)' : undefined}
                            onChange={(event) =>
                              updateRow(
                                variant.key,
                                name,
                                name === 'seed' ? Number(event.target.value) : event.target.value,
                              )
                            }
                          />
                        </td>
                      ))}
                      <td style={{ width: 1, whiteSpace: 'nowrap' }}>
                        {duplicates.length > 0 && (
                          <span
                            className="dup"
                            title={`Same parameters as ${duplicates
                              .map((entry) => jobLabel(entry))
                              .join(', ')}`}
                          >
                            duplicate
                          </span>
                        )}{' '}
                        <button
                          type="button"
                          className="btn btn--small btn--ghost"
                          title="Duplicate this row"
                          onClick={() =>
                            setVariants((current) => {
                              const position = current.findIndex(
                                (entry) => entry.key === variant.key,
                              )
                              const copy = {
                                ...variant,
                                key: `${variant.key}-copy-${current.length}`,
                                params: { ...variant.params },
                              }
                              return [
                                ...current.slice(0, position + 1),
                                copy,
                                ...current.slice(position + 1),
                              ]
                            })
                          }
                        >
                          ⧉
                        </button>
                        <button
                          type="button"
                          className="btn btn--small btn--ghost"
                          title="Remove this row"
                          disabled={variants.length === 1}
                          onClick={() =>
                            setVariants((current) =>
                              current.filter((entry) => entry.key !== variant.key),
                            )
                          }
                        >
                          ✕
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            <p className="note">
              Seeds run sequentially from the parent's, so a family is recognisable at a glance and
              re-running the batch tomorrow will not silently collide. An empty caption means BLIP-2
              captions the image again — a real option, but a choice.
            </p>
          </section>

          <section>
            <div className="section-head">
              <span className="eyebrow">Shared parameters</span>
              {changedNames.length > 0 && (
                <span className="mono">
                  {changedNames.length} changed: {changedNames.map(paramLabel).join(', ')}
                </span>
              )}
            </div>
            <div className="field">
              <label htmlFor="run-again-budget">Budget for each run</label>
              <div>
                <input
                  id="run-again-budget"
                  className="input"
                  type="number"
                  min={1}
                  max={Number(base.num_iter ?? job.num_iter)}
                  value={targetEpoch}
                  onChange={(event) => setTargetEpoch(Number(event.target.value))}
                />
                <div className="note">
                  Where each run stops. The horizon below sets the schedule it stops partway
                  through.
                </div>
              </div>
            </div>
            <ParamFields
              params={base}
              changedAgainst={parentParams}
              sections={SECTIONS}
              hide={columns.filter((name) => name !== 'caption')}
              onChange={(name, value) => setBase((current) => ({ ...current, [name]: value }))}
            />
          </section>
        </div>

        <div className="panel__head" style={{ borderTop: '1px solid var(--rule)', borderBottom: 'none' }}>
          <span className="mono">
            {enabledCount} variant{enabledCount === 1 ? '' : 's'} · {targetEpoch} iterations each
            {batchSeconds !== null ? ` · ~${formatDuration(batchSeconds)} queued` : ''}
          </span>
          <button
            type="button"
            className="btn btn--primary"
            disabled={busy || enabledCount === 0}
            onClick={submit}
          >
            {enabledCount === 1
              ? `Queue job${changedNames.length ? ` (${changedNames.length} changes)` : ''}`
              : `Queue ${enabledCount} jobs`}
          </button>
        </div>
      </div>
    </div>
  )
}

function pick(params: Params, names: string[]): Params {
  return Object.fromEntries(
    names.filter((name) => params[name] !== undefined).map((name) => [name, params[name]]),
  )
}
