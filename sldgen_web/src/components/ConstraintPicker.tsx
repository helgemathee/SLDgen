import { useRef, useState } from 'react'
import { api } from '../api/client'
import type { JobSummary, Partition } from '../api/types'
import type { InputRef, OptionalField } from '../lib/formstate'
import { jobLabel } from '../lib/format'
import { SPEC_BY_NAME } from '../lib/params'
import { refKey, refLabel, searchSources, uploadRef } from '../lib/sources'
import { FramePicker } from './FramePicker'
import { JobThumb } from './JobThumb'

/**
 * Picks the files that feed `--avoid`, `--attract`, `--init-points` and
 * `--stipple-weight` (Spec 3 SS8.3).
 *
 * Three ways in, because "route this curve around that geometry" arrives in
 * three different shapes:
 *
 *   * **another job** — searchable by title, caption or id, and then down to a
 *     particular iteration of it, since the run you want to avoid is often
 *     better at 1700 than at 4000 (see `FramePicker`);
 *   * **a committed partition**, unchanged;
 *   * **an uploaded SVG**, for geometry drawn elsewhere.
 *
 * Which frames may be used is decided by the service, not here: a job whose
 * intermediates are in a different coordinate space than its final SVG has
 * them refused at submission, and the picker does not offer them.
 *
 * The row keeps its value when switched off (SS9), so turning a constraint back
 * on does not mean finding the source job again.
 */
export function ConstraintPicker({
  role,
  field,
  jobs,
  partitions,
  onChange,
  onError,
}: {
  role: 'avoid' | 'attract' | 'init_points' | 'stipple_weight'
  field: OptionalField | undefined
  jobs: JobSummary[]
  partitions: Partition[]
  onChange: (patch: Partial<OptionalField>) => void
  onError?: (message: string) => void
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [picking, setPicking] = useState<JobSummary | null>(null)
  const [uploading, setUploading] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)
  const enabled = field?.enabled ?? false
  const chosen = field?.inputs ?? []
  const multiple = role === 'avoid' || role === 'attract'
  const spec = SPEC_BY_NAME[role]

  const candidates = searchSources(jobs, query)

  const add = (reference: InputRef) => {
    onChange({
      enabled: true,
      inputs: multiple ? [...chosen, reference] : [reference],
    })
    setPicking(null)
    if (!multiple) setOpen(false)
  }

  const receiveSvg = async (file: File) => {
    setUploading(true)
    try {
      const result = await api.upload(file, file.name)
      add(uploadRef(result, file.name))
    } catch (problem) {
      onError?.(problem instanceof Error ? problem.message : 'Could not upload that SVG')
    } finally {
      setUploading(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  return (
    <div className={`optional${enabled ? '' : ' optional--off'}`}>
      <input
        type="checkbox"
        checked={enabled}
        aria-label={`Use ${spec.label}`}
        onChange={(event) => onChange({ enabled: event.target.checked })}
      />
      <div className="optional__body">
        <strong>{spec.label}</strong>
        <div className="note">
          {role === 'stipple_weight'
            ? 'A grayscale image. The prep canvas produces one for you in the guide and control modes.'
            : 'Any job’s final SVG or one of its frames, a committed partition, or an SVG you upload.'}
        </div>

        {chosen.length > 0 && (
          <ul className="mono" style={{ margin: '4px 0', paddingLeft: 16 }}>
            {chosen.map((reference, index) => (
              <li key={`${refKey(reference)}-${index}`}>
                {refLabel(reference)}{' '}
                <button
                  type="button"
                  className="btn btn--small btn--ghost"
                  aria-label={`Remove ${refLabel(reference)}`}
                  onClick={() =>
                    onChange({ inputs: chosen.filter((_unused, position) => position !== index) })
                  }
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        )}

        <button type="button" className="btn btn--small" onClick={() => setOpen((value) => !value)}>
          {open ? 'Close' : chosen.length > 0 && !multiple ? 'Change…' : 'Choose…'}
        </button>

        {open && (
          <div className="panel" style={{ marginTop: 6 }}>
            <div className="panel__body">
              {role !== 'stipple_weight' && (
                <>
                  {picking ? (
                    <FramePicker job={picking} onPick={add} onBack={() => setPicking(null)} />
                  ) : (
                    <div className="picker">
                      <div className="picker__head">
                        <input
                          className="input"
                          style={{ flex: 1 }}
                          placeholder="Search jobs by title, caption or id"
                          value={query}
                          aria-label={`Search source jobs for ${spec.label}`}
                          onChange={(event) => setQuery(event.target.value)}
                        />
                        <button
                          type="button"
                          className="btn btn--small"
                          disabled={uploading}
                          onClick={() => fileInput.current?.click()}
                        >
                          {uploading ? 'Uploading…' : 'Upload SVG…'}
                        </button>
                        <input
                          ref={fileInput}
                          type="file"
                          accept=".svg,image/svg+xml"
                          hidden
                          onChange={(event) => {
                            const file = event.target.files?.[0]
                            if (file) receiveSvg(file)
                          }}
                        />
                      </div>
                      <div className="note" style={{ margin: '0 0 6px' }}>
                        An uploaded SVG is used as-is, in canvas pixels at the render size — nothing
                        can check that it registers with your target, so draw it on the same canvas.
                      </div>

                      <div className="eyebrow" style={{ marginBottom: 6 }}>
                        Jobs {query && `· ${candidates.length} match`}
                      </div>
                      {candidates.length === 0 ? (
                        <div className="note">
                          {query
                            ? 'Nothing matches that.'
                            : 'Nothing has drawn a frame yet, so there is no SVG to use.'}
                        </div>
                      ) : (
                        <div className="recent">
                          {candidates.map((job) => (
                            <button
                              key={job.id}
                              type="button"
                              title={`${jobLabel(job)} — pick an iteration`}
                              onClick={() => setPicking(job)}
                            >
                              <JobThumb job={job} alt={jobLabel(job)} />
                            </button>
                          ))}
                        </div>
                      )}

                      <div className="eyebrow" style={{ margin: '10px 0 6px' }}>
                        Committed partitions
                      </div>
                      {partitions.length === 0 ? (
                        <div className="note">
                          None yet. Commit a partition on a completed job to make its pieces
                          available here.
                        </div>
                      ) : (
                        <div style={{ display: 'grid', gap: 3 }}>
                          {partitions.map((partition) =>
                            Array.from({ length: partition.n }, (_unused, index) => (
                              <button
                                key={`${partition.id}-${index}`}
                                type="button"
                                className="btn btn--small"
                                style={{ textAlign: 'left' }}
                                onClick={() =>
                                  add({
                                    source_kind: 'partition',
                                    source_partition_id: partition.id,
                                    path: `partition_${index}.svg`,
                                    label: `${partition.strategy}×${partition.n} · partition_${index}.svg`,
                                  })
                                }
                              >
                                {partition.strategy}×{partition.n} · partition_{index}.svg
                              </button>
                            )),
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}

              {role === 'stipple_weight' && (
                <div className="note">
                  Choose “Guide the ink” or “Control the ink” above and paint with the density
                  brush; the weight image is generated from what you paint. Uploading one directly
                  is not offered here, because a hand-made weight map has no way to register with
                  the prepared target.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
