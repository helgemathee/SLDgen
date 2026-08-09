import { useState } from 'react'
import type { JobSummary, Partition } from '../api/types'
import type { InputRef, OptionalField } from '../lib/formstate'
import { jobLabel } from '../lib/format'
import { SPEC_BY_NAME } from '../lib/params'

/**
 * Picks the files that feed `--avoid`, `--attract`, `--init-points` and
 * `--stipple-weight` (Spec 3 SS8.3).
 *
 * Restricted to `final_sld.svg` of completed jobs and to committed partitions.
 * That restriction is the UI half of Spec 1 SS7: intermediate SVGs from a run
 * that rescaled its object are in a different coordinate space and would
 * misregister silently. The API refuses them too — this just means you never
 * get far enough to be refused.
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
}: {
  role: 'avoid' | 'attract' | 'init_points' | 'stipple_weight'
  field: OptionalField | undefined
  jobs: JobSummary[]
  partitions: Partition[]
  onChange: (patch: Partial<OptionalField>) => void
}) {
  const [open, setOpen] = useState(false)
  const enabled = field?.enabled ?? false
  const chosen = field?.inputs ?? []
  const multiple = role === 'avoid' || role === 'attract'
  const spec = SPEC_BY_NAME[role]

  // Only a finished run has a final_sld.svg to offer.
  const candidates = jobs.filter((job) => job.state === 'complete')

  const add = (reference: InputRef) => {
    onChange({
      enabled: true,
      inputs: multiple ? [...chosen, reference] : [reference],
    })
    if (!multiple) setOpen(false)
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
            : 'Only final SVGs of completed jobs and committed partitions — intermediates are in a different coordinate space.'}
        </div>

        {chosen.length > 0 && (
          <ul className="mono" style={{ margin: '4px 0', paddingLeft: 16 }}>
            {chosen.map((reference, index) => (
              <li key={`${reference.source_job_id ?? reference.source_partition_id}-${index}`}>
                {reference.label ?? reference.path}{' '}
                <button
                  type="button"
                  className="btn btn--small btn--ghost"
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
                  <div className="eyebrow" style={{ marginBottom: 6 }}>
                    Completed jobs
                  </div>
                  {candidates.length === 0 ? (
                    <div className="note">
                      Nothing has reached its horizon yet, so there is no final SVG to use.
                    </div>
                  ) : (
                    <div className="recent">
                      {candidates.map((job) => (
                        <button
                          key={job.id}
                          type="button"
                          title={jobLabel(job)}
                          onClick={() =>
                            add({
                              source_kind: 'job',
                              source_job_id: job.id,
                              path: 'final_sld.svg',
                              label: `${jobLabel(job)} · final_sld.svg`,
                            })
                          }
                        >
                          <img src={job.preview_url} alt={jobLabel(job)} />
                        </button>
                      ))}
                    </div>
                  )}

                  <div className="eyebrow" style={{ margin: '10px 0 6px' }}>
                    Committed partitions
                  </div>
                  {partitions.length === 0 ? (
                    <div className="note">
                      None yet. Commit a partition on a completed job to make its pieces available
                      here.
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
