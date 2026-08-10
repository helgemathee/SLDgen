import { useRef, useState } from 'react'
import type { JobDetail, Segment } from '../api/types'
import { copyText } from '../lib/clipboard'
import { formatDuration, formatTimestamp } from '../lib/format'
import { PARAM_SPECS, SECTION_LABELS, formatParamValue } from '../lib/params'

const COPY_LABEL = { idle: 'copy command', copied: 'copied', failed: 'copy failed' } as const

/**
 * What was actually run (Spec 3 SS6.4).
 *
 * Read-only, always: parameters define this job and cannot be edited once it
 * exists (Spec 2 SS4.2), so the panel's one action is `Run again with changes…`
 * rather than a set of disabled-looking inputs that hint otherwise. The four
 * operational settings are the exception and are marked as such.
 */
export function ParamTable({ job, onRunAgain }: { job: JobDetail; onRunAgain: () => void }) {
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle')
  const commandDetails = useRef<HTMLDetailsElement>(null)
  const sections = Array.from(new Set(PARAM_SPECS.map((spec) => spec.section)))

  return (
    <div className="panel">
      <div className="panel__head">
        <span className="eyebrow">Parameters</span>
        <button
          type="button"
          className="btn btn--small"
          onClick={() => {
            copyText(job.command).then((ok) => {
              setCopyState(ok ? 'copied' : 'failed')
              // Nothing reached the clipboard, so put the command on screen
              // where it can at least be selected by hand.
              if (!ok && commandDetails.current) commandDetails.current.open = true
              setTimeout(() => setCopyState('idle'), 1500)
            })
          }}
        >
          {COPY_LABEL[copyState]}
        </button>
        <button type="button" className="btn btn--small btn--primary" onClick={onRunAgain}>
          Run again with changes…
        </button>
      </div>
      <div className="panel__body">
        {sections.map((section) => {
          const specs = PARAM_SPECS.filter((spec) => spec.section === section)
          return (
            <details key={section} className="group" open={section !== 'losses'}>
              <summary>
                <span className="eyebrow">{SECTION_LABELS[section]}</span>
              </summary>
              <div className="group__body">
                <table className="table">
                  <tbody>
                    {specs.map((spec) => (
                      <tr key={spec.name}>
                        <th>
                          {spec.name}
                          {spec.group === 'operational' && (
                            <span className="muted" title="Editable at any time; never affects the result">
                              {' '}
                              ·op
                            </span>
                          )}
                        </th>
                        <td>{formatParamValue(job.params[spec.name] ?? null)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )
        })}

        <details className="group" ref={commandDetails}>
          <summary>
            <span className="eyebrow">Reproduction command</span>
          </summary>
          <div className="group__body">
            <div className="mono" style={{ wordBreak: 'break-all', lineHeight: 1.6 }}>
              {job.command}
            </div>
          </div>
        </details>
      </div>
    </div>
  )
}

/**
 * One row per SLDgen invocation (Spec 2 SS13.3).
 *
 * Presented as an ordered set rather than concatenated into one stream, because
 * each segment has its own argv header and its own exit status.
 */
export function SegmentList({
  job,
  selected,
  onSelect,
}: {
  job: JobDetail
  selected: number | null
  onSelect: (seq: number) => void
}) {
  if (job.segments.length === 0) {
    return (
      <div className="panel">
        <div className="panel__head">
          <span className="eyebrow">Segments</span>
        </div>
        <div className="panel__body note">
          Nothing has run yet. A segment appears when the worker claims this job.
        </div>
      </div>
    )
  }

  return (
    <div className="panel">
      <div className="panel__head">
        <span className="eyebrow">Segments</span>
        <span className="note">Each row opens that segment's log.</span>
      </div>
      <div className="panel__body">
        <table className="table table--rows">
          <thead>
            <tr>
              <th>#</th>
              <th>epochs</th>
              <th>exit</th>
              <th>wall</th>
              <th>it/s</th>
              <th>started</th>
            </tr>
          </thead>
          <tbody>
            {[...job.segments].reverse().map((segment) => (
              <tr
                key={segment.id}
                data-clickable="true"
                aria-selected={segment.seq === selected}
                style={
                  segment.seq === selected ? { background: 'var(--paper-sunk)' } : undefined
                }
                onClick={() => onSelect(segment.seq)}
              >
                <td>{segment.seq}</td>
                <td>
                  {segment.start_epoch}→{segment.end_epoch ?? segment.stop_at}
                  {segment.end_epoch === null && ' …'}
                </td>
                <td className={segment.error_class ? 'state-failed' : undefined}>
                  {segment.finished_at === null
                    ? 'running'
                    : segment.exit_code === null
                      ? 'adopted'
                      : segment.error_class ?? segment.exit_code}
                </td>
                <td>{wallTime(segment)}</td>
                <td>{rate(segment)}</td>
                <td>{formatTimestamp(segment.started_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function wallTime(segment: Segment): string {
  if (!segment.finished_at) return '—'
  return formatDuration(
    (Date.parse(segment.finished_at) - Date.parse(segment.started_at)) / 1000,
  )
}

function rate(segment: Segment): string {
  if (!segment.finished_at || segment.end_epoch === null) return '—'
  const seconds = (Date.parse(segment.finished_at) - Date.parse(segment.started_at)) / 1000
  const epochs = segment.end_epoch - segment.start_epoch
  if (seconds <= 0 || epochs <= 0) return '—'
  return (epochs / seconds).toFixed(1)
}

/** Where a job came from, and what came out of it (Spec 3 SS6.5). */
export function LineagePanel({
  lineage,
  onOpen,
  onCompare,
}: {
  lineage: { parent: { id: string; title: string | null } | null; variants: { id: string }[] }
  onOpen: (id: string) => void
  onCompare: (ids: string[]) => void
}) {
  if (!lineage.parent && lineage.variants.length === 0) return null
  return (
    <div className="note" style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
      {lineage.parent && (
        <span>
          derived from{' '}
          <button
            type="button"
            className="btn btn--small btn--ghost"
            onClick={() => onOpen(lineage.parent!.id)}
          >
            {lineage.parent.title ?? lineage.parent.id.slice(-6)}
          </button>
        </span>
      )}
      {lineage.variants.length > 0 && (
        <span>
          {lineage.variants.length} variant{lineage.variants.length === 1 ? '' : 's'}
          <button
            type="button"
            className="btn btn--small btn--ghost"
            onClick={() => onCompare(lineage.variants.map((variant) => variant.id))}
          >
            compare
          </button>
        </span>
      )}
    </div>
  )
}
