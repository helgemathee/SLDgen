import { useCallback, useEffect, useState } from 'react'
import { fileUrl } from '../api/client'
import type { JobDetail } from '../api/types'
import {
  parseImageLossLog,
  sparkline,
  summarizeImageLoss,
  type ImageLossRow,
} from '../lib/imageloss'

const LOG_PATH = 'target/run/image_loss_log.csv'

/**
 * What the image fidelity term did (Spec 6 SS11.4), read from the run's own
 * `image_loss_log.csv`.
 *
 * The cosine answers "did the two terms fight": near 0 is the expected
 * unrelated pull, persistently negative means an unstable run, near 1 means the
 * image term was redundant. The SDS norm's spread is why the blend exists at
 * all: a loss-level weight would have wandered by that factor.
 */
export function ImageLossDiagnostics({ job }: { job: JobDetail }) {
  const listed = job.artifacts.some((artifact) => artifact.path === LOG_PATH)
  const [rows, setRows] = useState<ImageLossRow[] | null>(null)
  const [problem, setProblem] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const response = await fetch(fileUrl(job.id, LOG_PATH), { cache: 'no-store' })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      setRows(parseImageLossLog(await response.text()))
      setProblem(null)
    } catch (error) {
      setProblem(error instanceof Error ? error.message : 'Could not read the log')
    }
  }, [job.id])

  // Refetch when the job changes state (a segment ends, the run completes).
  useEffect(() => {
    if (listed) load()
  }, [listed, load, job.state])

  if (!listed) return null
  const summary = rows ? summarizeImageLoss(rows) : null
  const fmt = (value: number | null | undefined, digits = 3) =>
    value === null || value === undefined ? '—' : value.toPrecision(digits)

  return (
    <div className="panel">
      <div className="panel__head">
        <span className="eyebrow">Image fidelity</span>
        <button type="button" className="btn btn--small" onClick={load}>
          refresh
        </button>
      </div>
      <div className="panel__body">
        {problem && <div className="warn">{problem}</div>}
        {summary && summary.rows === 0 && <div className="note">No steps logged yet.</div>}
        {summary && summary.rows > 0 && rows && (
          <>
            <table className="table">
              <tbody>
                <tr>
                  <th>alpha</th>
                  <td className="mono">
                    {fmt(summary.alpha)} at epoch {summary.lastEpoch}
                  </td>
                  <td>
                    <Spark values={rows.map((row) => row.alpha)} range={[0, 1]} />
                  </td>
                </tr>
                <tr>
                  <th title="Between the raw SDS and image gradients">cosine</th>
                  <td className="mono">
                    mean {fmt(summary.cosineMean, 2)} · {fmt((summary.cosineNegativeShare ?? 0) * 100, 2)}%
                    negative
                  </td>
                  <td>
                    <Spark values={rows.map((row) => row.cosine)} range={[-1, 1]} />
                  </td>
                </tr>
                <tr>
                  <th title="How much the SDS gradient's size varies step to step">sds norm</th>
                  <td className="mono">
                    {summary.sdsNorm
                      ? `${fmt(summary.sdsNorm.min)} / ${fmt(summary.sdsNorm.median)} / ${fmt(summary.sdsNorm.max)}`
                      : '—'}
                  </td>
                  <td>
                    <Spark values={rows.map((row) => row.sds_norm)} />
                  </td>
                </tr>
                <tr>
                  <th>terms</th>
                  <td className="mono" colSpan={2}>
                    chamfer {fmt(summary.latest.chamfer)} px · pyramid {fmt(summary.latest.pyramid)} ·
                    landmark {fmt(summary.latest.landmark)} px
                  </td>
                </tr>
              </tbody>
            </table>
            {summary.cosineMean !== null && summary.cosineMean < -0.2 && (
              <div className="warn">
                The image and SDS gradients mostly pull against each other; expect an unstable
                drawing. Try a lower weight, or a decay schedule.
              </div>
            )}
            {summary.skippedShare > 0.1 && (
              <div className="note">
                {Math.round(summary.skippedShare * 100)}% of steps skipped the blend because one
                gradient vanished.
              </div>
            )}
            <div className="note">min / median / max. {summary.rows} steps logged.</div>
          </>
        )}
      </div>
    </div>
  )
}

function Spark({ values, range }: { values: (number | null)[]; range?: [number, number] }) {
  return (
    <svg width={90} height={20} role="img" aria-hidden="true">
      <polyline
        points={sparkline(values, 90, 20, range)}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.2}
      />
    </svg>
  )
}
