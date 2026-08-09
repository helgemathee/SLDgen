import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { JobDetail, Partition, PartitionPreview } from '../api/types'
import { useApp } from '../state/store'

const STRATEGIES = ['sequence', 'horizontal', 'vertical', 'radial', 'cluster', 'labelmap'] as const
type Strategy = (typeof STRATEGIES)[number]

/**
 * Partitioning (Spec 3 SS12).
 *
 * CPU-only and synchronous (Spec 2 SS11), so parameters drive a *live preview*
 * rather than a submission: scrub the strategy, see the split, commit when it is
 * right. This is the one place colour appears outside job state, and it is
 * functional — the colours are the partition identity.
 *
 * Committing writes a `partitions` row, and committed partitions then appear in
 * the constraint picker as attract and avoid sources, which is how the
 * sequential compositional workflow closes the loop.
 */
export function PartitionPanel({ job }: { job: JobDetail }) {
  const { toast } = useApp()
  const [strategy, setStrategy] = useState<Strategy>('sequence')
  const [count, setCount] = useState(3)
  const [connectTails, setConnectTails] = useState(false)
  const [sampleSpacing, setSampleSpacing] = useState<number | ''>('')
  const [preview, setPreview] = useState<PartitionPreview | null>(null)
  const [committed, setCommitted] = useState<Partition[]>([])
  const [busy, setBusy] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
  const generation = useRef(0)

  useEffect(() => {
    api
      .listPartitions(job.id)
      .then((result) => setCommitted(result.partitions))
      .catch(() => undefined)
  }, [job.id])

  // Debounced: dragging N from 3 to 8 should run the script once at the end, not
  // six times, since each run is a subprocess of a few seconds.
  useEffect(() => {
    if (!open) return
    const token = ++generation.current
    const timer = setTimeout(() => {
      setBusy(true)
      setProblem(null)
      api
        .partitionPreview({
          source_job_id: job.id,
          strategy,
          n: count,
          params: {
            connect_tails: connectTails,
            ...(sampleSpacing === '' ? {} : { sample_spacing: sampleSpacing }),
          },
        })
        .then((result) => {
          if (token === generation.current) setPreview(result)
        })
        .catch((error) => {
          if (token === generation.current) {
            setProblem(error instanceof Error ? error.message : 'Preview failed')
            setPreview(null)
          }
        })
        .finally(() => {
          if (token === generation.current) setBusy(false)
        })
    }, 350)
    return () => clearTimeout(timer)
  }, [open, job.id, strategy, count, connectTails, sampleSpacing])

  const commit = async () => {
    setBusy(true)
    try {
      const result = await api.commitPartition({
        source_job_id: job.id,
        strategy,
        n: count,
        params: {
          connect_tails: connectTails,
          ...(sampleSpacing === '' ? {} : { sample_spacing: sampleSpacing }),
        },
      })
      toast(`Committed ${count} partitions.`)
      setCommitted((current) => [result, ...current])
    } catch (error) {
      toast(error instanceof Error ? error.message : 'Could not commit')
    } finally {
      setBusy(false)
    }
  }

  return (
    <details className="panel" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="panel__head" style={{ cursor: 'pointer', listStyle: 'none' }}>
        <span className="eyebrow">Partitions</span>
        <span className="note">
          {committed.length > 0 ? `${committed.length} committed` : 'Split this curve'}
        </span>
      </summary>

      <div className="panel__body" style={{ display: 'grid', gap: 10 }}>
        <div className="field">
          <label htmlFor="partition-strategy">Strategy</label>
          <select
            id="partition-strategy"
            value={strategy}
            onChange={(event) => setStrategy(event.target.value as Strategy)}
          >
            {STRATEGIES.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="partition-n">Partitions</label>
          <input
            id="partition-n"
            type="number"
            min={2}
            max={24}
            value={count}
            onChange={(event) => setCount(Number(event.target.value))}
          />
        </div>
        <div className="field">
          <label htmlFor="partition-spacing">Sample spacing</label>
          <input
            id="partition-spacing"
            type="number"
            step="0.5"
            placeholder="(script default)"
            value={sampleSpacing}
            onChange={(event) =>
              setSampleSpacing(event.target.value === '' ? '' : Number(event.target.value))
            }
          />
        </div>
        <label className="note" style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <input
            type="checkbox"
            checked={connectTails}
            onChange={(event) => setConnectTails(event.target.checked)}
          />
          connect tails
        </label>

        {strategy === 'labelmap' && (
          <p className="note">
            Labels default to this job's own condition image, which is already in canvas space.
          </p>
        )}

        <div
          className="canvas-wrap"
          style={{ minHeight: 200, opacity: busy ? 0.55 : 1 }}
          aria-busy={busy}
        >
          {preview?.preview_url ? (
            <img
              src={`${preview.preview_url}?v=${strategy}-${count}-${connectTails}-${sampleSpacing}`}
              alt="Partition preview"
              style={{ maxWidth: '100%', maxHeight: 320 }}
            />
          ) : (
            <span className="muted note">{busy ? 'Running…' : problem ?? 'No preview yet.'}</span>
          )}
        </div>

        {problem && <div className="warn">{problem}</div>}

        <div className="btn-row">
          <button
            type="button"
            className="btn btn--primary"
            disabled={busy || !preview}
            onClick={commit}
          >
            Commit these {count}
          </button>
          <span className="note">
            Committed partitions become attract and avoid sources for the next job.
          </span>
        </div>

        {committed.length > 0 && (
          <table className="table">
            <tbody>
              {committed.map((partition) => (
                <tr key={partition.id}>
                  <th>
                    {partition.strategy} × {partition.n}
                  </th>
                  <td>
                    <a href={`/api/partitions/${partition.id}/download.zip`} download>
                      download zip
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </details>
  )
}
