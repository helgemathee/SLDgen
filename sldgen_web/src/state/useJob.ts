import { useCallback, useEffect, useState } from 'react'
import { api } from '../api/client'
import { subscribe } from '../api/stream'
import type { FramesResponse, JobDetail, Lineage, ProgressEvent } from '../api/types'

/**
 * One job, kept current.
 *
 * The SSE stream carries progress only; the detail object (segments, artifacts,
 * parameters) is refetched when the stream reports something that could have
 * changed it. Pushing the whole detail every second would be wasteful and would
 * also fight the log viewer's own stream for connections.
 */
export function useJob(jobId: string | null) {
  const [job, setJob] = useState<JobDetail | null>(null)
  const [frames, setFrames] = useState<FramesResponse | null>(null)
  const [lineage, setLineage] = useState<Lineage | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(() => {
    if (!jobId) return
    api
      .getJob(jobId)
      .then((detail) => {
        setJob(detail)
        setError(null)
      })
      .catch((cause) => setError(cause instanceof Error ? cause.message : String(cause)))
    api.frames(jobId).then(setFrames).catch(() => undefined)
    api.lineage(jobId).then(setLineage).catch(() => undefined)
  }, [jobId])

  useEffect(() => {
    if (!jobId) {
      setJob(null)
      setFrames(null)
      setLineage(null)
      return
    }
    setLoading(true)
    setJob(null)
    setFrames(null)
    refresh()
    setLoading(false)
  }, [jobId, refresh])

  useEffect(() => {
    if (!jobId) return
    let lastEpoch = -1
    let lastState = ''
    return subscribe<ProgressEvent>({
      url: `/api/jobs/${jobId}/events`,
      events: ['progress', 'settled'],
      poll: async () => {
        const detail = await api.getJob(jobId)
        return {
          id: detail.id,
          state: detail.state,
          desired_state: detail.desired_state,
          epoch: detail.current_epoch,
          target_epoch: detail.target_epoch,
          num_iter: detail.num_iter,
          phase: detail.state_json?.phase ?? null,
          iters_per_sec: detail.state_json?.iters_per_sec ?? null,
          resolved_caption: detail.resolved_caption,
          preview_url: detail.preview_url,
          error_class: detail.error_class,
        }
      },
      onData: (event) => {
        setJob((current) =>
          current && current.id === event.id
            ? {
                ...current,
                state: event.state,
                current_epoch: event.epoch,
                target_epoch: event.target_epoch,
                resolved_caption: event.resolved_caption,
                error_class: event.error_class,
                state_json: {
                  ...current.state_json,
                  phase: event.phase ?? undefined,
                  iters_per_sec: event.iters_per_sec ?? undefined,
                },
              }
            : current,
        )
        // A state change alters the segment list and the available artefacts; a
        // new save interval adds frames. Both need the fuller fetch.
        const crossedSave = event.epoch !== lastEpoch
        if (event.state !== lastState) {
          lastState = event.state
          refresh()
        } else if (crossedSave) {
          lastEpoch = event.epoch
          api.frames(event.id).then(setFrames).catch(() => undefined)
        }
      },
    })
  }, [jobId, refresh])

  return { job, frames, lineage, error, loading, refresh }
}
