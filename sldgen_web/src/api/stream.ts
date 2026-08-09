import { SSE_FAILURES_BEFORE_POLLING, backoffDelay } from '../lib/backoff'

export type ConnectionState = 'connecting' | 'live' | 'reconnecting' | 'polling'

export interface StreamOptions<T> {
  /** SSE endpoint. */
  url: string
  /** Event names to listen for. */
  events: string[]
  /** The polling fallback. Must return the same shape the SSE event carries. */
  poll: () => Promise<T>
  onData: (data: T) => void
  onState?: (state: ConnectionState) => void
  pollInterval?: number
}

/**
 * One live view of a server-side resource, by SSE where possible and by polling
 * where not.
 *
 * Two rules from Spec 3 SS13 are enforced here rather than left to callers:
 *
 * - **On reconnect, refetch rather than assuming continuity.** Every transition
 *   back to a working connection runs `poll()` once, so a client that missed
 *   events while disconnected is corrected instead of drifting.
 * - **Every SSE-fed view must also work under plain polling.** The fallback is
 *   not a degraded mode bolted on afterwards; it is the same `onData` path, so
 *   it cannot rot from disuse.
 */
export function subscribe<T>(options: StreamOptions<T>): () => void {
  const { url, events, poll, onData, onState, pollInterval = 2000 } = options
  let closed = false
  let source: EventSource | null = null
  let timer: ReturnType<typeof setTimeout> | null = null
  let pollTimer: ReturnType<typeof setInterval> | null = null
  let failures = 0

  const setState = (state: ConnectionState) => {
    if (!closed) onState?.(state)
  }

  const refetch = () => {
    poll()
      .then((data) => {
        if (!closed) onData(data)
      })
      .catch(() => {
        // A failed refetch is not fatal: the stream (or the next poll tick) will
        // deliver the same state shortly.
      })
  }

  const startPolling = () => {
    if (pollTimer !== null) return
    setState('polling')
    refetch()
    pollTimer = setInterval(refetch, pollInterval)
  }

  const stopPolling = () => {
    if (pollTimer !== null) {
      clearInterval(pollTimer)
      pollTimer = null
    }
  }

  const connect = () => {
    if (closed) return
    setState(failures === 0 ? 'connecting' : 'reconnecting')
    source = new EventSource(url)

    source.onopen = () => {
      failures = 0
      stopPolling()
      setState('live')
      // Refetch on every (re)connect: the stream only reports what happens from
      // now on, and we may have missed a transition.
      refetch()
    }

    for (const name of events) {
      source.addEventListener(name, (event) => {
        if (closed) return
        try {
          onData(JSON.parse((event as MessageEvent).data) as T)
        } catch {
          // A truncated frame is discarded; the next one carries full state.
        }
      })
    }

    source.onerror = () => {
      source?.close()
      source = null
      if (closed) return
      failures += 1
      if (failures >= SSE_FAILURES_BEFORE_POLLING) {
        // Stop fighting the proxy. Polling keeps the view correct, just slower.
        startPolling()
        return
      }
      setState('reconnecting')
      timer = setTimeout(connect, backoffDelay(failures))
    }
  }

  connect()

  return () => {
    closed = true
    source?.close()
    if (timer !== null) clearTimeout(timer)
    stopPolling()
  }
}
