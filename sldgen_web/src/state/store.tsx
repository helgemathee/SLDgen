import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { api } from '../api/client'
import { subscribe, type ConnectionState } from '../api/stream'
import type { DiskReport, Health, JobState, JobSummary, JobsEvent } from '../api/types'

interface AppState {
  jobs: JobSummary[]
  jobsById: Map<string, JobSummary>
  health: Health | null
  connection: ConnectionState
  queueDepth: number
  /** The rail's state filter. Lifted here so the status bar can drive it:
   *  clicking the queue depth filters the rail to `queued` (Spec 3 SS10). */
  stateFilter: Set<JobState>
  setStateFilter: (states: Set<JobState>) => void
  /** Total bytes under the work root, and the growth since this session opened. */
  disk: DiskReport | null
  diskDelta: number
  selection: string[]
  setSelection: (ids: string[]) => void
  toggleSelected: (id: string, additive: boolean) => void
  refreshJobs: () => void
  refreshDisk: (recompute?: boolean) => void
  toast: (message: string) => void
  message: string | null
}

const Context = createContext<AppState | null>(null)

export function useApp(): AppState {
  const value = useContext(Context)
  if (!value) throw new Error('useApp must be used inside <AppProvider>')
  return value
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [health, setHealth] = useState<Health | null>(null)
  const [connection, setConnection] = useState<ConnectionState>('connecting')
  const [queueDepth, setQueueDepth] = useState(0)
  const [disk, setDisk] = useState<DiskReport | null>(null)
  const [selection, setSelection] = useState<string[]>([])
  const [stateFilter, setStateFilter] = useState<Set<JobState>>(() => new Set())
  const [message, setMessage] = useState<string | null>(null)
  // Captured once, so the status bar can show growth without arithmetic (SS10).
  const baselineDisk = useRef<number | null>(null)
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const applyJobsEvent = useCallback((payload: JobsEvent) => {
    setJobs(payload.jobs)
    setQueueDepth(payload.queue_depth)
    setHealth((current) =>
      current ? { ...current, worker_alive: payload.worker_alive } : current,
    )
  }, [])

  // The rail's live view: one SSE connection for the whole list, with polling
  // as the fallback that must keep working (SS13).
  useEffect(
    () =>
      subscribe<JobsEvent>({
        url: '/api/events',
        events: ['jobs'],
        onData: applyJobsEvent,
        onState: setConnection,
        poll: async () => {
          const [list, current] = await Promise.all([api.listJobs(), api.health()])
          return {
            jobs: list.jobs,
            worker_alive: current.worker_alive,
            queue_depth: list.jobs.filter((job) => job.state === 'queued').length,
          }
        },
      }),
    [applyJobsEvent],
  )

  // Health carries gpu_free_mb, which the stream deliberately does not: it costs
  // an nvidia-smi call and is diagnostic only (Spec 2 SS9), so 10 s is plenty.
  useEffect(() => {
    let cancelled = false
    const tick = () => {
      api
        .health()
        .then((value) => {
          if (!cancelled) setHealth(value)
        })
        .catch(() => {
          if (!cancelled) setHealth(null)
        })
    }
    tick()
    const timer = setInterval(tick, 10_000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [])

  const refreshDisk = useCallback((recompute = false) => {
    api
      .disk(recompute)
      .then((report) => {
        setDisk(report)
        if (baselineDisk.current === null) baselineDisk.current = report.total_bytes
      })
      .catch(() => undefined)
  }, [])

  useEffect(() => {
    refreshDisk()
    const timer = setInterval(() => refreshDisk(), 30_000)
    return () => clearInterval(timer)
  }, [refreshDisk])

  const refreshJobs = useCallback(() => {
    api
      .listJobs()
      .then((list) => setJobs(list.jobs))
      .catch(() => undefined)
  }, [])

  const toast = useCallback((text: string) => {
    setMessage(text)
    if (toastTimer.current) clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setMessage(null), 4000)
  }, [])

  const toggleSelected = useCallback((id: string, additive: boolean) => {
    setSelection((current) => {
      if (!additive) return current.length === 1 && current[0] === id ? [] : [id]
      return current.includes(id)
        ? current.filter((value) => value !== id)
        : [...current, id]
    })
  }, [])

  // Jobs that vanish (deleted) must not linger in the compare selection.
  useEffect(() => {
    setSelection((current) => {
      const live = new Set(jobs.map((job) => job.id))
      const kept = current.filter((id) => live.has(id))
      return kept.length === current.length ? current : kept
    })
  }, [jobs])

  const value = useMemo<AppState>(
    () => ({
      jobs,
      jobsById: new Map(jobs.map((job) => [job.id, job])),
      health,
      connection,
      queueDepth,
      stateFilter,
      setStateFilter,
      disk,
      diskDelta:
        disk && baselineDisk.current !== null ? disk.total_bytes - baselineDisk.current : 0,
      selection,
      setSelection,
      toggleSelected,
      refreshJobs,
      refreshDisk,
      toast,
      message,
    }),
    [
      jobs,
      health,
      connection,
      queueDepth,
      stateFilter,
      disk,
      selection,
      toggleSelected,
      refreshJobs,
      refreshDisk,
      toast,
      message,
    ],
  )

  return <Context.Provider value={value}>{children}</Context.Provider>
}
