import type {
  CleanupResult,
  DiskReport,
  FavoritesResponse,
  CannyPreview,
  FramesResponse,
  Health,
  JobDetail,
  JobSummary,
  Lineage,
  LogChunk,
  Params,
  Partition,
  PartitionPreview,
  Preset,
  UploadResult,
} from './types'

/** A non-2xx response, carrying the body so callers can show what the API said. */
export class ApiError extends Error {
  status: number
  body: unknown

  constructor(status: number, message: string, body: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData)
        ? { 'Content-Type': 'application/json' }
        : {}),
      ...init?.headers,
    },
  })
  if (!response.ok) {
    let body: unknown = null
    let message = `${response.status} ${response.statusText}`
    try {
      body = await response.json()
      const detail = (body as { detail?: unknown }).detail
      if (typeof detail === 'string') message = detail
      else if (detail) message = JSON.stringify(detail)
    } catch {
      // A non-JSON error body (an html 502 from a proxy, say) is not worth
      // failing over -- the status line already says enough.
    }
    throw new ApiError(response.status, message, body)
  }
  if (response.status === 204) return undefined as T
  const type = response.headers.get('content-type') ?? ''
  if (type.includes('application/json')) return (await response.json()) as T
  return (await response.text()) as unknown as T
}

const json = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) })

export const api = {
  health: () => request<Health>('/api/health'),
  settings: () => request<Record<string, unknown>>('/api/settings'),

  listJobs: (options: {
    state?: string
    batchId?: string
    parentJobId?: string
    withParams?: boolean
    limit?: number
  } = {}) => {
    const query = new URLSearchParams()
    if (options.state) query.set('state', options.state)
    if (options.batchId) query.set('batch_id', options.batchId)
    if (options.parentJobId) query.set('parent_job_id', options.parentJobId)
    if (options.withParams) query.set('with_params', 'true')
    query.set('limit', String(options.limit ?? 500))
    return request<{ jobs: JobSummary[]; next_cursor: string | null }>(`/api/jobs?${query}`)
  },

  getJob: (id: string) => request<JobDetail>(`/api/jobs/${id}`),
  frames: (id: string) => request<FramesResponse>(`/api/jobs/${id}/frames`),

  /** `null` means "follow the newest frame again" (Spec 3 SS6.2). */
  setViewedEpoch: (id: string, epoch: number | null) =>
    request<{ job_id: string; viewed_epoch: number | null }>(`/api/jobs/${id}/viewed-epoch`, {
      method: 'PUT',
      body: JSON.stringify({ epoch }),
    }),
  favorites: (id: string) => request<FavoritesResponse>(`/api/jobs/${id}/favorites`),
  addFavorite: (id: string, epoch: number) =>
    request<FavoritesResponse>(`/api/jobs/${id}/favorites/${epoch}`, { method: 'PUT' }),
  removeFavorite: (id: string, epoch: number) =>
    request<FavoritesResponse>(`/api/jobs/${id}/favorites/${epoch}`, { method: 'DELETE' }),

  lineage: (id: string) => request<Lineage>(`/api/jobs/${id}/lineage`),
  command: (id: string) => request<string>(`/api/jobs/${id}/command`),

  createJob: (body: {
    title?: string
    target_sha256: string
    params: Params
    target_epoch: number
    inputs?: unknown[]
  }) => request<JobDetail>('/api/jobs', json(body)),

  patchJob: (id: string, body: Record<string, unknown>) =>
    request<JobDetail>(`/api/jobs/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),

  pause: (id: string) => request<JobSummary>(`/api/jobs/${id}/pause`, { method: 'POST' }),
  resume: (id: string) => request<JobSummary>(`/api/jobs/${id}/resume`, { method: 'POST' }),
  retry: (id: string) => request<JobSummary>(`/api/jobs/${id}/retry`, { method: 'POST' }),
  promote: (id: string, targetEpoch: number) =>
    request<JobSummary>(`/api/jobs/${id}/promote`, json({ target_epoch: targetEpoch })),
  remove: (id: string) => request<unknown>(`/api/jobs/${id}`, { method: 'DELETE' }),

  runAgain: (id: string, variants: unknown[]) =>
    request<{ jobs: JobSummary[] }>(`/api/jobs/${id}/run-again`, json({ variants })),

  upload: async (file: File | Blob, filename = 'upload.png') => {
    const form = new FormData()
    form.append('file', file, filename)
    return request<UploadResult>('/api/uploads', { method: 'POST', body: form })
  },

  log: (id: string, options: { segment?: number; from?: number; raw?: boolean } = {}) => {
    const query = new URLSearchParams()
    if (options.segment !== undefined) query.set('segment', String(options.segment))
    query.set('from', String(options.from ?? 0))
    if (options.raw) query.set('raw', 'true')
    return request<LogChunk>(`/api/jobs/${id}/log?${query}`)
  },

  workerLog: (lines = 400) => request<string>(`/api/logs/worker?lines=${lines}`),

  disk: (recompute = false) =>
    request<DiskReport>(`/api/disk${recompute ? '?recompute=true' : ''}`),

  cleanup: (body: Record<string, unknown>) =>
    request<CleanupResult>('/api/maintenance/cleanup', json(body)),

  cannyPreview: (body: Record<string, unknown>) =>
    request<CannyPreview>('/api/canny/preview', json(body)),

  partitionPreview: (body: Record<string, unknown>) =>
    request<PartitionPreview>('/api/partitions/preview', json(body)),
  commitPartition: (body: Record<string, unknown>) =>
    request<Partition>('/api/partitions', json(body)),
  listPartitions: (sourceJobId?: string) =>
    request<{ partitions: Partition[] }>(
      `/api/partitions${sourceJobId ? `?source_job_id=${sourceJobId}` : ''}`,
    ),

  lastParams: () => request<{ params: unknown }>('/api/params/last'),
  saveLastParams: (params: unknown) =>
    request<{ params: unknown }>('/api/params/last', {
      method: 'PUT',
      body: JSON.stringify({ params }),
    }),
  presets: () => request<{ presets: Preset[] }>('/api/params/presets'),
  savePreset: (name: string, params: unknown) =>
    request<Preset>('/api/params/presets', json({ name, params })),
  deletePreset: (id: string) =>
    request<void>(`/api/params/presets/${id}`, { method: 'DELETE' }),
}

export const fileUrl = (jobId: string, path: string) => `/api/jobs/${jobId}/files/${path}`
