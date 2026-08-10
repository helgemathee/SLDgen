/** Wire shapes, mirroring sldgen_api/app.py. */

export type JobState =
  | 'queued'
  | 'running'
  | 'waiting'
  | 'paused'
  | 'complete'
  | 'failed'
  | 'deleting'

export const JOB_STATES: JobState[] = [
  'queued',
  'running',
  'waiting',
  'paused',
  'complete',
  'failed',
  'deleting',
]

export type ErrorClass = 'validation' | 'environment' | 'oom' | 'interrupted' | 'unknown'

/** A parameter value as the service stores it. `null` means "flag omitted". */
export type ParamValue = string | number | boolean | number[] | string[] | null

export type Params = Record<string, ParamValue>

export interface JobSummary {
  id: string
  title: string | null
  state: JobState
  desired_state: 'run' | 'pause' | 'delete'
  num_iter: number
  target_epoch: number
  current_epoch: number
  progress: number
  resolved_caption: string | null
  target_sha256: string
  parent_job_id: string | null
  batch_id: string | null
  priority: number
  error_class: ErrorClass | null
  error_message: string | null
  disk_bytes: number | null
  created_at: string
  updated_at: string
  started_at: string | null
  finished_at: string | null
  preview_url: string
  /** The frame the job is parked on, or null while it follows the newest one. */
  viewed_epoch: number | null
  favorite_count: number
  /** Only present when the list was fetched with `with_params`. */
  params?: Params
}

export interface Segment {
  id: number
  job_id: string
  seq: number
  start_epoch: number
  stop_at: number
  end_epoch: number | null
  resume_from: string | null
  argv_json: string
  operational_diff_json: string | null
  pid: number | null
  boot_id: string | null
  exit_code: number | null
  error_class: ErrorClass | null
  log_path: string | null
  started_at: string
  finished_at: string | null
}

export interface JobInput {
  id: number
  job_id: string
  role: 'avoid' | 'attract' | 'init_points' | 'stipple_weight' | 'labels'
  ordinal: number
  source_kind: 'job' | 'partition' | 'upload'
  source_job_id: string | null
  source_partition_id: string | null
  stored_path: string
  source_sha256: string
}

export interface Artifact {
  name: string
  path: string
  bytes: number
  kind: 'svg' | 'image' | 'video' | 'json' | 'log' | 'checkpoint' | 'csv' | 'other'
}

/** The SLDgen heartbeat (Spec 2 SS2.2), read straight from state.json. */
export interface StateJson {
  epoch?: number
  num_iter?: number
  stop_at?: number
  phase?: 'init' | 'optimizing' | 'finalizing' | 'done'
  iters_per_sec?: number
  latest_checkpoint?: string
  latest_preview?: string
  resolved_caption?: string
  updated_at?: string
}

export interface JobDetail extends JobSummary {
  params: Params
  structural_params: Params
  operational_params: Params
  segments: Segment[]
  inputs: JobInput[]
  artifacts: Artifact[]
  state_json: StateJson | null
  command: string
  favorite_epochs: number[]
}

export interface Favorite {
  epoch: number
  created_at: string
  png_url: string | null
  svg_url: string | null
}

export interface FavoritesResponse {
  job_id: string
  favorites: Favorite[]
}

export interface Frame {
  epoch: number
  png: string
  png_url: string
  svg: string | null
  svg_url: string | null
  bytes: number
}

export interface FramesResponse {
  job_id: string
  frames: Frame[]
  save_interval: number | null
  /** True when svg_logs/ is in a different coordinate space than final_sld.svg. */
  rescaled: boolean
  final_svg_url: string | null
  video_url: string | null
}

export interface Lineage {
  id: string
  parent: JobSummary | null
  variants: JobSummary[]
  batch_id: string | null
  batch_siblings: JobSummary[]
}

export interface Health {
  ok: boolean
  worker_alive: boolean
  gpu_free_mb: number | null
  db_ok: boolean
  root: string
}

export interface DiskReport {
  total_bytes: number
  by_category: Record<string, number>
  by_job: { job_id: string; title: string | null; bytes: number }[]
}

export interface LogChunk {
  from: number
  to: number
  text: string
  eof: boolean
  size: number
  running: boolean
  segment?: number
}

export interface Partition {
  id: string
  source_job_id: string
  source_svg: string
  strategy: string
  n: number
  params: Record<string, unknown>
  output_dir: string
  created_at: string
  svgs?: string[]
  preview?: string | null
}

export interface PartitionPreview {
  source_job_id: string
  strategy: string
  n: number
  preview_url: string | null
  svg_urls: string[]
  svgs: string[]
}

export interface Preset {
  id: string
  name: string
  params: unknown
  created_at: string
}

export interface CleanupResult {
  action: string
  dry_run: boolean
  job_count: number
  bytes: number
  items: { id: string; title: string | null; bytes: number }[]
}

export interface UploadResult {
  sha256: string
  width: number | null
  height: number | null
  bytes: number
  filename: string | null
  url: string
}

/** Payload of the per-job SSE `progress` event. */
export interface ProgressEvent {
  id: string
  state: JobState
  desired_state: string
  epoch: number
  target_epoch: number
  num_iter: number
  phase: StateJson['phase'] | null
  iters_per_sec: number | null
  resolved_caption: string | null
  preview_url: string
  error_class: ErrorClass | null
}

/** Payload of the global SSE `jobs` event. */
export interface JobsEvent {
  jobs: JobSummary[]
  worker_alive: boolean
  queue_depth: number
}
