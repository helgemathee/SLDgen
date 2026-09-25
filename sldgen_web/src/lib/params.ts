import type { ParamValue, Params } from '../api/types'

/**
 * A mirror of `sldgen_service/params.py`'s PARAM_SPECS, plus the presentation
 * facts the server has no opinion about: which section a parameter belongs to,
 * what to call it, and what to say about it.
 *
 * This file duplicates the server's defaults on purpose and accepts the risk
 * that comes with it. The alternative -- fetching the schema -- would make the
 * new-job form unusable until a round trip completed, and the server remains the
 * authority regardless: it canonicalises and validates every submission, so a
 * drift here produces a rejected submission with a clear message rather than a
 * job that ran with something other than what was shown. `test_service_web.py`
 * parses this file and asserts the two lists agree, name for name and default
 * for default, so the drift is caught by the test suite rather than by a user.
 */

export type ParamKind =
  | 'int'
  | 'float'
  | 'str'
  | 'float_or_str'
  | 'true_flag'
  | 'false_flag'
  | 'float_pair'
  | 'path'
  | 'path_list'

export type ParamSection = 'prompt' | 'curve' | 'guidance' | 'constraints' | 'losses' | 'run'

export interface ParamSpec {
  name: string
  kind: ParamKind
  group: 'structural' | 'operational'
  section: ParamSection
  default: ParamValue
  label: string
  hint?: string
  /** Nullable flags: the UI renders these with an enable toggle (SS8.3). */
  optional?: boolean
  /** Values the field is limited to. */
  choices?: string[]
  step?: number
  min?: number
  max?: number
  /** Set by an input picker rather than typed (SS4.3: inputs are copied). */
  viaInput?: boolean
}

export const PARAM_SPECS: ParamSpec[] = [
  {
    name: 'caption',
    kind: 'str',
    group: 'structural',
    section: 'prompt',
    default: '',
    label: 'Caption',
    hint:
      'Leave empty and BLIP-2 captions the image itself. The caption it chose appears on the job once it starts.',
  },

  { name: 'n_control_points', kind: 'int', group: 'structural', section: 'curve', default: 385, label: 'Control points', min: 3 },
  {
    name: 'init_method',
    kind: 'str',
    group: 'structural',
    section: 'curve',
    default: 'tsp',
    label: 'Init method',
    choices: ['tsp', 'trefoil', 'contour'],
    hint: 'Origin, init points and stipple weight all need tsp.',
  },
  {
    name: 'width',
    kind: 'float_or_str',
    group: 'structural',
    section: 'curve',
    default: 1.0,
    label: 'Width',
    hint: 'A number, or random / optim / optim_random.',
  },
  { name: 'seed', kind: 'int', group: 'structural', section: 'curve', default: 0, label: 'Seed' },
  { name: 'render_size', kind: 'int', group: 'structural', section: 'curve', default: 512, label: 'Render size', min: 64 },
  { name: 'lr', kind: 'float', group: 'structural', section: 'curve', default: 0.8, label: 'Learning rate', step: 0.05 },
  { name: 'fixed_endpoints', kind: 'true_flag', group: 'structural', section: 'curve', default: false, label: 'Fixed endpoints', hint: 'Mutually exclusive with origin.' },
  { name: 'calligraphy', kind: 'true_flag', group: 'structural', section: 'curve', default: false, label: 'Calligraphy' },
  { name: 'optimize_cp_weights', kind: 'false_flag', group: 'structural', section: 'curve', default: true, label: 'Optimise CP weights' },
  { name: 'prune_low_weights', kind: 'false_flag', group: 'structural', section: 'curve', default: true, label: 'Prune low weights' },
  { name: 'object_size_ratio', kind: 'float', group: 'structural', section: 'curve', default: 0.75, label: 'Object size ratio', step: 0.05, hint: 'Rescaling puts intermediate SVGs in a different space than the final one.' },
  { name: 'sampling_rate', kind: 'int', group: 'structural', section: 'curve', default: 5000, label: 'Sampling rate' },
  { name: 'use_cpu', kind: 'true_flag', group: 'structural', section: 'curve', default: false, label: 'Use CPU' },

  { name: 'condition', kind: 'str', group: 'structural', section: 'guidance', default: 'depth', label: 'Condition', choices: ['depth', 'canny'] },
  { name: 'conditioning_scale', kind: 'float', group: 'structural', section: 'guidance', default: 0.5, label: 'Conditioning scale', step: 0.05, min: 0 },
  { name: 'lora_weight', kind: 'float', group: 'structural', section: 'guidance', default: 0.1, label: 'LoRA weight', step: 0.05, min: 0 },
  { name: 'lora_model', kind: 'str', group: 'structural', section: 'guidance', default: './SLDgen/guidance/sld-lora.safetensors', label: 'LoRA model' },

  {
    name: 'origin',
    kind: 'float_pair',
    group: 'structural',
    section: 'constraints',
    default: null,
    label: 'Origin',
    optional: true,
    hint: 'Normalised (x, y). Placed on the prep canvas. Needs init method tsp.',
  },
  { name: 'avoid', kind: 'path_list', group: 'structural', section: 'constraints', default: null, label: 'Avoid', optional: true, viaInput: true },
  { name: 'avoidance_weight', kind: 'float', group: 'structural', section: 'constraints', default: 0.004, label: 'Avoidance weight', step: 0.001 },
  { name: 'avoidance_distance', kind: 'float', group: 'structural', section: 'constraints', default: 25.0, label: 'Avoidance distance' },
  { name: 'attract', kind: 'path_list', group: 'structural', section: 'constraints', default: null, label: 'Attract', optional: true, viaInput: true },
  { name: 'attraction_weight', kind: 'float', group: 'structural', section: 'constraints', default: 0.004, label: 'Attraction weight', step: 0.001 },
  { name: 'attraction_distance', kind: 'float', group: 'structural', section: 'constraints', default: 25.0, label: 'Attraction distance' },
  { name: 'attract_canny', kind: 'true_flag', group: 'structural', section: 'constraints', default: false, label: 'Attract to Canny edges', hint: 'The run derives an edge map from the target and pulls the curve onto it. Generated in canvas space during the run and saved as attract_canny.svg; adds to any Attract files above.' },
  { name: 'attract_canny_low', kind: 'float', group: 'structural', section: 'constraints', default: 100.0, label: 'Canny low' },
  { name: 'attract_canny_high', kind: 'float', group: 'structural', section: 'constraints', default: 200.0, label: 'Canny high' },
  { name: 'attract_canny_blur', kind: 'int', group: 'structural', section: 'constraints', default: 3, label: 'Canny blur', min: 0, hint: 'Odd kernel, 0 disables. Raise it on portraits so hair texture does not eat the budget.' },
  { name: 'attract_canny_simplify', kind: 'float', group: 'structural', section: 'constraints', default: 1.0, label: 'Canny simplify', step: 0.5, min: 0 },
  { name: 'attract_canny_min_length', kind: 'float', group: 'structural', section: 'constraints', default: 12.0, label: 'Canny min length', min: 0 },
  { name: 'attract_canny_max_points', kind: 'int', group: 'structural', section: 'constraints', default: 400, label: 'Canny max points', min: 2, hint: 'Keep at or below the control-point count: the attraction loss sums over every target.' },
  { name: 'init_points', kind: 'path', group: 'structural', section: 'constraints', default: null, label: 'Init points', optional: true, viaInput: true },
  { name: 'stipple_weight', kind: 'path', group: 'structural', section: 'constraints', default: null, label: 'Stipple weight', optional: true, viaInput: true },
  { name: 'stipple_weight_mode', kind: 'str', group: 'structural', section: 'constraints', default: 'multiply', label: 'Stipple weight mode', choices: ['multiply', 'replace'] },

  { name: 'repulsion_loss_weight', kind: 'float', group: 'structural', section: 'losses', default: 0.004, label: 'Repulsion', step: 0.001 },
  { name: 'sparse_loss_weight', kind: 'float', group: 'structural', section: 'losses', default: 2000.0, label: 'Sparse' },
  { name: 'sparse_loss_type', kind: 'float', group: 'structural', section: 'losses', default: 1.0, label: 'Sparse type', step: 0.1 },
  { name: 'sparse_loss_progressive', kind: 'str', group: 'structural', section: 'losses', default: 'linear', label: 'Sparse ramp' },
  { name: 'length_shortening_loss_weight', kind: 'float', group: 'structural', section: 'losses', default: 0.1, label: 'Length shortening', step: 0.05 },
  { name: 'image_loss', kind: 'true_flag', group: 'structural', section: 'losses', default: false, label: 'Image fidelity', hint: 'Blend a pull toward the target\'s edges into the SDS gradient, at a fixed share of its strength. Writes image_loss_target.png and image_loss_log.csv.' },
  { name: 'image_loss_weight', kind: 'float', group: 'structural', section: 'losses', default: 0.2, label: 'Fidelity weight', step: 0.05, min: 0.05, max: 1, hint: 'Share of the control-point gradient from the image. With decay/ramp, where the run ends.' },
  { name: 'image_loss_schedule', kind: 'str', group: 'structural', section: 'losses', default: 'constant', label: 'Fidelity schedule', choices: ['constant', 'decay', 'ramp'] },
  { name: 'image_loss_schedule_start', kind: 'float', group: 'structural', section: 'losses', default: null, label: 'Fidelity start', step: 0.05, min: 0, max: 1, hint: 'Weight at epoch 0 for decay/ramp. Empty: 0.5 for decay, 0.05 for ramp.' },
  { name: 'image_loss_chamfer', kind: 'float', group: 'structural', section: 'losses', default: 1.0, label: 'Chamfer term', step: 0.1, min: 0 },
  { name: 'image_loss_pyramid', kind: 'float', group: 'structural', section: 'losses', default: 0.0, label: 'Pyramid term', step: 0.1, min: 0 },
  { name: 'image_loss_landmark', kind: 'float', group: 'structural', section: 'losses', default: 0.0, label: 'Landmark term', step: 0.1, min: 0 },
  { name: 'image_loss_target', kind: 'path', group: 'structural', section: 'losses', default: null, label: 'Edge target', optional: true, viaInput: true },
  { name: 'image_loss_canny_low', kind: 'float', group: 'structural', section: 'losses', default: 100.0, label: 'Edge Canny low' },
  { name: 'image_loss_canny_high', kind: 'float', group: 'structural', section: 'losses', default: 200.0, label: 'Edge Canny high' },
  { name: 'image_loss_canny_blur', kind: 'int', group: 'structural', section: 'losses', default: 3, label: 'Edge Canny blur', min: 0 },
  { name: 'image_loss_curve_samples', kind: 'int', group: 'structural', section: 'losses', default: 2000, label: 'Curve samples', min: 2 },
  { name: 'image_loss_landmarks', kind: 'path', group: 'structural', section: 'losses', default: null, label: 'Landmarks', optional: true, viaInput: true },
  { name: 'aesthetic_predictor_model_path', kind: 'str', group: 'structural', section: 'losses', default: './SLDgen/metrics/aesthetic_predictor_v2_5.pth', label: 'Aesthetic predictor' },

  { name: 'num_iter', kind: 'int', group: 'structural', section: 'run', default: 4000, label: 'Horizon (num_iter)', min: 1, hint: 'Sets the schedule for every iteration. Changing it means a new job, not a promotion.' },
  { name: 'save_interval', kind: 'int', group: 'operational', section: 'run', default: 100, label: 'Save interval', min: 1, hint: 'Frames exist only at this granularity.' },
  { name: 'checkpoint_interval', kind: 'int', group: 'operational', section: 'run', default: 200, label: 'Checkpoint interval', min: 0, hint: 'A crash never costs more than this many iterations.' },
  { name: 'save_video', kind: 'false_flag', group: 'operational', section: 'run', default: false, label: 'Save mp4', hint: 'Off by default: the filmstrip scrubs the frames, and the mp4 needs ffmpeg on the worker host.' },
  { name: 'verbose', kind: 'true_flag', group: 'operational', section: 'run', default: false, label: 'Verbose' },
  { name: 'debug', kind: 'true_flag', group: 'operational', section: 'run', default: false, label: 'Debug' },
]

export const SPEC_BY_NAME: Record<string, ParamSpec> = Object.fromEntries(
  PARAM_SPECS.map((spec) => [spec.name, spec]),
)

export const SECTION_LABELS: Record<ParamSection, string> = {
  prompt: 'Prompt',
  curve: 'Curve',
  guidance: 'Guidance',
  constraints: 'Constraints',
  losses: 'Losses',
  run: 'Run',
}

export const OPERATIONAL_NAMES = new Set(
  PARAM_SPECS.filter((spec) => spec.group === 'operational').map((spec) => spec.name),
)

/** Every parameter at its default -- the starting point for a first-ever job. */
export function defaultParams(): Params {
  return Object.fromEntries(PARAM_SPECS.map((spec) => [spec.name, spec.default]))
}

/** Every parameter filled through an input role rather than set directly. */
export const INPUT_PARAMS = [
  'avoid',
  'attract',
  'init_points',
  'stipple_weight',
  'image_loss_target',
  'image_loss_landmarks',
] as const

/** Owned by ImageLossPanel, and hidden from the generic field lists. */
export const IMAGE_LOSS_PARAMS = PARAM_SPECS.filter((spec) =>
  spec.name.startsWith('image_loss'),
).map((spec) => spec.name)

/** Where decay/ramp start when image_loss_schedule_start is empty. */
export const IMAGE_LOSS_DEFAULT_START: Record<string, number> = { decay: 0.5, ramp: 0.05 }

/**
 * Drop the input-backed parameters.
 *
 * They hold service-owned paths to *copies* under `jobs/<id>/inputs/`, and
 * `create_job` rejects a submission that sets them directly, because doing so
 * would bypass both the copy and the provenance record (Spec 2 SS4.3). A new
 * job re-declares its inputs and the service fills these in.
 */
export function withoutInputPaths(params: Params): Params {
  const stripped = { ...params }
  for (const name of INPUT_PARAMS) {
    delete stripped[name]
  }
  return stripped
}

/** Format a value for the monospace parameter table. */
export function formatParamValue(value: ParamValue): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  if (Array.isArray(value)) return value.length ? value.join(', ') : '—'
  if (value === '') return '(auto)'
  return String(value)
}

/** Parse a form field back into the kind the API expects. */
export function coerceParam(spec: ParamSpec, raw: string | boolean | number[]): ParamValue {
  if (spec.kind === 'true_flag' || spec.kind === 'false_flag') return Boolean(raw)
  if (spec.kind === 'float_pair') {
    const pair = raw as number[]
    return [Number(pair[0]), Number(pair[1])]
  }
  if (spec.kind === 'int') {
    const parsed = parseInt(String(raw), 10)
    return Number.isNaN(parsed) ? (spec.default as ParamValue) : parsed
  }
  if (spec.kind === 'float') {
    const parsed = parseFloat(String(raw))
    return Number.isNaN(parsed) ? (spec.default as ParamValue) : parsed
  }
  if (spec.kind === 'float_or_str') {
    const text = String(raw).trim()
    const parsed = parseFloat(text)
    return !Number.isNaN(parsed) && String(parsed) === text ? parsed : text
  }
  return String(raw)
}

/**
 * Client-side mirror of `validate_params`. Not a substitute for the server's
 * check -- it is the same rules stated early, so a bad combination is a greyed
 * submit button with a reason rather than a 400 after a round trip.
 */
export function validateParams(params: Params): string[] {
  const problems: string[] = []
  const num = (name: string) => Number(params[name])

  if (num('num_iter') <= 0) problems.push('Horizon must be greater than zero.')
  if (num('render_size') <= 0) problems.push('Render size must be greater than zero.')
  if (num('save_interval') <= 0) problems.push('Save interval must be greater than zero.')
  if (num('checkpoint_interval') < 0) problems.push('Checkpoint interval cannot be negative.')
  if (!['depth', 'canny'].includes(String(params.condition)))
    problems.push("Condition must be 'depth' or 'canny'.")
  if (!['multiply', 'replace'].includes(String(params.stipple_weight_mode)))
    problems.push("Stipple weight mode must be 'multiply' or 'replace'.")
  if (!['tsp', 'trefoil', 'contour'].includes(String(params.init_method)))
    problems.push("Init method must be 'tsp', 'trefoil' or 'contour'.")

  const width = params.width
  if (typeof width === 'string' && !['random', 'optim', 'optim_random'].includes(width))
    problems.push("Width must be a number or one of 'random', 'optim', 'optim_random'.")

  if (params.origin !== null && params.origin !== undefined) {
    if (params.fixed_endpoints) problems.push('Origin and fixed endpoints cannot both be set.')
    if (params.init_method !== 'tsp') problems.push("Origin needs init method 'tsp'.")
    const pair = params.origin as number[]
    if (!pair.every((value) => value >= 0 && value <= 1))
      problems.push('Origin coordinates must be between 0 and 1.')
  }

  if (params.attract_canny) {
    if (num('attract_canny_low') >= num('attract_canny_high'))
      problems.push('Canny low must be below Canny high.')
    if (num('attract_canny_max_points') < 2)
      problems.push('Canny max points must be at least 2.')
    if (num('attract_canny_blur') < 0) problems.push('Canny blur cannot be negative.')
  }

  if (params.image_loss) problems.push(...imageLossProblems(params))

  for (const name of ['init_points', 'stipple_weight']) {
    if (params[name] != null && params.init_method !== 'tsp')
      problems.push(`${SPEC_BY_NAME[name].label} needs init method 'tsp'.`)
  }
  return problems
}

/** Mirror of `params.py::_validate_image_loss`, only consulted with the gate on. */
function imageLossProblems(params: Params): string[] {
  const problems: string[] = []
  const num = (name: string) => Number(params[name])
  const weight = num('image_loss_weight')
  const schedule = String(params.image_loss_schedule)
  const start = params.image_loss_schedule_start
  if (!(weight > 0 && weight <= 1)) problems.push('Fidelity weight must be above 0 and at most 1.')
  if (!['constant', 'decay', 'ramp'].includes(schedule))
    problems.push("Fidelity schedule must be 'constant', 'decay' or 'ramp'.")
  if (start != null) {
    if (!(Number(start) >= 0 && Number(start) <= 1))
      problems.push('Fidelity start must be between 0 and 1.')
    if (schedule === 'constant') problems.push('Fidelity start only applies to decay and ramp.')
  }
  if (schedule in IMAGE_LOSS_DEFAULT_START) {
    const begin = start == null ? IMAGE_LOSS_DEFAULT_START[schedule] : Number(start)
    if (schedule === 'decay' && !(begin > weight))
      problems.push('Decay needs a start above the fidelity weight.')
    if (schedule === 'ramp' && !(begin < weight))
      problems.push('Ramp needs a start below the fidelity weight.')
  }
  const terms = ['image_loss_chamfer', 'image_loss_pyramid', 'image_loss_landmark'].map(num)
  if (terms.some((value) => value < 0)) problems.push('Fidelity terms cannot be negative.')
  if (!terms.some((value) => value > 0)) problems.push('At least one fidelity term must be above 0.')
  if (num('image_loss_canny_low') >= num('image_loss_canny_high'))
    problems.push('Edge Canny low must be below Edge Canny high.')
  if (num('image_loss_canny_blur') < 0) problems.push('Edge Canny blur cannot be negative.')
  const samples = num('image_loss_curve_samples')
  if (!(samples >= 2 && samples <= num('sampling_rate')))
    problems.push('Curve samples must be between 2 and the sampling rate.')
  return problems
}
