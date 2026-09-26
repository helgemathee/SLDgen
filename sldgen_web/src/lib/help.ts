import type { ParamSection, ParamSpec } from './params'

/**
 * Hover text for the new-job form: one short, plain-language explanation per
 * parameter, section and panel control. Kept out of PARAM_SPECS so that literal
 * stays the compact server mirror `test_service_web.py` parses; `help.test.ts`
 * checks every parameter has an entry here.
 *
 * Written for someone who knows what the drawing should look like, not how the
 * optimiser works: say what the knob changes in the picture, and what higher
 * and lower do.
 */
export const PARAM_HELP: Record<string, string> = {
  caption:
    'What the picture shows, in words. The diffusion model uses it to understand the subject. Leave empty and BLIP-2 writes one from the image.',

  n_control_points:
    'How many points shape the line at the start. More gives more detail and a busier line; fewer gives a simpler, more abstract one. Points the run no longer needs are pruned along the way.',
  init_method:
    'How the starting line is laid out. tsp: dots are scattered by where the image is dark and joined into one continuous tour (the default, and usually best). trefoil: starts from a knot shape. contour: starts from the subject’s outline.',
  width:
    'Stroke thickness in pixels. “optim” lets the thickness vary along the line and be learned. random and optim_random are accepted but not implemented — avoid them.',
  seed:
    'Random seed. The same settings with the same seed give nearly the same drawing; change it to get a variation.',
  render_size:
    'Size of the square canvas, in pixels, that the line is drawn and judged at. 512 matches the diffusion model; larger is slower, and pixel-based distances elsewhere are measured at this size.',
  lr:
    'Learning rate: how far the line may move each step. Higher changes faster and more wildly; lower is calmer but slower to settle.',
  fixed_endpoints:
    'Lock both ends of the line where they start, so several drawings can be joined end to end. Cannot be combined with Origin.',
  calligraphy:
    'For black-on-white letters or glyphs: the subject is taken as every non-white pixel instead of using background removal.',
  optimize_cp_weights:
    'Let each control point learn how strongly it pulls the line. This is what lets the sparse loss switch points off and simplify the drawing.',
  prune_low_weights:
    'Drop control points whose weight falls to almost nothing, so the final line uses fewer points.',
  object_size_ratio:
    'How much of the canvas the subject fills while drawing (0.75 = 75%). The result is scaled back to the original size at the end, so in-progress frames are in a different space than the final SVG.',
  sampling_rate:
    'How many points are sampled along the line to draw it. Higher gives a smoother rendering but is slower.',
  use_cpu: 'Run without the GPU. Extremely slow — only for debugging.',

  condition:
    'Which view of the photo the diffusion model is shown. depth: a depth map — guides volume and pose, leaves line placement free. canny: an edge map — keeps closer to outlines and detail.',
  conditioning_scale:
    'How strongly the depth or edge map constrains the diffusion model. Higher sticks closer to the photo’s structure; lower gives it more freedom.',
  lora_weight:
    'Strength of the single-line-drawing style (a LoRA added to the diffusion model). 0 turns the style off.',
  lora_model: 'Path to the style LoRA file on the worker. Empty means no LoRA.',

  origin:
    'Pin where the line starts, as a fraction of the canvas (0, 0 is top-left). Place it on the canvas above. Needs init method tsp; cannot be combined with Fixed endpoints.',
  avoid:
    'Earlier drawings (SVGs) this line should keep away from — for layering several lines without them running into each other.',
  avoidance_weight: 'How hard the line is pushed away from the Avoid drawings.',
  avoidance_distance:
    'How close, in canvas pixels, the line may come to an Avoid drawing before the push starts.',
  attract:
    'Drawings (SVGs) this line should follow — for example its own piece of a partitioned master drawing.',
  attraction_weight: 'How hard the line is pulled toward the Attract drawings and the Canny edges.',
  attraction_distance:
    'A free zone, in canvas pixels. Within this distance of a target there is no pull, so the line can move freely nearby; further out, it is pulled back.',
  attract_canny:
    'Find edges in the photo and pull the line onto them. Pairs well with depth: depth gives volume, the edges give features like eyes and mouth.',
  attract_canny_low:
    'Edge sensitivity, lower threshold. Edges weaker than this are ignored. Lower finds more edges.',
  attract_canny_high:
    'Edge sensitivity, upper threshold. Edges stronger than this are always kept; ones in between only if they connect to a strong edge.',
  attract_canny_blur:
    'Blur before finding edges (odd size, 0 is off). More blur ignores fine texture such as hair or fabric.',
  attract_canny_simplify:
    'How much the traced edges are smoothed out, in pixels. 0 keeps every wiggle.',
  attract_canny_min_length: 'Ignore edge fragments shorter than this many pixels — removes speckle.',
  attract_canny_max_points:
    'The most edge points used as targets. Keep it at or below Control points, or the pull drowns out the diffusion guidance.',
  init_points:
    'Start the line from this SVG’s points instead of scattering dots from the image. Needs init method tsp.',
  stipple_weight:
    'A grayscale map of where the starting dots go: white means more ink, black none. Only shapes the starting line — long runs tend to even it out. Needs init method tsp.',
  stipple_weight_mode:
    'multiply: your map adjusts the density the image already suggests. replace: your map alone decides it.',

  repulsion_loss_weight:
    'Pushes parts of the line apart where they come close, preventing clumps and strokes piling on top of each other.',
  sparse_loss_weight:
    'Pressure to switch control points off over the run. Higher gives a simpler, sparser drawing. Needs Optimise CP weights.',
  sparse_loss_type:
    'The shape of that pressure. 1 is standard; below 1 pushes harder toward switching points fully off; 0 uses a different variant that spreads the weights apart.',
  sparse_loss_progressive:
    '“linear” grows the sparse pressure from nothing to full over the run, so detail forms first and simplification comes later. Anything else applies full pressure from the start.',
  length_shortening_loss_weight:
    'Penalises total line length, so the line takes shorter routes and avoids needless detours.',
  image_loss:
    'Also compare the drawing with the photo itself, and blend that pull in with the diffusion guidance. Improves likeness at some cost of freedom.',
  image_loss_weight:
    'How much of each step’s push comes from the photo rather than the diffusion model (0.2 = 20%). With decay or ramp, the value the run ends on.',
  image_loss_schedule:
    'How that share changes over the run. constant: fixed. decay: strong early to lock the composition, then eases off. ramp: weak early, stronger later to pull the finished drawing back toward the photo.',
  image_loss_schedule_start:
    'The share at the start of the run (decay and ramp only). Empty uses 0.5 for decay and 0.05 for ramp.',
  image_loss_chamfer:
    'Pulls the line onto the photo’s edges: wherever the line goes, it should be near a real edge. It may leave edges out, but should not invent lines. A relative weight; 0 turns it off.',
  image_loss_pyramid:
    'Compares where the ink sits with where the photo is dark, at several zoom levels — about overall composition and balance, not exact lines. A little slower. A relative weight; 0 turns it off.',
  image_loss_landmark:
    'Makes sure the line passes near a few key points, such as eye corners, nose and mouth. Needs a landmarks file. A relative weight; 0 turns it off.',
  image_loss_target:
    'The edge map the chamfer term pulls toward. By default it is found in the photo during the run.',
  image_loss_canny_low:
    'Edge sensitivity, lower threshold. Edges weaker than this are ignored. Lower finds more edges.',
  image_loss_canny_high:
    'Edge sensitivity, upper threshold. Edges stronger than this are always kept; ones in between only if they connect to a strong edge.',
  image_loss_canny_blur:
    'Blur before finding edges (odd size, 0 is off). More blur ignores fine texture such as hair or fabric.',
  image_loss_curve_samples:
    'How many points along the line are checked against the edges and landmarks. More is more precise but slower. At most the Sampling rate.',
  image_loss_landmarks:
    'The landmark file: the key points of a face and how much each one matters.',
  aesthetic_predictor_model_path:
    'Model that scores how good the final drawing looks, written to metrics.json. Only reports — it does not steer the run.',

  num_iter:
    'Total length of the run, in steps. The sparse ramp and the fidelity schedule are laid out against it, so changing it later means a new job.',
  save_interval:
    'Save a frame (PNG and SVG) every this many steps. The filmstrip can only show these frames.',
  checkpoint_interval:
    'Save a resumable checkpoint every this many steps; 0 is off. A crash never costs more than this.',
  save_video: 'Put the frames together into sketch.mp4 at the end. Needs ffmpeg on the worker.',
  verbose: 'Log loss values and extra diagnostics while the run goes.',
  debug: 'Debug mode: shows the output of helper tools and extra detail, for troubleshooting.',
}

export const SECTION_HELP: Record<ParamSection, string> = {
  prompt: 'What the diffusion model is told the picture shows.',
  curve:
    'The line itself: how many points it has, how it starts, how thick it is and how it is optimised.',
  guidance:
    'How the diffusion model (Stable Diffusion 3.5 with ControlNet) sees the photo and steers the drawing, and how strongly the line-drawing style is applied.',
  constraints:
    'Optional extra rules: where the line starts, what it avoids or follows, and where the starting ink goes.',
  losses:
    'Extra pressures on the line besides the diffusion guidance: keep it apart, simple and short.',
  run: 'How long it runs and what gets saved along the way.',
}

/** Help for the controls that are not parameters: page sections and panel widgets. */
export const UI_HELP = {
  source: 'The photo to turn into a single-line drawing. Drop one in or pick a recent one.',
  prepare:
    'Optional clean-up before the run: remove distractions from the photo, or paint where the ink should go.',
  parameters: 'Everything that shapes the drawing. Presets save and reload a whole set.',
  preset: 'Load a saved set of parameters, replacing the current ones.',
  presetName: 'Name to save the current parameters under. The same name overwrites.',
  title: 'A name for the job in the list. Defaults to the image’s filename.',
  budget: 'How long the job is, and how much of it to run now.',
  horizon:
    'Total length of the run, in steps. The sparse ramp and the fidelity schedule are laid out against it, so changing it later means a new job.',
  runBudget:
    'Where this run stops. Stopping early gives a preview you can continue later — the same drawing, not a shorter job.',
  imageFidelity:
    'Compare the drawing with the photo itself and blend that pull into the diffusion guidance, for better likeness.',
  constraintsGroup: SECTION_HELP.constraints,
  strength: 'How much the photo comparison counts, and how that changes over the run.',
  terms:
    'The three ways the drawing is compared with the photo. Their numbers are relative to each other; 0 switches one off.',
  landmarks: 'Key points of the face the line should pass near, found automatically or uploaded.',
  extractLandmarks: 'Find the face in a previous run’s canvas image and mark its key points.',
  edgeMap: 'The outlines the chamfer term pulls the line toward.',
  edgeDerived: 'Find edges in the photo during the run, with the thresholds below.',
  edgePrepared:
    'Find edges here, with extra controls for dark or busy photos, and attach the result.',
  edgeFile: 'Use an edge map you already have: an upload, or one from an earlier run of this image.',
  claheClip:
    'Brightens detail hidden in shadows before finding edges (dark hair, for example). 0 is off; 2–4 is typical.',
  claheGrid:
    'How many tiles that brightening works in across the image. More tiles act more locally.',
  roi: 'Keep only edges inside this box, in canvas pixels (x0 y0 x1 y1) — for example the face, not the shoulders. Empty keeps everything.',
  keepSilhouette:
    'Put the subject’s outline back in when the shadow brightening has washed it out.',
  maskMode: 'What your selection on the canvas below is used for.',
  selectSimilar:
    'Click a colour to select everything similar to it. Shift-click adds, Alt-click subtracts.',
  brushAdd: 'Paint to add to the selection.',
  brushRemove: 'Paint to remove from the selection.',
  tolerance: 'How different a colour may be and still count as similar. Higher selects more.',
  contiguous:
    'On: only select similar colours connected to where you clicked. Off: select them anywhere in the image.',
  radius: 'Brush size, in pixels.',
  hardness: 'How sharp the brush edge is. 1 is a hard edge; lower fades out softly.',
  density: 'The ink density you paint: 1 is full ink, 0 holds the ink back entirely.',
  feather: 'Softens the edge of the selection, so it fades rather than cuts.',
  cannyAttraction: PARAM_HELP.attract_canny,
  constraintFiles: {
    avoid: PARAM_HELP.avoid,
    attract: PARAM_HELP.attract,
    init_points: PARAM_HELP.init_points,
    stipple_weight: PARAM_HELP.stipple_weight,
  } as Record<string, string>,
}

/** The full hover text for a parameter: its explanation, then its name in the job's params. */
export function paramTooltip(spec: ParamSpec): string {
  const help = PARAM_HELP[spec.name]
  return help ? `${help}\n\n(${spec.name})` : spec.name
}
