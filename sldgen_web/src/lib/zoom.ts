/**
 * Pan/zoom arithmetic for the artwork viewer (Spec 3 SS6.1).
 *
 * Kept out of the component so the awkward parts -- anchor invariance and the
 * fit computation -- can be tested without a DOM.
 *
 * Zooming is button-driven rather than wheel-driven on purpose: a trackpad's
 * two-finger scroll fires dozens of wheel events per gesture, so a viewer that
 * zooms on wheel jumps several hundred percent from a scroll the user meant as
 * a scroll. Discrete 10% steps are slower to drive and impossible to trigger by
 * accident, which is the trade the viewer wants.
 */

export interface View {
  scale: number
  x: number
  y: number
}

export interface Size {
  width: number
  height: number
}

export const MIN_SCALE = 0.1
export const MAX_SCALE = 24

/** One press of + or -. Multiplicative, so a step feels the same at any zoom. */
export const ZOOM_STEP = 1.1

/** Leave a little air around a fitted image so it reads as a whole object. */
const FIT_PADDING = 0.96

export function clampScale(scale: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale))
}

/**
 * Scale by `factor`, keeping the point (anchorX, anchorY) -- in viewport
 * coordinates -- over the same pixel of the content.
 *
 * The stage's `transform-origin` is `0 0`, so the content point under the
 * anchor is `(anchor - offset) / scale`; holding that constant across the
 * change gives the offset below.
 */
export function zoomAbout(view: View, factor: number, anchorX: number, anchorY: number): View {
  const scale = clampScale(view.scale * factor)
  // Clamping can make this a no-op at the ends of the range; the ratio then
  // comes out 1 and the offsets are unchanged, which is what we want.
  const ratio = scale / view.scale
  return {
    scale,
    x: anchorX - (anchorX - view.x) * ratio,
    y: anchorY - (anchorY - view.y) * ratio,
  }
}

/** Zoom about the middle of the viewport -- what the +/- buttons do. */
export function zoomCentered(view: View, factor: number, viewport: Size): View {
  return zoomAbout(view, factor, viewport.width / 2, viewport.height / 2)
}

/**
 * The view that shows the whole of `content` centred in `viewport`.
 *
 * Returns null when either box has no area yet -- an image that has not loaded,
 * or a container measured before layout. The caller keeps its current view
 * rather than snapping to something meaningless.
 */
export function fitView(content: Size, viewport: Size): View | null {
  if (
    !(content.width > 0) ||
    !(content.height > 0) ||
    !(viewport.width > 0) ||
    !(viewport.height > 0)
  ) {
    return null
  }
  const scale = clampScale(
    Math.min(viewport.width / content.width, viewport.height / content.height) * FIT_PADDING,
  )
  return {
    scale,
    x: (viewport.width - content.width * scale) / 2,
    y: (viewport.height - content.height * scale) / 2,
  }
}

/** The view at actual size, centred -- the `100%` button. */
export function actualSizeView(content: Size, viewport: Size): View {
  if (!(content.width > 0) || !(content.height > 0)) {
    return { scale: 1, x: 0, y: 0 }
  }
  return {
    scale: 1,
    x: (viewport.width - content.width) / 2,
    y: (viewport.height - content.height) / 2,
  }
}
